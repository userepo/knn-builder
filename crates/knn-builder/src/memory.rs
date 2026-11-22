//! The memory the metrics report: the process's resident memory, read
//! from `/proc` at each scrape, and the GPU's, read by a thread of its own
//! once every so often, since a build under way reports nothing of it.

use std::fs;
use std::io;
use std::sync::mpsc::{self, RecvTimeoutError, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crate::metrics::GpuMemory;
use crate::{BuildError, Builder, GraphParams, Vectors};

/// The panic of a thread that held the meter's lock: a bug.
const POISONED: &str = "a thread panicked holding the GPU meter";

/// Readings missed in a row, at the idle pace, before the last is no
/// longer reported.
const MISSED_READINGS: u32 = 5;

/// The least time the last reading is reported for, so that a reader the
/// host keeps waiting does not drop the GPU from the page.
const LEAST_KEPT: Duration = Duration::from_secs(1);

/// The process's resident memory (`VmRSS`), in bytes; `None` where
/// `/proc/self/status` does not give it, as off Linux.
#[must_use]
pub fn resident() -> Option<u64> {
    vm_rss(&fs::read_to_string("/proc/self/status").ok()?)
}

/// The `VmRSS` of a `/proc/<pid>/status` text, in bytes; the kernel
/// writes it in KiB.
fn vm_rss(status: &str) -> Option<u64> {
    let line = status
        .lines()
        .find_map(|line| line.strip_prefix("VmRSS:"))?;
    let mut words = line.split_whitespace();
    let kib: u64 = words.next()?.parse().ok()?;
    (words.next() == Some("kB")).then(|| kib.saturating_mul(1024))
}

/// One reading of the device's memory, in bytes, every process on it
/// counted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reading {
    /// The device's memory.
    pub total: u64,
    /// Free.
    pub free: u64,
}

/// How often a [`GpuMeter`] reads; by default once a second, and every
/// 50 ms during a build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pace {
    /// With no build under way: how fresh the page's free memory is.
    pub idle: Duration,
    /// During a build, whose memory peaks for a fraction of a second: the
    /// longest peak the meter can miss.
    pub building: Duration,
}

impl Default for Pace {
    fn default() -> Self {
        Self {
            // A scrape every 15 s or more sees it fresh.
            idle: Duration::from_secs(1),
            building: Duration::from_millis(50),
        }
    }
}

/// The GPU's memory as a thread reads it, and the most in use during
/// each build, which [`Metered`] marks.
///
/// The thread reads at its [`Pace`], so a build's peak shorter than the
/// pace during builds may be missed. Once readings have failed for a few
/// idle intervals, the meter reports none, rather than the last as if it
/// were current. The thread ends once the last clone of its meter is
/// dropped.
#[derive(Debug, Clone)]
pub struct GpuMeter {
    shared: Arc<Shared>,
}

#[derive(Debug)]
struct Shared {
    readings: Mutex<Readings>,
    /// Wakes the reader as a build starts; dropped, it ends the reader.
    wake: SyncSender<()>,
}

/// What the meter knows.
#[derive(Debug)]
struct Readings {
    /// The last reading, and when it was taken.
    last: Option<(Reading, Instant)>,
    /// How long the last reading is reported for.
    kept: Duration,
    /// Builds started: a reading taken before the one under way started is
    /// not that build's.
    builds: u64,
    /// The least free during the build under way, as far as known.
    least_free: Option<u64>,
    /// Whether a build is under way.
    building: bool,
    /// The most in use during the last build that ended; 0 when it is
    /// not known.
    last_build_peak: u64,
}

/// How long a reading is reported for, read once every `idle`.
fn kept_for(idle: Duration) -> Duration {
    idle.saturating_mul(MISSED_READINGS).max(LEAST_KEPT)
}

impl GpuMeter {
    /// Starts the thread, calling `read` at `pace`; a reading of `None`
    /// is a failed one.
    ///
    /// # Errors
    ///
    /// When the thread cannot be started.
    #[expect(
        clippy::missing_panics_doc,
        reason = "the lock is taken on the thread it starts, not here"
    )]
    pub fn start<R>(mut read: R, pace: Pace) -> io::Result<Self>
    where
        R: FnMut() -> Option<Reading> + Send + 'static,
    {
        // One wake waiting is enough: the reader reads once for any number.
        let (wake, woken) = mpsc::sync_channel(1);
        let shared = Arc::new(Shared {
            readings: Mutex::new(Readings::new(kept_for(pace.idle))),
            wake,
        });
        let weak = Arc::downgrade(&shared);
        thread::Builder::new()
            .name("gpu-memory".to_owned())
            .spawn(move || {
                // The meter's clones hold the readings; once they are gone,
                // nothing reads what this thread would write.
                while let Some(shared) = weak.upgrade() {
                    let builds = shared.readings.lock().expect(POISONED).builds;
                    // Read without the lock, which a build would wait for.
                    let reading = read();
                    let mut readings = shared.readings.lock().expect(POISONED);
                    if let Some(reading) = reading {
                        readings.record(reading, builds, Instant::now());
                    }
                    let wait = if readings.building {
                        pace.building
                    } else {
                        pace.idle
                    };
                    drop(readings);
                    drop(shared);
                    // Until the next reading is due, a build starts, or
                    // the last meter is dropped.
                    if woken.recv_timeout(wait) == Err(RecvTimeoutError::Disconnected) {
                        return;
                    }
                }
            })?;
        Ok(Self { shared })
    }

    /// The memory last read, and the peak of the last build that ended;
    /// `None` before the first reading, and once readings have failed for
    /// a while.
    ///
    /// # Panics
    ///
    /// After a thread panicked holding the meter's readings, a bug.
    #[must_use]
    pub fn memory(&self) -> Option<GpuMemory> {
        let readings = self.shared.readings.lock().expect(POISONED);
        readings.memory(Instant::now())
    }

    /// Marks a build under way, until the guard is dropped; `free` is the
    /// device's free memory read as it starts, where the builder can tell.
    fn building(&self, free: Option<u64>) -> Building<'_> {
        let mut readings = self.shared.readings.lock().expect(POISONED);
        readings.start(free, Instant::now());
        drop(readings);
        // To the pace of a build at once. A wake already waiting, or a
        // reader that has ended, needs none.
        _ = self.shared.wake.try_send(());
        Building {
            readings: &self.shared.readings,
        }
    }
}

impl Readings {
    fn new(kept: Duration) -> Self {
        Self {
            last: None,
            kept,
            builds: 0,
            least_free: None,
            building: false,
            last_build_peak: 0,
        }
    }

    fn record(&mut self, reading: Reading, builds: u64, now: Instant) {
        self.last = Some((reading, now));
        // A reading begun before this build started may hold the last
        // build's memory.
        if self.building && builds == self.builds {
            self.lower(reading.free);
        }
    }

    /// The last reading, while it is kept.
    fn current(&self, now: Instant) -> Option<Reading> {
        let (reading, at) = self.last?;
        (now.saturating_duration_since(at) <= self.kept).then_some(reading)
    }

    fn memory(&self, now: Instant) -> Option<GpuMemory> {
        let reading = self.current(now)?;
        Some(GpuMemory {
            total: reading.total,
            free: reading.free,
            last_build_peak: self.last_build_peak,
        })
    }

    fn start(&mut self, free: Option<u64>, now: Instant) {
        self.builds += 1;
        self.building = true;
        // Without a reading of its own, what was free when last read is
        // the least it can show: up to an interval old, so possibly taken
        // during the build before. One no longer reported shows nothing.
        self.least_free = free.or(self.current(now).map(|reading| reading.free));
    }

    fn end(&mut self) {
        self.building = false;
        // Nothing known, nothing to tell: not the peak of the build
        // before, which this one would pass for.
        self.last_build_peak = match (self.last, self.least_free.take()) {
            (Some((reading, _)), Some(least)) => reading.total.saturating_sub(least),
            _ => 0,
        };
    }

    fn lower(&mut self, free: u64) {
        self.least_free = Some(self.least_free.map_or(free, |least| least.min(free)));
    }
}

/// A build under way; dropped, it records the build's peak, a build that
/// panicked included.
struct Building<'a> {
    readings: &'a Mutex<Readings>,
}

impl Drop for Building<'_> {
    fn drop(&mut self) {
        // Not a panic during a panic: a poisoned lock leaves the peak.
        if let Ok(mut readings) = self.readings.lock() {
            readings.end();
        }
    }
}

/// A builder whose builds a [`GpuMeter`] watches.
#[derive(Debug)]
pub struct Metered<B> {
    builder: B,
    meter: GpuMeter,
}

impl<B> Metered<B> {
    /// `builder`, its builds marked on `meter`.
    #[must_use]
    pub fn new(builder: B, meter: GpuMeter) -> Self {
        Self { builder, meter }
    }
}

impl<B: Builder> Builder for Metered<B> {
    fn build(
        &mut self,
        vectors: &Vectors,
        dimension: usize,
        params: &GraphParams,
    ) -> Result<(Vec<u32>, usize), BuildError> {
        // Read before the build takes any, and after the one before let
        // its memory go.
        let free = self.builder.free_device_memory();
        let _building = self.meter.building(free);
        self.builder.build(vectors, dimension, params)
    }

    fn free_device_memory(&self) -> Option<u64> {
        self.builder.free_device_memory()
    }
}

#[cfg(test)]
mod tests {
    use std::panic::{self, AssertUnwindSafe};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{Duration, Instant};

    use super::{GpuMeter, Metered, Pace, Reading, Readings, kept_for, resident, vm_rss};
    use crate::job::tests::job;
    use crate::metrics::GpuMemory;
    use crate::{BuildError, Builder, FakeBuilder, GraphParams, Vectors};

    const GIB: u64 = 1 << 30;
    const MILLISECOND: Duration = Duration::from_millis(1);
    const HOUR: Duration = Duration::from_secs(3600);

    #[test]
    fn reads_the_resident_memory_in_bytes() {
        let status = "Name:\tknn-builder\nVmPeak:\t  900 kB\nVmRSS:\t    1234 kB\nThreads:\t9\n";
        assert_eq!(vm_rss(status), Some(1234 * 1024));
        assert_eq!(vm_rss("Name:\tkthreadd\n"), None);
        assert_eq!(vm_rss("VmRSS:\t12 MB\n"), None);
        assert_eq!(vm_rss("VmRSS:\t\n"), None);
        // Where the file is, it is read.
        assert_eq!(resident().is_some(), cfg!(target_os = "linux"));
    }

    /// A reading of 8 GiB with `free` of it free.
    fn reading(free: u64) -> Reading {
        Reading {
            total: 8 * GIB,
            free,
        }
    }

    /// A meter reading `free` of 8 GiB at `pace`; `None` while `free` is
    /// 0. Returns once it has read.
    fn meter(free: &Arc<AtomicU64>, pace: Pace) -> GpuMeter {
        let shared = Arc::clone(free);
        let read = move || {
            let free = shared.load(Ordering::SeqCst);
            (free > 0).then_some(reading(free))
        };
        let meter = GpuMeter::start(read, pace).unwrap();
        if free.load(Ordering::SeqCst) > 0 {
            seen(&meter, free.load(Ordering::SeqCst));
        }
        meter
    }

    /// Reading every millisecond, building or not.
    const FAST: Pace = Pace {
        idle: MILLISECOND,
        building: MILLISECOND,
    };

    /// Waits until `meter` has read `free`.
    fn seen(meter: &GpuMeter, free: u64) {
        let start = Instant::now();
        while meter.memory().map(|memory| memory.free) != Some(free) {
            assert!(
                start.elapsed() < Duration::from_secs(5),
                "never read {free}"
            );
            std::thread::sleep(MILLISECOND);
        }
    }

    #[test]
    fn keeps_the_least_free_during_a_build() {
        let free = Arc::new(AtomicU64::new(0));
        let meter = meter(&free, FAST);
        assert_eq!(meter.memory(), None);
        free.store(7 * GIB, Ordering::SeqCst);
        seen(&meter, 7 * GIB);
        let expect = |last_build_peak| GpuMemory {
            total: 8 * GIB,
            free: 7 * GIB,
            last_build_peak,
        };
        assert_eq!(meter.memory(), Some(expect(0)));
        let building = meter.building(None);
        for gib in [5, 2, 6, 7] {
            free.store(gib * GIB, Ordering::SeqCst);
            seen(&meter, gib * GIB);
        }
        drop(building);
        assert_eq!(meter.memory(), Some(expect(6 * GIB)));
        // Less free between builds is no build's.
        free.store(GIB, Ordering::SeqCst);
        seen(&meter, GIB);
        free.store(7 * GIB, Ordering::SeqCst);
        seen(&meter, 7 * GIB);
        drop(meter.building(None));
        assert_eq!(meter.memory(), Some(expect(GIB)));
    }

    #[test]
    fn reads_at_the_pace_of_a_build_once_one_starts() {
        let free = Arc::new(AtomicU64::new(7 * GIB));
        let pace = Pace {
            idle: HOUR,
            building: MILLISECOND,
        };
        // Read once, and not again for an hour.
        let meter = meter(&free, pace);
        free.store(6 * GIB, Ordering::SeqCst);
        std::thread::sleep(20 * MILLISECOND);
        assert_eq!(meter.memory().unwrap().free, 7 * GIB);
        // A build wakes the reader, which then keeps to the build's pace.
        let building = meter.building(Some(7 * GIB));
        for gib in [6, 3, 5] {
            free.store(gib * GIB, Ordering::SeqCst);
            seen(&meter, gib * GIB);
        }
        drop(building);
        assert_eq!(meter.memory().unwrap().last_build_peak, 5 * GIB);
        // Asleep for the hour or not, it ends with its meter, and drops
        // what it read with.
        drop(meter);
        let start = Instant::now();
        while Arc::strong_count(&free) > 1 {
            assert!(start.elapsed() < Duration::from_secs(5), "still reading");
            std::thread::sleep(MILLISECOND);
        }
    }

    #[test]
    fn starts_each_build_from_its_own_reading() {
        let at = Instant::now();
        let mut readings = Readings::new(Duration::from_secs(1));
        // Last read during a build before, which held 7 GiB.
        readings.record(reading(GIB), 0, at);
        readings.start(Some(6 * GIB), at);
        readings.end();
        assert_eq!(readings.last_build_peak, 2 * GIB);
        // Without a reading of its own, the last is all it has, while it
        // is kept.
        readings.start(None, at + Duration::from_secs(1));
        readings.end();
        assert_eq!(readings.last_build_peak, 7 * GIB);
        readings.start(None, at + Duration::from_secs(2));
        readings.record(
            reading(5 * GIB),
            readings.builds,
            at + Duration::from_secs(2),
        );
        readings.end();
        assert_eq!(readings.last_build_peak, 3 * GIB);
        // A build with no reading of its own, and the last one no longer
        // kept, does not pass the build before's peak for its own.
        readings.start(None, at + Duration::from_secs(10));
        readings.end();
        assert_eq!(readings.last_build_peak, 0);
        // A build with nothing read at all tells nothing.
        let mut unread = Readings::new(Duration::from_secs(1));
        unread.start(None, at);
        unread.end();
        unread.start(Some(GIB), at);
        unread.end();
        assert_eq!(unread.last_build_peak, 0);
    }

    #[test]
    fn leaves_a_reading_begun_before_the_build_out() {
        let at = Instant::now();
        let mut readings = Readings::new(Duration::from_secs(1));
        readings.record(reading(7 * GIB), 0, at);
        // Begun as the build before ran, recorded once this one started.
        let begun = readings.builds;
        readings.start(Some(6 * GIB), at);
        readings.record(reading(GIB), begun, at);
        readings.record(reading(5 * GIB), readings.builds, at);
        readings.end();
        assert_eq!(readings.last_build_peak, 3 * GIB);
    }

    #[test]
    fn reports_a_reading_only_while_it_is_kept() {
        assert_eq!(Pace::default().idle, Duration::from_secs(1));
        assert_eq!(Pace::default().building, Duration::from_millis(50));
        // Five idle intervals, and a second at the least.
        assert_eq!(kept_for(Duration::from_secs(1)), Duration::from_secs(5));
        assert_eq!(kept_for(Duration::from_secs(3)), Duration::from_secs(15));
        assert_eq!(kept_for(MILLISECOND), Duration::from_secs(1));
        assert_eq!(kept_for(Duration::MAX), Duration::MAX);
        let at = Instant::now();
        let mut readings = Readings::new(Duration::from_secs(5));
        assert_eq!(readings.memory(at), None);
        readings.record(reading(6 * GIB), 0, at);
        let free = |readings: &Readings, after| {
            let memory = readings.memory(at + after);
            memory.map(|memory| memory.free)
        };
        assert_eq!(free(&readings, Duration::from_secs(5)), Some(6 * GIB));
        assert_eq!(free(&readings, Duration::from_millis(5_001)), None);
        // And again once one is read.
        readings.record(reading(4 * GIB), 0, at + Duration::from_secs(9));
        assert_eq!(free(&readings, Duration::from_secs(10)), Some(4 * GIB));
    }

    #[test]
    fn marks_each_build_of_its_builder() {
        // Read with 6 GiB free throughout; the builder reads 5 as its
        // build starts, so that is the least the build shows.
        let free = Arc::new(AtomicU64::new(6 * GIB));
        let meter = meter(&free, FAST);
        let fake = FakeBuilder::new(Duration::ZERO).with_free_device_memory(5 * GIB);
        let mut metered = Metered::new(fake.clone(), meter.clone());
        let job = job("bucket", 10, 8);
        let params = job.graph_params().unwrap();
        let vectors = Vectors::F32(vec![0.0; 80]);
        let (graph, degree) = metered.build(&vectors, 8, &params).unwrap();
        assert_eq!(graph.len(), 10 * degree);
        assert_eq!(fake.builds(), 1);
        assert_eq!(metered.free_device_memory(), Some(5 * GIB));
        assert_eq!(meter.memory().unwrap().last_build_peak, 3 * GIB);
    }

    /// A builder whose builds panic, with 5 GiB free as each starts.
    struct Panics;

    impl Builder for Panics {
        #[expect(clippy::panic_in_result_fn, reason = "the panic is what it is for")]
        fn build(
            &mut self,
            _: &Vectors,
            _: usize,
            _: &GraphParams,
        ) -> Result<(Vec<u32>, usize), BuildError> {
            panic!("a build that panics");
        }

        fn free_device_memory(&self) -> Option<u64> {
            Some(5 * GIB)
        }
    }

    #[test]
    fn ends_a_build_that_panics() {
        let free = Arc::new(AtomicU64::new(6 * GIB));
        let meter = meter(&free, FAST);
        let mut metered = Metered::new(Panics, meter.clone());
        let params = job("bucket", 10, 8).graph_params().unwrap();
        let vectors = Vectors::F32(vec![0.0; 80]);
        let built = panic::catch_unwind(AssertUnwindSafe(|| metered.build(&vectors, 8, &params)));
        assert!(built.is_err());
        // Its peak recorded, and the meter back at the idle pace.
        assert_eq!(meter.memory().unwrap().last_build_peak, 3 * GIB);
        assert!(!meter.shared.readings.lock().unwrap().building);
    }

    #[test]
    fn stops_reading_once_dropped() {
        let reads = Arc::new(AtomicU64::new(0));
        let counter = Arc::clone(&reads);
        let read = move || {
            counter.fetch_add(1, Ordering::SeqCst);
            None
        };
        let meter = GpuMeter::start(read, FAST).unwrap();
        let clone = meter.clone();
        drop(meter);
        let start = Instant::now();
        while reads.load(Ordering::SeqCst) < 3 {
            assert!(start.elapsed() < Duration::from_secs(5));
            std::thread::sleep(MILLISECOND);
        }
        assert_eq!(clone.memory(), None);
        drop(clone);
        // The thread drops its reader as it ends.
        while Arc::strong_count(&reads) > 1 {
            assert!(start.elapsed() < Duration::from_secs(5), "still reading");
            std::thread::sleep(MILLISECOND);
        }
    }
}
