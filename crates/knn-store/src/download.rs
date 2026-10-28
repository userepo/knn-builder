use std::iter::Enumerate;
use std::pin::pin;
use std::slice::ChunksMut;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use aws_sdk_s3::types::ChecksumMode;
use tokio::sync::Notify;

use crate::Error;
use crate::store::{Blob, Store, body_error, s3_error};

/// Attempts at reading one GET's body. The SDK retries a request until
/// its response starts, not a body cut off midway; s3transfer, behind
/// the upstream service's downloads, gives a body five attempts.
const BODY_ATTEMPTS: u32 = 5;

/// Stops a download or an upload from another thread: a download's
/// threads stop at once, waiting for a response or a frame included, and
/// it returns [`Error::Canceled`]; an upload stops taking parts and is
/// aborted (see [`Store::upload`]). Clones share the flag.
#[derive(Debug, Clone, Default)]
pub struct Cancel(Arc<Flag>);

#[derive(Debug, Default)]
struct Flag {
    set: AtomicBool,
    /// Wakes the transfers waiting on [`Cancel::canceled`].
    notify: Notify,
}

impl Cancel {
    /// A flag not yet set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Stops the transfers that check this flag.
    pub fn cancel(&self) {
        self.0.set.store(true, Ordering::SeqCst);
        self.0.notify.notify_waiters();
    }

    /// Whether [`Cancel::cancel`] was called.
    #[must_use]
    pub fn is_canceled(&self) -> bool {
        self.0.set.load(Ordering::SeqCst)
    }

    /// Resolves once the flag is set: what a wait races to end early.
    pub async fn canceled(&self) {
        let mut notified = pin!(self.0.notify.notified());
        // Registered before the flag is read, so a cancel in between still
        // wakes it.
        notified.as_mut().enable();
        if !self.is_canceled() {
            notified.await;
        }
    }
}

impl Store {
    /// Downloads a blob of little-endian values, `N` bytes each, into
    /// `out`, one value per element, through `decode`: `f32::from_le_bytes`
    /// for fp32, or a conversion, such as fp32 to fp16. The values arrive
    /// in ranged GETs, each decoded into its own part of `out` as its body
    /// streams in, so the blob is never held twice. The GETs run on up to
    /// [`Transfers::download_concurrency`] threads of the download's own,
    /// one range at a time each, so decoding, 2 to 4 GB/s a core, spreads
    /// over the CPUs; the connections run on the store's runtime.
    ///
    /// `blob` comes from [`Store::head`]; every GET asks for its `ETag`.
    /// `cancel` stops the download early, a wait for S3 included.
    ///
    /// It blocks: call it from a blocking thread, such as one from
    /// `tokio::task::spawn_blocking`, never from async code, where it
    /// panics. The store's runtime must be multi-threaded, or driven by
    /// another thread meanwhile, to run the requests.
    ///
    /// [`Transfers::download_concurrency`]: crate::Transfers::download_concurrency
    ///
    /// # Errors
    ///
    /// Returns [`Error::Size`] when the blob's size is not `out`'s length
    /// times `N`, before any GET; [`Error::S3`] when a GET fails, the
    /// blob replaced since `blob` was read among other causes, or its body
    /// is cut off on every attempt; [`Error::Range`] when a body is longer
    /// or shorter than its range; and [`Error::Canceled`] once `cancel` is
    /// set. The first error stops the download once the ranges in progress
    /// end; `out` then holds some of the values.
    ///
    /// # Panics
    ///
    /// When called in an async context; when `decode` panics, once the
    /// other threads end their ranges; and, a bug, on a poisoned lock.
    pub fn download<T, D, const N: usize>(
        &self,
        bucket: &str,
        key: &str,
        blob: &Blob,
        out: &mut [T],
        decode: D,
        cancel: &Cancel,
    ) -> Result<(), Error>
    where
        T: Send,
        D: Fn([u8; N]) -> T + Copy + Send + Sync,
    {
        const { assert!(N > 0, "a value takes at least one byte") };
        // `Handle::block_on` panics in an async context. Calling it first
        // fails every download there alike; a blob of several ranges would
        // otherwise block a runtime thread for the whole transfer.
        self.runtime.block_on(async {});
        let expected = u64::try_from(out.len())
            .ok()
            .and_then(|values| values.checked_mul(N as u64));
        if expected != Some(blob.size) {
            return Err(Error::Size {
                bucket: bucket.to_owned(),
                key: key.to_owned(),
                expected: expected.unwrap_or(u64::MAX),
                actual: blob.size,
            });
        }
        let get = &Get {
            store: self,
            bucket,
            key,
            e_tag: blob.e_tag.as_deref(),
            cancel,
        };
        let per_range = (self.transfers.range_size / N).max(1);
        if out.len() <= per_range {
            return self.runtime.block_on(get.fill(None, out, decode));
        }
        let threads = self
            .transfers
            .download_concurrency
            .min(out.len().div_ceil(per_range));
        let work = &Mutex::new(Work {
            ranges: out.chunks_mut(per_range).enumerate(),
            error: None,
        });
        thread::scope(|scope| {
            for _ in 0..threads {
                scope.spawn(move || {
                    while let Some((index, part)) = Work::next(work) {
                        let offset = (index * per_range * N) as u64;
                        let fetched = self.runtime.block_on(get.fill(Some(offset), part, decode));
                        if let Err(error) = fetched {
                            Work::fail(work, error);
                        }
                    }
                });
            }
        });
        let work = work.lock().expect(POISONED);
        match &work.error {
            Some(error) => Err(error.clone()),
            None => Ok(()),
        }
    }
}

/// A download's lock is held only to take a range or record an error,
/// neither of which can panic; a poisoned lock is a bug.
const POISONED: &str = "download work lock poisoned: a holder panicked";

/// The ranges a download's threads share, and its first error.
struct Work<'a, T> {
    ranges: Enumerate<ChunksMut<'a, T>>,
    error: Option<Error>,
}

impl<'a, T> Work<'a, T> {
    /// The next range and its index, until the ranges run out or one
    /// fails.
    fn next(work: &Mutex<Self>) -> Option<(usize, &'a mut [T])> {
        let mut work = work.lock().expect(POISONED);
        if work.error.is_some() {
            return None;
        }
        work.ranges.next()
    }

    /// Records an error; the first one stands.
    fn fail(work: &Mutex<Self>, error: Error) {
        let mut work = work.lock().expect(POISONED);
        work.error.get_or_insert(error);
    }
}

/// The GETs of one blob.
struct Get<'a> {
    store: &'a Store,
    bucket: &'a str,
    key: &'a str,
    e_tag: Option<&'a str>,
    cancel: &'a Cancel,
}

impl Get<'_> {
    /// [`Error::Canceled`] once the download is canceled.
    fn check_cancel(&self) -> Result<(), Error> {
        if self.cancel.is_canceled() {
            Err(self.canceled())
        } else {
            Ok(())
        }
    }

    fn canceled(&self) -> Error {
        Error::Canceled {
            bucket: self.bucket.to_owned(),
            key: self.key.to_owned(),
        }
    }

    /// GETs the bytes `out` decodes from, at `offset`, or the whole blob
    /// without one, and decodes them into `out`. A cancel ends a wait for
    /// the response or for a frame; one between frames that arrive without
    /// a wait shows at the next frame.
    async fn fill<T, D, const N: usize>(
        &self,
        offset: Option<u64>,
        out: &mut [T],
        decode: D,
    ) -> Result<(), Error>
    where
        D: Fn([u8; N]) -> T + Copy,
    {
        let expected = (out.len() * N) as u64;
        let range_error = |received| Error::Range {
            bucket: self.bucket.to_owned(),
            key: self.key.to_owned(),
            offset: offset.unwrap_or(0),
            expected,
            received,
        };
        // One registration for every wait of this range. Racing each frame
        // against it costs nothing measurable: 6.2 to 6.8 GB/s over loopback
        // with 8 threads, with the race or without.
        let mut canceled = pin!(self.cancel.canceled());
        let mut attempt = 1;
        loop {
            self.check_cancel()?;
            let mut request = self
                .store
                .client
                .get_object()
                .bucket(self.bucket)
                .key(self.key)
                .set_if_match(self.e_tag.map(str::to_owned))
                .checksum_mode(ChecksumMode::Enabled);
            if let Some(offset) = offset {
                // A ranged GET is never empty: `out` has a value at least.
                let last = offset + expected.saturating_sub(1);
                request = request.range(format!("bytes={offset}-{last}"));
            }
            let response = tokio::select! {
                biased;
                response = request.send() => response,
                () = &mut canceled => return Err(self.canceled()),
            };
            let mut body = response
                .map_err(|e| s3_error("GetObject", self.bucket, self.key, &e))?
                .body;
            let mut values = Values::new(&mut *out, decode);
            let cut = loop {
                self.check_cancel()?;
                let next = tokio::select! {
                    biased;
                    next = body.next() => next,
                    () = &mut canceled => return Err(self.canceled()),
                };
                match next {
                    None => break None,
                    Some(Ok(frame)) => values.push(&frame).map_err(range_error)?,
                    Some(Err(e)) => break Some(e),
                }
            };
            match cut {
                None => return values.finish().map_err(range_error),
                Some(_) if attempt < BODY_ATTEMPTS => attempt += 1,
                Some(e) => return Err(body_error(self.bucket, self.key, &e)),
            }
        }
    }
}

/// Decodes little-endian values, `N` bytes each, into a slice, from
/// frames of any length: a value may straddle two frames.
struct Values<'a, T, D, const N: usize> {
    out: &'a mut [T],
    filled: usize,
    /// The first bytes of a value the last frame cut off.
    held: Vec<u8>,
    /// Bytes pushed.
    received: u64,
    decode: D,
}

impl<'a, T, D, const N: usize> Values<'a, T, D, N>
where
    D: Fn([u8; N]) -> T,
{
    fn new(out: &'a mut [T], decode: D) -> Self {
        Self {
            out,
            filled: 0,
            held: Vec::with_capacity(N),
            received: 0,
            decode,
        }
    }

    /// Decodes a frame. Returns the bytes received so far when they are
    /// more than `out` takes.
    fn push(&mut self, mut frame: &[u8]) -> Result<(), u64> {
        self.received += frame.len() as u64;
        if !self.held.is_empty() {
            let (head, rest) = frame.split_at((N - self.held.len()).min(frame.len()));
            self.held.extend_from_slice(head);
            frame = rest;
            if let Ok(value) = <[u8; N]>::try_from(self.held.as_slice()) {
                self.held.clear();
                self.put(&[value])?;
            }
        }
        let (whole, rest) = frame.as_chunks::<N>();
        self.put(whole)?;
        self.held.extend_from_slice(rest);
        Ok(())
    }

    fn put(&mut self, values: &[[u8; N]]) -> Result<(), u64> {
        let end = self.filled + values.len();
        let slots = self.out.get_mut(self.filled..end).ok_or(self.received)?;
        for (slot, &value) in slots.iter_mut().zip(values) {
            *slot = (self.decode)(value);
        }
        self.filled = end;
        Ok(())
    }

    /// Checks every value arrived whole. Returns the bytes received when
    /// not.
    fn finish(&self) -> Result<(), u64> {
        if self.filled == self.out.len() && self.held.is_empty() {
            Ok(())
        } else {
            Err(self.received)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::thread;
    use std::time::{Duration, Instant};

    use tokio::runtime::Runtime;

    use super::{Cancel, Values};
    use crate::testing::{FakeS3Server, Get, test_store};
    use crate::{Blob, Error, Store, Transfers};

    const E_TAG: &str = "\"fake-e-tag\"";

    fn le(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// A store on the fake, with the service's client settings.
    fn store(runtime: &Runtime, fake: &FakeS3Server, transfers: Transfers) -> Store {
        runtime.block_on(test_store(Some(fake), transfers))
    }

    /// Downloads every value of the fake's blob as fp32.
    fn download(runtime: &Runtime, store: &Store, values: usize) -> Result<Vec<f32>, Error> {
        let blob = runtime.block_on(store.head("bucket", "vectors.knnvec"))?;
        let mut out = vec![0.0; values];
        store.download(
            "bucket",
            "vectors.knnvec",
            &blob,
            &mut out,
            f32::from_le_bytes,
            &Cancel::new(),
        )?;
        Ok(out)
    }

    #[test]
    fn stops_when_canceled() {
        let values: Vec<f32> = (0..2_500_u16).map(f32::from).collect();
        let fake = FakeS3Server::serving(le(&values), 0);
        let runtime = Runtime::new().unwrap();
        // Ten ranges, one at a time.
        let transfers = Transfers {
            range_size: 1_000,
            download_concurrency: 1,
            ..Transfers::default()
        };
        let store = store(&runtime, &fake, transfers);
        let blob = runtime
            .block_on(store.head("bucket", "vectors.knnvec"))
            .unwrap();
        let canceled = Error::Canceled {
            bucket: "bucket".to_owned(),
            key: "vectors.knnvec".to_owned(),
        };

        // Canceled midway, from the first value decoded: the first range
        // is the only one asked for.
        let cancel = Cancel::new();
        let mut out = vec![0.0; values.len()];
        let decode = |b| {
            cancel.cancel();
            f32::from_le_bytes(b)
        };
        let result = store.download("bucket", "vectors.knnvec", &blob, &mut out, decode, &cancel);
        assert_eq!(result, Err(canceled.clone()));
        assert_eq!(fake.gets().len(), 1);

        // Canceled before it starts: no GET at all.
        let result = store.download(
            "bucket",
            "vectors.knnvec",
            &blob,
            &mut out,
            f32::from_le_bytes,
            &cancel,
        );
        assert_eq!(result, Err(canceled));
        assert_eq!(fake.gets().len(), 1);
    }

    #[test]
    fn stops_a_wait_when_canceled() {
        let silent = FakeS3Server::silent();
        let runtime = Runtime::new().unwrap();
        let store = store(&runtime, &silent, Transfers::default());
        // Its size, since a HEAD would wait too.
        let blob = Blob {
            size: 400,
            e_tag: None,
            kms_key: None,
        };
        let cancel = Cancel::new();
        let canceler = cancel.clone();
        thread::spawn(move || {
            thread::sleep(Duration::from_millis(200));
            canceler.cancel();
        });
        let start = Instant::now();
        let mut out = vec![0.0; 100];
        let result = store.download(
            "bucket",
            "vectors.knnvec",
            &blob,
            &mut out,
            f32::from_le_bytes,
            &cancel,
        );
        assert!(matches!(result, Err(Error::Canceled { .. })), "{result:?}");
        // Long before the 60-second read timeout.
        assert!(start.elapsed() < Duration::from_secs(10));
    }

    #[tokio::test(flavor = "multi_thread")]
    // tokio's message: "Cannot start a runtime from within a runtime".
    #[should_panic(expected = "runtime")]
    async fn panics_in_an_async_context() {
        // Several ranges, which without the check would block this
        // runtime thread and succeed.
        let values: Vec<f32> = (0..2_500_u16).map(f32::from).collect();
        let fake = FakeS3Server::serving(le(&values), 0);
        let transfers = Transfers {
            range_size: 1_000,
            ..Transfers::default()
        };
        let store = test_store(Some(&fake), transfers).await;
        let blob = store.head("bucket", "vectors.knnvec").await.unwrap();
        let mut out = vec![0.0; values.len()];
        _ = store.download(
            "bucket",
            "vectors.knnvec",
            &blob,
            &mut out,
            f32::from_le_bytes,
            &Cancel::new(),
        );
    }

    #[test]
    fn retries_bodies_cut_off() {
        let values: Vec<f32> = (0..2_500_u16).map(|i| f32::from(i) - 7.5).collect();
        let fake = FakeS3Server::serving(le(&values), 2);
        let runtime = Runtime::new().unwrap();
        let transfers = Transfers {
            range_size: 4_000,
            download_concurrency: 2,
            ..Transfers::default()
        };
        let store = store(&runtime, &fake, transfers);
        assert!(
            download(&runtime, &store, values.len()).unwrap() == values,
            "values differ"
        );

        // Three ranges, two of them asked for twice; each GET names the
        // blob's ETag.
        let gets = fake.gets();
        assert_eq!(gets.len(), 5, "{gets:?}");
        assert!(
            gets.iter()
                .all(|get| get.if_match.as_deref() == Some(E_TAG))
        );
        let mut ranges: Vec<_> = gets.into_iter().filter_map(|get| get.range).collect();
        ranges.sort();
        ranges.dedup();
        assert_eq!(
            ranges,
            ["bytes=0-3999", "bytes=4000-7999", "bytes=8000-9999"]
        );
    }

    #[test]
    fn gives_up_after_five_cut_bodies() {
        let fake = FakeS3Server::serving(le(&[1.0; 100]), usize::MAX);
        let runtime = Runtime::new().unwrap();
        let store = store(&runtime, &fake, Transfers::default());
        let error = download(&runtime, &store, 100).unwrap_err();
        assert!(
            matches!(
                error,
                Error::S3 {
                    call: "GetObject",
                    ..
                }
            ),
            "{error}"
        );
        assert_eq!(fake.gets().len(), 5);
    }

    #[test]
    fn fetches_a_small_blob_in_one_get() {
        let values = [0.25, -1.0, 3.5];
        let runtime = Runtime::new().unwrap();
        // Far smaller than a range, and exactly one.
        for range_size in [Transfers::default().range_size, 12] {
            let fake = FakeS3Server::serving(le(&values), 0);
            let transfers = Transfers {
                range_size,
                ..Transfers::default()
            };
            let store = store(&runtime, &fake, transfers);
            assert_eq!(download(&runtime, &store, 3).unwrap(), values);
            // Whole, so the SDK can check the object's checksum.
            assert_eq!(
                fake.gets(),
                [Get {
                    range: None,
                    if_match: Some(E_TAG.to_owned()),
                }],
                "range size {range_size}"
            );
        }
    }

    #[test]
    fn decodes_values_split_across_frames() {
        let values: Vec<f32> = (0..9_u8).map(|i| f32::from(i) * 1.5 - 4.0).collect();
        let bytes = le(&values);
        // Every pair of cut points, so values straddle frames at every
        // offset, and empty frames come in between.
        for first in 0..=bytes.len() {
            for second in first..=bytes.len() {
                let mut out = [0.0_f32; 9];
                let mut fill = Values::new(&mut out, f32::from_le_bytes);
                for frame in [
                    &bytes[..first],
                    &[],
                    &bytes[first..second],
                    &bytes[second..],
                ] {
                    fill.push(frame).unwrap();
                }
                fill.finish().unwrap();
                assert_eq!(out, values.as_slice(), "cut at {first} and {second}");
            }
        }
    }

    #[test]
    fn converts_through_decode() {
        let bytes = le(&[1.0, -2.0, 0.5]);
        let mut out = [0_u16; 3];
        let mut fill = Values::new(&mut out, |b| {
            // The top half of an f32 that has no low mantissa bits.
            u16::try_from(u32::from_le_bytes(b) >> 16).unwrap()
        });
        fill.push(&bytes[..5]).unwrap();
        fill.push(&bytes[5..]).unwrap();
        fill.finish().unwrap();
        assert_eq!(out, [0x3f80, 0xc000, 0x3f00]);
    }

    #[test]
    fn refuses_a_body_of_another_length() {
        let bytes = le(&[1.0, 2.0]);

        // Short by a whole value, and by part of one.
        let mut out = [0.0_f32; 3];
        let mut fill = Values::new(&mut out, f32::from_le_bytes);
        fill.push(&bytes).unwrap();
        assert_eq!(fill.finish(), Err(8));
        let mut out = [0.0_f32; 2];
        let mut fill = Values::new(&mut out, f32::from_le_bytes);
        fill.push(&bytes[..7]).unwrap();
        assert_eq!(fill.finish(), Err(7));

        // Long by a whole value, and by part of one.
        let mut out = [0.0_f32; 1];
        let mut fill = Values::new(&mut out, f32::from_le_bytes);
        assert_eq!(fill.push(&bytes), Err(8));
        let mut out = [0.0_f32; 2];
        let mut fill = Values::new(&mut out, f32::from_le_bytes);
        fill.push(&bytes).unwrap();
        fill.push(&[0]).unwrap();
        assert_eq!(fill.finish(), Err(9));
    }
}
