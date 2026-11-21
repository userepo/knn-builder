//! The service's metrics, and the page a scraper reads them from.
//!
//! A [`Snapshot`] holds every value at one moment, and
//! [`Snapshot::render`] writes it in the Prometheus text exposition
//! format, version 0.0.4. Its [`Counted`] half is what the service adds to
//! as it runs, the counters and the histograms; the rest is read when the
//! page is asked for.
//!
//! Every label is a [`Label`], a closed set of values, so the page lists
//! the same series from the service's start to its end, each counter
//! from 0: a rate over a series is defined before its first event. None
//! names a bucket, a path, a tenant or a job, so the page grows with
//! nothing a client sends and tells nothing of a tenant's data. The
//! requests answered are the one metric listed as it comes, by the status
//! codes seen, and the process's and the GPU's memory are left out where
//! they cannot be read.
//!
//! Nothing here reads a clock or a file: the values are the caller's, and
//! the same snapshot renders the same page. [`Answers`] counts the
//! requests answered without a lock, for the API to count each as it
//! answers it.
//!
//! ```
//! use knn_builder::metrics::{Outcome, Snapshot};
//!
//! let mut snapshot = Snapshot::default();
//! snapshot.counted.jobs_ended.add(Outcome::Completed, 1);
//! let page = snapshot.render();
//! assert!(page.contains("knn_builder_jobs_ended_total{outcome=\"completed\"} 1\n"));
//! ```

use std::fmt::{self, Display, Write as _};
use std::iter;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{DataType, Stage};

/// The buckets of the histograms of seconds: from a flushed segment of a
/// few thousand vectors to most of the data node's default timeout.
pub const SECONDS: [f64; 14] = [
    0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0,
];

/// The buckets of the histogram of a job's time over its forecast.
pub const RATIOS: [f64; 9] = [0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 4.0];

/// The buckets of the histogram of recall.
pub const RECALLS: [f64; 8] = [0.5, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 1.0];

/// A label whose values are a closed set.
///
/// A [`Family`] holds one value for each, so a metric with the label has
/// all of its series from the start.
pub trait Label: Copy + PartialEq + 'static {
    /// The label's name on the page.
    const KEY: &'static str;
    /// Every value, in the order the page lists them.
    const ALL: &'static [Self];
    /// The value as the page writes it.
    fn text(self) -> &'static str;
}

/// Defines a label: the enum of its values, each with its text on the
/// page, so no value can be left out of [`Label::ALL`].
macro_rules! label {
    (
        $(#[$meta:meta])*
        $name:ident, $key:literal {
            $($(#[$variant_meta:meta])* $variant:ident => $text:literal,)+
        }
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum $name {
            $($(#[$variant_meta])* $variant,)+
        }

        impl Label for $name {
            const KEY: &'static str = $key;
            const ALL: &'static [Self] = &[$(Self::$variant,)+];

            fn text(self) -> &'static str {
                match self {
                    $(Self::$variant => $text,)+
                }
            }
        }
    };
}

label! {
    /// How a build request was answered: the `result` label.
    BuildAnswer, "result" {
        /// A new job was taken.
        Accepted => "accepted",
        /// The job was held already: a retry.
        Held => "held",
        /// The body was refused.
        Invalid => "invalid",
        /// 409: the job id is held for other parameters.
        Conflict => "conflict",
        /// 507: the job store is full.
        StoreFull => "store_full",
        /// 507: a memory estimate is over a limit, or the build over the
        /// GPU's free memory.
        TooLarge => "too_large",
        /// 507: the job is forecast to end past the data node's timeout.
        Late => "late",
        /// 507: the data node would build the segment sooner itself.
        CpuSooner => "cpu_sooner",
    }
}

label! {
    /// How a job ended: the `outcome` label.
    Outcome, "outcome" {
        /// Its index is stored.
        Completed => "completed",
        /// An input could not be read, or is of another size than the
        /// request implies.
        DownloadFailed => "download_failed",
        /// Its inputs did not fit in host memory.
        HostMemory => "host_memory",
        /// A request parameter no build can take.
        BadParameters => "bad_parameters",
        /// Too little device memory was free for its build.
        DeviceMemory => "device_memory",
        /// The builder failed.
        BuildFailed => "build_failed",
        /// The index could not be written.
        WriteFailed => "write_failed",
        /// The index could not be stored.
        UploadFailed => "upload_failed",
        /// Its recall was below the least the service takes.
        LowRecall => "low_recall",
        /// Dropped before its build: forecast again, it would have ended
        /// past the data node's timeout.
        ForecastLate => "forecast_late",
        /// Dropped once the data node's timeout had passed.
        TimedOut => "timed_out",
        /// Canceled, or cut short by the service's stop.
        Canceled => "canceled",
        /// A stage panicked on it.
        Panicked => "panicked",
    }
}

label! {
    /// Where a job in flight is: the `stage` label of the jobs in flight.
    FlightStage, "stage" {
        /// Waiting to download.
        Queued => "queued",
        /// Downloading.
        Downloading => "downloading",
        /// Downloaded, waiting for the builder.
        Loaded => "loaded",
        /// Building.
        Building => "building",
        /// Built, waiting for the upload stage.
        Graphed => "graphed",
        /// Projecting, checking, writing and uploading.
        Uploading => "uploading",
    }
}

impl From<Stage> for FlightStage {
    fn from(stage: Stage) -> Self {
        match stage {
            Stage::Queued => Self::Queued,
            Stage::Downloading => Self::Downloading,
            Stage::Loaded => Self::Loaded,
            Stage::Building => Self::Building,
            Stage::Graphed => Self::Graphed,
            Stage::Uploading => Self::Uploading,
        }
    }
}

label! {
    /// A stage that is timed: the `stage` label of the stage times.
    TimedStage, "stage" {
        /// Reading the inputs' sizes and downloading them.
        Download => "download",
        /// Building the graph.
        Build => "build",
        /// Projecting layer-1 links into the graph.
        Projection => "projection",
        /// The recall check.
        RecallCheck => "recall_check",
        /// Writing and uploading the index.
        Upload => "upload",
    }
}

label! {
    /// How a job's vectors are encoded: the `data_type` label.
    VectorType, "data_type" {
        /// `float`.
        Float => "float",
        /// `half_float`.
        HalfFloat => "half_float",
        /// `byte`.
        Byte => "byte",
        /// `binary`.
        Binary => "binary",
    }
}

impl From<DataType> for VectorType {
    fn from(data_type: DataType) -> Self {
        match data_type {
            DataType::Float => Self::Float,
            DataType::HalfFloat => Self::HalfFloat,
            DataType::Byte => Self::Byte,
            DataType::Binary => Self::Binary,
        }
    }
}

label! {
    /// The route a request was for: the `route` label.
    Route, "route" {
        /// `/_build`.
        Build => "build",
        /// `/_status/{job_id}`.
        Status => "status",
        /// `/_jobs`.
        Jobs => "jobs",
        /// `/_heart_beat`.
        HeartBeat => "heart_beat",
        /// `/metrics`.
        Metrics => "metrics",
        /// Any other path.
        Other => "other",
    }
}

/// One value for each value of the label `L`.
#[derive(Debug, Clone, PartialEq)]
pub struct Family<L, T> {
    /// As many as `L::ALL`, in its order.
    values: Vec<T>,
    label: PhantomData<L>,
}

impl<L: Label, T> Family<L, T> {
    /// A family with `value` for every value of the label.
    #[must_use]
    pub fn new(value: T) -> Self
    where
        T: Clone,
    {
        Self {
            values: vec![value; L::ALL.len()],
            label: PhantomData,
        }
    }

    /// Sets the value for `label`.
    pub fn set(&mut self, label: L, value: T) {
        if let Some(slot) = self.slot(label) {
            *slot = value;
        }
    }

    /// Every value of the label with its value, in the page's order.
    pub fn iter(&self) -> impl Iterator<Item = (L, &T)> {
        L::ALL.iter().copied().zip(&self.values)
    }

    /// The value for `label`: there for every value in `L::ALL`.
    fn slot(&mut self, label: L) -> Option<&mut T> {
        let index = L::ALL.iter().position(|&known| known == label)?;
        self.values.get_mut(index)
    }
}

impl<L: Label, T: Default + Clone> Default for Family<L, T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

impl<L: Label> Family<L, u64> {
    /// Adds `count` to the value for `label`, up to the most it holds.
    pub fn add(&mut self, label: L, count: u64) {
        if let Some(slot) = self.slot(label) {
            *slot = slot.saturating_add(count);
        }
    }
}

impl<L: Label> Family<L, Histogram> {
    /// Counts `value` in the histogram for `label`, as
    /// [`Histogram::observe`] does.
    pub fn observe(&mut self, label: L, value: f64) {
        if let Some(slot) = self.slot(label) {
            slot.observe(value);
        }
    }
}

/// Counts of values by the bucket each falls in, and their sum.
#[derive(Debug, Clone, PartialEq)]
pub struct Histogram {
    bounds: &'static [f64],
    /// The values at or below each bound and above the one before it, then
    /// those above every bound.
    counts: Vec<u64>,
    sum: f64,
}

impl Histogram {
    /// An empty histogram whose buckets end at `bounds`, ascending.
    #[must_use]
    pub fn new(bounds: &'static [f64]) -> Self {
        Self {
            bounds,
            counts: vec![0; bounds.len() + 1],
            sum: 0.0,
        }
    }

    /// Counts `value` in the first bucket whose bound it does not pass.
    ///
    /// A value that is not finite is left out: it would leave the sum not
    /// a number of anything from then on.
    pub fn observe(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }
        let bucket = self
            .bounds
            .iter()
            .position(|&bound| value <= bound)
            .unwrap_or(self.bounds.len());
        if let Some(count) = self.counts.get_mut(bucket) {
            *count = count.saturating_add(1);
            self.sum += value;
        }
    }

    /// How many values were counted.
    #[must_use]
    pub fn count(&self) -> u64 {
        self.counts
            .iter()
            .fold(0, |total, &count| total.saturating_add(count))
    }

    /// The sum of the values counted.
    #[must_use]
    pub fn sum(&self) -> f64 {
        self.sum
    }

    /// Each bucket's bound and its own count, the last bound infinite.
    fn buckets(&self) -> impl Iterator<Item = (f64, u64)> {
        let bounds = self.bounds.iter().copied().chain(iter::once(f64::INFINITY));
        bounds.zip(self.counts.iter().copied())
    }
}

/// What the service counts as it runs: the counters and the histograms.
///
/// The times, the forecast ratio and the vectors are of jobs that ended
/// with their index stored; the recall is of those and of the jobs failed
/// for their recall.
#[derive(Debug, Clone, PartialEq)]
pub struct Counted {
    /// Build requests, by how they were answered.
    pub build_requests: Family<BuildAnswer, u64>,
    /// Jobs ended, by how.
    pub jobs_ended: Family<Outcome, u64>,
    /// Vectors in the indexes stored.
    pub vectors_built: Family<VectorType, u64>,
    /// Seconds from a job's request to its end, in [`SECONDS`].
    pub job_seconds: Histogram,
    /// Seconds from a job's request until its download started, in
    /// [`SECONDS`].
    pub queue_seconds: Histogram,
    /// Seconds each stage took, in [`SECONDS`].
    pub stage_seconds: Family<TimedStage, Histogram>,
    /// Seconds a job took over the seconds forecast when it was taken, in
    /// [`RATIOS`].
    pub forecast_ratio: Histogram,
    /// The recall the recall check found, in [`RECALLS`].
    pub recall: Histogram,
}

impl Default for Counted {
    fn default() -> Self {
        Self {
            build_requests: Family::default(),
            jobs_ended: Family::default(),
            vectors_built: Family::default(),
            job_seconds: Histogram::new(&SECONDS),
            queue_seconds: Histogram::new(&SECONDS),
            stage_seconds: Family::new(Histogram::new(&SECONDS)),
            forecast_ratio: Histogram::new(&RATIOS),
            recall: Histogram::new(&RECALLS),
        }
    }
}

/// The status codes [`Answers`] counts, as HTTP has them.
const CODES: std::ops::RangeInclusive<u16> = 100..=999;

/// The requests answered, by route and status code, each count atomic.
///
/// A slot for every route and code, so counting takes no lock and
/// allocates nothing; [`Answers::answered`] lists the ones counted.
pub struct Answers {
    /// By route, then by code from the first of [`CODES`].
    counts: Vec<AtomicU64>,
}

impl Default for Answers {
    fn default() -> Self {
        let slots = Route::ALL.len() * CODES.len();
        Self {
            counts: iter::repeat_with(AtomicU64::default).take(slots).collect(),
        }
    }
}

/// The requests counted, not every slot.
impl fmt::Debug for Answers {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Answers")
            .field("answered", &self.answered())
            .finish()
    }
}

impl Answers {
    /// Counts one request on `route` answered with `code`; a code HTTP
    /// does not have is not counted.
    pub fn count(&self, route: Route, code: u16) {
        if let Some(count) = self.slot(route, code) {
            count.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Every route and code counted, with its count.
    #[must_use]
    pub fn answered(&self) -> Vec<Answered> {
        let mut answered = Vec::new();
        for &route in Route::ALL {
            for code in CODES {
                let requests = self
                    .slot(route, code)
                    .map_or(0, |c| c.load(Ordering::Relaxed));
                if requests > 0 {
                    answered.push(Answered {
                        route,
                        code,
                        requests,
                    });
                }
            }
        }
        answered
    }

    fn slot(&self, route: Route, code: u16) -> Option<&AtomicU64> {
        if !CODES.contains(&code) {
            return None;
        }
        let route = Route::ALL.iter().position(|&known| known == route)?;
        let code = usize::from(code - CODES.start());
        self.counts.get(route * CODES.len() + code)
    }
}

/// The requests answered on one route with one status code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Answered {
    /// The route.
    pub route: Route,
    /// The HTTP status code.
    pub code: u16,
    /// How many.
    pub requests: u64,
}

/// The GPU's memory, in bytes, every process on the device counted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuMemory {
    /// The device's memory.
    pub total: u64,
    /// Free when last read.
    pub free: u64,
    /// The most in use during the last build that ended; 0 before one has.
    pub last_build_peak: u64,
}

/// Every metric's value at one moment.
///
/// The default is the page of a service that has done nothing: every
/// count 0, every ratio 1, this crate's version, and no memory read.
#[derive(Debug, Clone, PartialEq)]
pub struct Snapshot {
    /// The service's version.
    pub version: &'static str,
    /// What the service has counted since it started.
    pub counted: Counted,
    /// Requests answered, by route and status code; the page lists them
    /// by route, then by code, two of one route and code as one.
    pub http_requests: Vec<Answered>,
    /// Jobs that have not ended, by pipeline stage.
    pub jobs_in_flight: Family<FlightStage, u64>,
    /// Jobs held: running, or ended and not yet expired.
    pub jobs_held: u64,
    /// Jobs held at most.
    pub jobs_held_limit: u64,
    /// Observed over modeled seconds of each stage, as the forecast has
    /// learned it.
    pub stage_model_ratio: Family<TimedStage, f64>,
    /// Forecast seconds until the last job in flight ends; 0 with none.
    pub drain_seconds: f64,
    /// Host memory the jobs' buffers may take at once, in bytes.
    pub host_budget: u64,
    /// Host memory the buffers of the jobs in the pipeline count for, in
    /// bytes.
    pub host_budget_used: u64,
    /// The process's resident memory, in bytes, where it can be read.
    pub resident_memory: Option<u64>,
    /// When the service started, in seconds of Unix time.
    pub start_time_seconds: f64,
    /// The GPU's memory, where there is a GPU to read.
    pub gpu_memory: Option<GpuMemory>,
}

impl Default for Snapshot {
    fn default() -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION"),
            counted: Counted::default(),
            http_requests: Vec::new(),
            jobs_in_flight: Family::default(),
            jobs_held: 0,
            jobs_held_limit: 0,
            stage_model_ratio: Family::new(1.0),
            drain_seconds: 0.0,
            host_budget: 0,
            host_budget_used: 0,
            resident_memory: None,
            start_time_seconds: 0.0,
            gpu_memory: None,
        }
    }
}

impl Snapshot {
    /// The page: every metric in the Prometheus text exposition format.
    #[must_use]
    pub fn render(&self) -> String {
        let mut page = Page::default();
        self.requests(&mut page);
        self.jobs(&mut page);
        self.times(&mut page);
        self.memory(&mut page);
        page.0
    }

    /// The version, and the requests answered.
    fn requests(&self, page: &mut Page) {
        page.metric(
            "knn_builder_build_info",
            Kind::Gauge,
            "The running version, as a label; always 1.",
        );
        page.sample("knn_builder_build_info", &[("version", self.version)], 1);

        page.metric(
            "knn_builder_http_requests_total",
            Kind::Counter,
            "Requests answered, by route and status code.",
        );
        let mut answered = self.http_requests.clone();
        answered.sort_by_key(|answered| {
            let route = Route::ALL.iter().position(|&route| route == answered.route);
            (route, answered.code)
        });
        // A series written twice would have the whole page refused.
        answered.dedup_by(|next, kept| {
            let same = (next.route, next.code) == (kept.route, kept.code);
            if same {
                kept.requests = kept.requests.saturating_add(next.requests);
            }
            same
        });
        for answered in answered {
            let code = answered.code.to_string();
            let labels = [(Route::KEY, answered.route.text()), ("code", code.as_str())];
            page.sample(
                "knn_builder_http_requests_total",
                &labels,
                answered.requests,
            );
        }

        page.counts(
            "knn_builder_build_requests_total",
            Kind::Counter,
            "Build requests, by how they were answered.",
            &self.counted.build_requests,
        );
    }

    /// The jobs: in flight, held and ended.
    fn jobs(&self, page: &mut Page) {
        let counted = &self.counted;
        page.counts(
            "knn_builder_jobs_in_flight",
            Kind::Gauge,
            "Jobs that have not ended, by pipeline stage.",
            &self.jobs_in_flight,
        );
        page.single(
            "knn_builder_jobs_held",
            "Jobs held: running, or ended and not yet expired.",
            self.jobs_held,
        );
        page.single(
            "knn_builder_jobs_held_limit",
            "Jobs held at most (REQUEST_STORE_MAX_SIZE).",
            self.jobs_held_limit,
        );
        page.counts(
            "knn_builder_jobs_ended_total",
            Kind::Counter,
            "Jobs ended, by how.",
            &counted.jobs_ended,
        );
        page.counts(
            "knn_builder_vectors_built_total",
            Kind::Counter,
            "Vectors in the indexes stored, by data type.",
            &counted.vectors_built,
        );
        page.metric(
            "knn_builder_recall",
            Kind::Histogram,
            "Recall the recall check found, for jobs that stored an index or failed for their recall.",
        );
        page.histogram("knn_builder_recall", &[], &counted.recall);
    }

    /// How long the jobs took, and how that compares with the forecast.
    fn times(&self, page: &mut Page) {
        let counted = &self.counted;
        page.metric(
            "knn_builder_job_seconds",
            Kind::Histogram,
            "Seconds from a build request to its job's end, for jobs that stored an index.",
        );
        page.histogram("knn_builder_job_seconds", &[], &counted.job_seconds);
        page.metric(
            "knn_builder_queue_seconds",
            Kind::Histogram,
            "Seconds from a build request until its download started, for jobs that stored an index.",
        );
        page.histogram("knn_builder_queue_seconds", &[], &counted.queue_seconds);
        page.metric(
            "knn_builder_stage_seconds",
            Kind::Histogram,
            "Seconds each stage took, for jobs that stored an index.",
        );
        for (stage, histogram) in counted.stage_seconds.iter() {
            let labels = [(TimedStage::KEY, stage.text())];
            page.histogram("knn_builder_stage_seconds", &labels, histogram);
        }
        page.metric(
            "knn_builder_forecast_ratio",
            Kind::Histogram,
            "Seconds a job took over the seconds forecast when it was taken, for jobs that stored an index.",
        );
        page.histogram("knn_builder_forecast_ratio", &[], &counted.forecast_ratio);

        page.metric(
            "knn_builder_stage_model_ratio",
            Kind::Gauge,
            "Observed over modeled seconds of each stage, as learned from the jobs that ended.",
        );
        for (stage, &ratio) in self.stage_model_ratio.iter() {
            let labels = [(TimedStage::KEY, stage.text())];
            page.sample("knn_builder_stage_model_ratio", &labels, Number(ratio));
        }
        page.single(
            "knn_builder_drain_seconds",
            "Forecast seconds until the last job in flight ends; 0 with none.",
            Number(self.drain_seconds),
        );
    }

    /// The host's memory, the process's and the GPU's.
    fn memory(&self, page: &mut Page) {
        page.single(
            "knn_builder_host_budget_bytes",
            "Host memory the jobs' buffers may take at once (CPU_MEMORY_LIMIT).",
            self.host_budget,
        );
        page.single(
            "knn_builder_host_budget_used_bytes",
            "Host memory the buffers of the jobs in the pipeline count for.",
            self.host_budget_used,
        );
        if let Some(resident) = self.resident_memory {
            page.single(
                "process_resident_memory_bytes",
                "Resident memory size in bytes.",
                resident,
            );
        }
        page.single(
            "process_start_time_seconds",
            "Start time of the process since the Unix epoch in seconds.",
            Number(self.start_time_seconds),
        );
        if let Some(gpu) = self.gpu_memory {
            page.single(
                "knn_builder_gpu_memory_total_bytes",
                "The GPU's memory.",
                gpu.total,
            );
            page.single(
                "knn_builder_gpu_memory_free_bytes",
                "The GPU's free memory when last read, every process on it counted.",
                gpu.free,
            );
            page.single(
                "knn_builder_gpu_memory_last_build_peak_bytes",
                "The most GPU memory in use during the last build that ended; 0 before one has.",
                gpu.last_build_peak,
            );
        }
    }
}

/// A metric's type, as its `# TYPE` line names it.
#[derive(Debug, Clone, Copy)]
enum Kind {
    Counter,
    Gauge,
    Histogram,
}

impl Kind {
    fn text(self) -> &'static str {
        match self {
            Self::Counter => "counter",
            Self::Gauge => "gauge",
            Self::Histogram => "histogram",
        }
    }
}

/// The page as it is written. Writing to a string cannot fail.
#[derive(Debug, Default)]
struct Page(String);

impl Page {
    /// Starts the metric `name`: its help and its type. `help` holds no
    /// backslash and no line break, which the format would have escaped.
    fn metric(&mut self, name: &str, kind: Kind, help: &str) {
        let _ = writeln!(self.0, "# HELP {name} {help}");
        let _ = writeln!(self.0, "# TYPE {name} {}", kind.text());
    }

    /// One series of the metric `name`: its labels and its value.
    fn sample(&mut self, name: &str, labels: &[(&str, &str)], value: impl Display + Copy) {
        self.0.push_str(name);
        for (index, (key, text)) in labels.iter().enumerate() {
            self.0.push(if index == 0 { '{' } else { ',' });
            let _ = write!(self.0, "{key}=\"");
            escape(&mut self.0, text);
            self.0.push('"');
        }
        if !labels.is_empty() {
            self.0.push('}');
        }
        let _ = writeln!(self.0, " {value}");
    }

    /// A gauge with no label.
    fn single(&mut self, name: &str, help: &str, value: impl Display + Copy) {
        self.metric(name, Kind::Gauge, help);
        self.sample(name, &[], value);
    }

    /// A metric with one series for each value of a label.
    fn counts<L: Label>(&mut self, name: &str, kind: Kind, help: &str, family: &Family<L, u64>) {
        self.metric(name, kind, help);
        for (label, count) in family.iter() {
            self.sample(name, &[(L::KEY, label.text())], count);
        }
    }

    /// The series of one histogram of the metric `name`: its buckets, each
    /// counting those before it too, then its sum and its count.
    fn histogram(&mut self, name: &str, labels: &[(&str, &str)], histogram: &Histogram) {
        let bucket = format!("{name}_bucket");
        let mut count = 0_u64;
        for (bound, own) in histogram.buckets() {
            count = count.saturating_add(own);
            let bound = Number(bound).to_string();
            let mut labels = labels.to_vec();
            labels.push(("le", bound.as_str()));
            self.sample(&bucket, &labels, count);
        }
        self.sample(&format!("{name}_sum"), labels, Number(histogram.sum));
        self.sample(&format!("{name}_count"), labels, count);
    }
}

/// Writes a label's value: a backslash, a quote and a line break escaped.
fn escape(out: &mut String, text: &str) {
    for c in text.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            c => out.push(c),
        }
    }
}

/// A float as the format writes it: `+Inf`, `-Inf` and `NaN` for what is
/// no finite number, and no exponent.
#[derive(Debug, Clone, Copy)]
struct Number(f64);

impl Display for Number {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self(value) = *self;
        if value.is_nan() {
            f.write_str("NaN")
        } else if value.is_infinite() {
            f.write_str(if value.is_sign_positive() {
                "+Inf"
            } else {
                "-Inf"
            })
        } else {
            write!(f, "{value}")
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use super::{
        Answered, Answers, BuildAnswer, Family, FlightStage, GpuMemory, Histogram, Label, Number,
        Outcome, RATIOS, RECALLS, Route, SECONDS, Snapshot, TimedStage, VectorType, escape,
    };
    use crate::{DataType, Stage};

    /// A snapshot with every optional metric and a value in each kind.
    fn busy() -> Snapshot {
        let mut snapshot = Snapshot {
            http_requests: [
                (Route::Status, 404, 2),
                (Route::Build, 507, 3),
                (Route::Status, 200, 40),
                (Route::Build, 200, 5),
                (Route::Status, 404, 1),
            ]
            .into_iter()
            .map(|(route, code, requests)| Answered {
                route,
                code,
                requests,
            })
            .collect(),
            resident_memory: Some(3 << 30),
            gpu_memory: Some(GpuMemory {
                total: 8 << 30,
                free: 7 << 30,
                last_build_peak: 2 << 30,
            }),
            ..Snapshot::default()
        };
        let counted = &mut snapshot.counted;
        counted.build_requests.add(BuildAnswer::Accepted, 5);
        counted.build_requests.add(BuildAnswer::Late, 3);
        counted.jobs_ended.add(Outcome::Completed, 4);
        counted.jobs_ended.add(Outcome::ForecastLate, 1);
        counted.vectors_built.add(VectorType::Float, 4_000_000);
        for seconds in [0.05, 3.0, 30.0, 4000.0] {
            counted.job_seconds.observe(seconds);
            counted.stage_seconds.observe(TimedStage::Build, seconds);
        }
        counted.recall.observe(0.96);
        snapshot.stage_model_ratio.set(TimedStage::Build, 0.5);
        snapshot
    }

    /// The page's lines that are of the metric `name`, labels and all.
    fn lines<'a>(page: &'a str, name: &str) -> Vec<&'a str> {
        page.lines()
            .filter(|line| {
                line.strip_prefix(name)
                    .is_some_and(|rest| rest.starts_with(['{', ' ']))
            })
            .collect()
    }

    #[test]
    fn buckets_ascend() {
        for bounds in [&SECONDS[..], &RATIOS[..], &RECALLS[..]] {
            assert!(bounds.iter().all(|bound| bound.is_finite()));
            assert!(bounds.windows(2).all(|pair| pair[0] < pair[1]));
        }
    }

    #[test]
    fn counts_a_value_in_the_first_bucket_it_does_not_pass() {
        static BOUNDS: [f64; 3] = [1.0, 2.0, 4.0];
        let mut histogram = Histogram::new(&BOUNDS);
        // A bound is in its own bucket; 0 and a value below it in the first.
        for value in [-1.0, 0.0, 1.0, 1.5, 2.0, 4.0, 4.5, 100.0] {
            histogram.observe(value);
        }
        let counts: Vec<u64> = histogram.buckets().map(|(_, count)| count).collect();
        assert_eq!(counts, [3, 2, 1, 2]);
        assert_eq!(histogram.count(), 8);
        assert!((histogram.sum() - 112.0).abs() < 1e-9);
        let last = histogram.buckets().last().unwrap().0;
        assert!(last.is_infinite() && last.is_sign_positive());
    }

    #[test]
    fn leaves_out_a_value_that_is_not_finite() {
        let mut histogram = Histogram::new(&RATIOS);
        histogram.observe(1.0);
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            histogram.observe(value);
        }
        assert_eq!(histogram.count(), 1);
        assert!((histogram.sum() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_family_holds_a_value_for_each_label() {
        let mut family = Family::<Outcome, u64>::default();
        assert_eq!(family.iter().count(), Outcome::ALL.len());
        family.add(Outcome::LowRecall, 2);
        family.add(Outcome::LowRecall, 3);
        family.add(Outcome::Completed, u64::MAX);
        family.add(Outcome::Completed, 1);
        for (outcome, &count) in family.iter() {
            let expected = match outcome {
                Outcome::LowRecall => 5,
                // As many as it holds, and no fewer for one more.
                Outcome::Completed => u64::MAX,
                _ => 0,
            };
            assert_eq!(count, expected, "{outcome:?}");
        }
        family.set(Outcome::LowRecall, 1);
        assert_eq!(family.iter().map(|(_, &count)| count).min(), Some(0));
        assert!(
            family
                .iter()
                .any(|(o, &c)| o == Outcome::LowRecall && c == 1)
        );
    }

    /// Each label's values are told apart on the page, and are written as
    /// the format takes a value with nothing to escape.
    #[test]
    fn labels_have_distinct_plain_texts() {
        fn check<L: Label + std::fmt::Debug>() {
            let texts: HashSet<&str> = L::ALL.iter().map(|label| label.text()).collect();
            assert_eq!(texts.len(), L::ALL.len(), "{:?}", L::ALL);
            for text in texts {
                assert_ne!(text, "");
                assert!(
                    text.chars()
                        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_'),
                    "{text}"
                );
            }
        }
        check::<BuildAnswer>();
        check::<Outcome>();
        check::<FlightStage>();
        check::<TimedStage>();
        check::<VectorType>();
        check::<Route>();
    }

    #[test]
    fn names_the_pipeline_stages_and_data_types_as_they_are() {
        let stages = [
            (Stage::Queued, "queued"),
            (Stage::Downloading, "downloading"),
            (Stage::Loaded, "loaded"),
            (Stage::Building, "building"),
            (Stage::Graphed, "graphed"),
            (Stage::Uploading, "uploading"),
        ];
        for (stage, text) in stages {
            assert_eq!(FlightStage::from(stage).text(), text);
        }
        let data_types = [
            (DataType::Float, "float"),
            (DataType::HalfFloat, "half_float"),
            (DataType::Byte, "byte"),
            (DataType::Binary, "binary"),
        ];
        for (data_type, text) in data_types {
            assert_eq!(VectorType::from(data_type).text(), text);
        }
    }

    #[test]
    fn counts_the_answers_by_route_and_code() {
        let answers = Answers::default();
        assert_eq!(answers.answered(), []);
        assert_eq!(format!("{answers:?}"), "Answers { answered: [] }");
        for _ in 0..3 {
            answers.count(Route::Status, 404);
        }
        answers.count(Route::Build, 200);
        answers.count(Route::Other, 999);
        answers.count(Route::Metrics, 100);
        // Not HTTP's: not counted, and nowhere else either.
        answers.count(Route::Build, 99);
        answers.count(Route::Build, 1000);
        let answered: Vec<(Route, u16, u64)> = answers
            .answered()
            .into_iter()
            .map(|a| (a.route, a.code, a.requests))
            .collect();
        assert_eq!(
            answered,
            [
                (Route::Build, 200, 1),
                (Route::Status, 404, 3),
                (Route::Metrics, 100, 1),
                (Route::Other, 999, 1),
            ]
        );
    }

    #[test]
    fn writes_floats_as_the_format_reads_them() {
        let cases = [
            (0.0, "0"),
            (1.0, "1"),
            (0.25, "0.25"),
            (2500.0, "2500"),
            (-3.5, "-3.5"),
            (1e21, "1000000000000000000000"),
            (1e-7, "0.0000001"),
            (f64::INFINITY, "+Inf"),
            (f64::NEG_INFINITY, "-Inf"),
            (f64::NAN, "NaN"),
        ];
        for (value, text) in cases {
            assert_eq!(Number(value).to_string(), text);
        }
    }

    #[test]
    fn escapes_a_label_value() {
        let mut out = String::new();
        escape(&mut out, "1.3.0-a\\b\"c\nd");
        assert_eq!(out, "1.3.0-a\\\\b\\\"c\\nd");
        let page = Snapshot {
            version: "1.3\"0",
            ..Snapshot::default()
        }
        .render();
        assert!(page.contains("knn_builder_build_info{version=\"1.3\\\"0\"} 1\n"));
    }

    #[test]
    fn an_idle_service_has_every_series_at_zero() {
        let page = Snapshot::default().render();
        let version = env!("CARGO_PKG_VERSION");
        assert!(page.contains(&format!(
            "knn_builder_build_info{{version=\"{version}\"}} 1\n"
        )));
        for (name, series) in [
            ("knn_builder_build_requests_total", BuildAnswer::ALL.len()),
            ("knn_builder_jobs_in_flight", FlightStage::ALL.len()),
            ("knn_builder_jobs_ended_total", Outcome::ALL.len()),
            ("knn_builder_vectors_built_total", VectorType::ALL.len()),
        ] {
            let lines = lines(&page, name);
            assert_eq!(lines.len(), series, "{name}");
            assert!(lines.iter().all(|line| line.ends_with(" 0")), "{name}");
        }
        // A bucket for each bound and one past them, for each stage.
        let buckets = lines(&page, "knn_builder_stage_seconds_bucket");
        assert_eq!(buckets.len(), TimedStage::ALL.len() * (SECONDS.len() + 1));
        assert!(buckets.iter().all(|line| line.ends_with(" 0")));
        assert_eq!(
            lines(&page, "knn_builder_recall_bucket").len(),
            RECALLS.len() + 1
        );
        assert_eq!(
            lines(&page, "knn_builder_forecast_ratio_bucket").len(),
            RATIOS.len() + 1
        );
        // No correction learned yet.
        let ratios = lines(&page, "knn_builder_stage_model_ratio");
        assert_eq!(ratios.len(), TimedStage::ALL.len());
        assert!(ratios.iter().all(|line| line.ends_with(" 1")));
        // Nothing answered, and no memory read.
        assert_eq!(lines(&page, "knn_builder_http_requests_total").len(), 0);
        assert!(!page.contains("process_resident_memory_bytes"));
        assert!(!page.contains("knn_builder_gpu_memory"));
    }

    #[test]
    fn writes_what_was_counted() {
        let page = busy().render();
        for line in [
            "knn_builder_build_requests_total{result=\"accepted\"} 5",
            "knn_builder_build_requests_total{result=\"late\"} 3",
            "knn_builder_build_requests_total{result=\"held\"} 0",
            "knn_builder_jobs_ended_total{outcome=\"completed\"} 4",
            "knn_builder_jobs_ended_total{outcome=\"forecast_late\"} 1",
            "knn_builder_vectors_built_total{data_type=\"float\"} 4000000",
            // Each bucket counts those below it too.
            "knn_builder_job_seconds_bucket{le=\"0.1\"} 1",
            "knn_builder_job_seconds_bucket{le=\"2.5\"} 1",
            "knn_builder_job_seconds_bucket{le=\"5\"} 2",
            "knn_builder_job_seconds_bucket{le=\"50\"} 3",
            "knn_builder_job_seconds_bucket{le=\"2500\"} 3",
            "knn_builder_job_seconds_bucket{le=\"+Inf\"} 4",
            "knn_builder_job_seconds_sum 4033.05",
            "knn_builder_job_seconds_count 4",
            "knn_builder_stage_seconds_bucket{stage=\"build\",le=\"5\"} 2",
            "knn_builder_stage_seconds_sum{stage=\"build\"} 4033.05",
            "knn_builder_stage_seconds_count{stage=\"build\"} 4",
            "knn_builder_stage_seconds_count{stage=\"upload\"} 0",
            "knn_builder_recall_bucket{le=\"0.95\"} 0",
            "knn_builder_recall_bucket{le=\"0.97\"} 1",
            "knn_builder_stage_model_ratio{stage=\"build\"} 0.5",
            "knn_builder_stage_model_ratio{stage=\"upload\"} 1",
            "process_resident_memory_bytes 3221225472",
            "knn_builder_gpu_memory_total_bytes 8589934592",
            "knn_builder_gpu_memory_free_bytes 7516192768",
            "knn_builder_gpu_memory_last_build_peak_bytes 2147483648",
        ] {
            assert!(page.lines().any(|l| l == line), "no line {line:?}");
        }
    }

    #[test]
    fn lists_the_requests_by_route_then_code() {
        let page = busy().render();
        assert_eq!(
            lines(&page, "knn_builder_http_requests_total"),
            [
                "knn_builder_http_requests_total{route=\"build\",code=\"200\"} 5",
                "knn_builder_http_requests_total{route=\"build\",code=\"507\"} 3",
                "knn_builder_http_requests_total{route=\"status\",code=\"200\"} 40",
                // Counted in two entries.
                "knn_builder_http_requests_total{route=\"status\",code=\"404\"} 3",
            ]
        );
    }

    /// The page as a reader of the format takes it apart: every series
    /// follows its metric's one help and type, a histogram's buckets count
    /// up to its count, and every value is a number.
    #[test]
    fn the_page_reads_as_the_format() {
        for snapshot in [Snapshot::default(), busy()] {
            let page = snapshot.render();
            assert!(page.ends_with('\n') && !page.contains('\r') && !page.contains("\n\n"));
            let mut kinds: HashMap<&str, &str> = HashMap::new();
            let mut helped: HashSet<&str> = HashSet::new();
            let mut current = "";
            // The last bucket's bound and count, and the count, by
            // histogram series.
            let mut buckets: HashMap<String, (&str, f64)> = HashMap::new();
            let mut counts: HashMap<String, f64> = HashMap::new();
            for line in page.lines() {
                if let Some(rest) = line.strip_prefix("# HELP ") {
                    let (name, help) = rest.split_once(' ').unwrap();
                    assert!(helped.insert(name), "{name} has two helps");
                    assert!(!help.is_empty() && !help.contains('\\'), "{line}");
                    current = name;
                    continue;
                }
                if let Some(rest) = line.strip_prefix("# TYPE ") {
                    let (name, kind) = rest.split_once(' ').unwrap();
                    assert_eq!(name, current, "a type under another metric's help");
                    assert!(kinds.insert(name, kind).is_none(), "{name} has two types");
                    continue;
                }
                let (series, value) = line.rsplit_once(' ').unwrap();
                let value: f64 = value.parse().unwrap();
                let (name, labels) = match series.split_once('{') {
                    Some((name, labels)) => (name, labels.strip_suffix('}').unwrap()),
                    None => (series, ""),
                };
                let kind = kinds[current];
                if kind != "histogram" {
                    assert_eq!(name, current, "{line}");
                    continue;
                }
                let part = name.strip_prefix(current).unwrap();
                let (own, le) = match labels.rsplit_once("le=\"") {
                    Some((own, le)) => (own.trim_end_matches(','), le.strip_suffix('"')),
                    None => (labels, None),
                };
                let key = format!("{current}{{{own}}}");
                match (part, le) {
                    ("_bucket", Some(le)) => {
                        let bound: f64 = le.parse().unwrap();
                        let (last, before) = buckets.insert(key, (le, value)).unwrap_or(("", 0.0));
                        // Bounds ascend, and none follows the one past all.
                        assert!(
                            last.parse().map_or(true, |last: f64| bound > last),
                            "{line}"
                        );
                        assert!(value >= before, "{line}");
                    }
                    ("_count", None) => {
                        let (last, bucket) = buckets[&key];
                        assert_eq!(last, "+Inf", "{line}");
                        assert!((bucket - value).abs() < 0.5, "{line}");
                        counts.insert(key, value);
                    }
                    ("_sum", None) => assert!(value.is_finite(), "{line}"),
                    _ => panic!("{line} is no part of a histogram"),
                }
            }
            assert_eq!(helped.len(), kinds.len());
            assert_eq!(buckets.len(), counts.len());
            assert!(!buckets.is_empty());
        }
    }
}
