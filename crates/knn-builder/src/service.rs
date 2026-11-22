//! The service's core: the jobs it holds, by id, and the pipeline that
//! runs them. `POST /_build` is [`Service::submit`] and `GET /_status` is
//! [`Service::status`]; the HTTP side is the binary's.
//!
//! As the upstream service: a request whose job is held gets that job, so
//! a retry never starts a second build; a full store, or a request whose
//! memory estimate is over a limit, gets 507, with the service's message.
//! Unlike it:
//!
//! - The same job id with other parameters gets 409, a conflict with the
//!   job held. The service compares only the vector path and tenant id,
//!   which the id is made of, and so returns the job held.
//! - A job expires `ttl` after it ends, and a running one never. The
//!   service counts from the request, so a build that runs past the TTL
//!   is dropped: its status answers 404, and its end is recorded nowhere.
//! - Only jobs that have not expired count toward the store's size; the
//!   service counts the expired ones until its sweep, every 5 s.
//! - Two identical requests at once start one job; the service can start
//!   two, since it checks and adds under separate locks.
//! - A job that passes is queued, and waits in the pipeline until its
//!   buffers fit beside the jobs ahead of it in `cpu_memory_limit`, where
//!   the service fails one whose estimate does not fit beside the jobs
//!   running. Either way the jobs held in memory stay within the limit;
//!   here measured by the buffers they take, there by their estimates.
//! - A job the data node has stopped waiting for, its timeout counted
//!   from the request, is canceled: one queued never starts, a transfer
//!   stops, and a build runs to its end but stores nothing. The service
//!   builds such a job anyway. And whenever a job ends, the jobs not yet
//!   building are forecast again, with what the ended ones taught, and
//!   those that would now miss are canceled at once, so their data nodes
//!   build the segments themselves at their next poll.
//! - A new job also gets 507 at once when the GPU's memory cannot hold its
//!   build, with the service's message, or when the data node, as it
//!   polls, would see it ended only past its timeout, or, when asked, when
//!   the data node would build it sooner itself (`crate::gate`). The
//!   service takes such a job, and the data node waits for it until its
//!   timeout or its failure.
//! - When set, a recall check samples each job's vectors as queries,
//!   searches its graph for them as k-NN will search the file, and logs the
//!   recall with the job's end; below a least recall set, the job fails
//!   with nothing stored ([`crate::recall`]).
//!
//! It also counts what it does, for [`Service::metrics`]: the build
//! requests by answer, and the jobs by how they ended, each once, with
//! the stage times, the forecast's accuracy and the recall of those that
//! report them.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use knn_store::Store;
use thiserror::Error;
use tokio::task::JoinSet;

use crate::cost::CpuThreads;
use crate::gate::Gate;
use crate::memory::{self, GpuMeter};
use crate::metrics::{BuildAnswer, Counted, Family, Outcome, Snapshot, TimedStage};
use crate::request::{BuildRequest, JobId};
use crate::{
    Builder, Built, Canceler, DataType, Flights, Job, JobError, Pipeline, RecallCheck, Span,
};

const POISONED: &str = "a thread panicked holding the job store";

/// How many jobs the service holds, for how long, how large a job it
/// takes, and what it knows of the data node and the GPU. The upstream
/// service's settings take its defaults, k-NN's take k-NN's.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Settings {
    /// Jobs held at most (`REQUEST_STORE_MAX_SIZE`); 10,000 by default.
    pub max_jobs: usize,
    /// How long a job is held after it ends (`REQUEST_STORE_TTL_SECONDS`);
    /// 1,800 s by default.
    pub ttl: Duration,
    /// The largest GPU memory estimate taken, in bytes
    /// (`GPU_MEMORY_LIMIT`, in GB); 24e9 by default.
    pub gpu_memory_limit: f64,
    /// The largest CPU memory estimate taken, in bytes
    /// (`CPU_MEMORY_LIMIT`, in GB); 32e9 by default. Also the host memory
    /// the buffers of the jobs in the pipeline take at most at once.
    pub cpu_memory_limit: f64,
    /// How long the data node waits for a job before it builds the
    /// segment itself (`CLIENT_TIMEOUT_SECONDS`, k-NN's
    /// `knn.remote_index_build.client.timeout`); 3,600 s by default.
    pub client_timeout: Duration,
    /// How often the data node asks for a job's status
    /// (`CLIENT_POLL_INTERVAL_SECONDS`, k-NN's
    /// `knn.remote_index_build.poll.interval`); 5 s by default.
    pub poll_interval: Duration,
    /// The threads the data node builds a graph on itself
    /// (`CPU_BUILD_THREADS`): when set, a job the data node would build
    /// sooner itself is refused. Unset by default.
    pub cpu_build: Option<CpuThreads>,
    /// The GPU, as read at start; without one, no job is refused for its
    /// device memory.
    pub device: Option<Device>,
    /// The recall check each job gets (`RECALL_CHECK_QUERIES`,
    /// `RECALL_CHECK_K`, `RECALL_CHECK_MIN`); off by default.
    pub recall_check: RecallCheck,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            max_jobs: 10_000,
            // 1800 s.
            ttl: Duration::from_mins(30),
            gpu_memory_limit: 24.0 * 1e9,
            cpu_memory_limit: 32.0 * 1e9,
            client_timeout: Duration::from_mins(60),
            poll_interval: Duration::from_secs(5),
            cpu_build: None,
            device: None,
            recall_check: RecallCheck::default(),
        }
    }
}

/// The GPU the service builds on, as read when it starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Device {
    /// The device memory free once the build's resources exist, in bytes:
    /// what each build, one at a time, may take.
    pub free_memory: u64,
    /// The shared memory a block may take, in bytes, which decides the
    /// kernels a build runs (`knn_gpu::shared_memory_per_block`).
    pub shared_memory_per_block: usize,
}

/// A job's state: the status response's `task_status`, with its
/// `file_name` or `error_message`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Status {
    /// `RUNNING_INDEX_BUILD`: queued or under way.
    Running,
    /// `COMPLETED_INDEX_BUILD`: the index is stored as `file_name`, next
    /// to the vectors.
    Completed {
        /// The index file's name.
        file_name: String,
    },
    /// `FAILED_INDEX_BUILD`.
    Failed {
        /// Why.
        error_message: String,
    },
}

impl Status {
    /// The status response's `task_status`.
    #[must_use]
    pub fn task_status(&self) -> &'static str {
        match self {
            Self::Running => "RUNNING_INDEX_BUILD",
            Self::Completed { .. } => "COMPLETED_INDEX_BUILD",
            Self::Failed { .. } => "FAILED_INDEX_BUILD",
        }
    }
}

/// A request the service does not take.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum Refusal {
    /// 409: the job id is held for other parameters.
    #[error("Job {job_id} with vector path {vector_path} is already held with other parameters")]
    Conflict {
        /// The request's job id.
        job_id: JobId,
        /// The request's vector path.
        vector_path: String,
    },
    /// 507: the store holds as many jobs as it may. The service's
    /// message.
    #[error(
        "Could not add job {job_id} with vector path {vector_path} to request store, because request store is at capacity"
    )]
    Full {
        /// The request's job id.
        job_id: JobId,
        /// The request's vector path.
        vector_path: String,
    },
    /// 507: the request's memory estimate is over a limit. The service's
    /// message.
    #[error(
        "Insufficient available resources to process job {job_id} with vector path {vector_path}"
    )]
    TooLarge {
        /// The request's job id.
        job_id: JobId,
        /// The request's vector path.
        vector_path: String,
    },
    /// 507: the data node would see the job ended, as it polls, only past
    /// its timeout, so it would wait for nothing.
    #[error(
        "Job {job_id} with vector path {vector_path} would be seen ended in about {seen} s, past the data node's timeout of {timeout} s"
    )]
    Late {
        /// The request's job id.
        job_id: JobId,
        /// The request's vector path.
        vector_path: String,
        /// Seconds until the data node's poll would see the job ended, at
        /// the latest, rounded up.
        seen: u64,
        /// The data node's timeout, in seconds.
        timeout: u64,
    },
    /// 507: the data node would build the segment sooner itself.
    #[error(
        "Job {job_id} with vector path {vector_path} would be back in about {remote} s; the data node builds it in about {cpu} s"
    )]
    CpuSooner {
        /// The request's job id.
        job_id: JobId,
        /// The request's vector path.
        vector_path: String,
        /// Seconds until the data node would have the index file.
        remote: u64,
        /// Seconds the data node would take to build it.
        cpu: u64,
    },
}

impl Refusal {
    /// The HTTP status: 409 or 507.
    #[must_use]
    pub fn status(&self) -> u16 {
        match self {
            Self::Conflict { .. } => 409,
            Self::Full { .. }
            | Self::TooLarge { .. }
            | Self::Late { .. }
            | Self::CpuSooner { .. } => 507,
        }
    }

    /// The answer, as the metrics count it.
    fn answer(&self) -> BuildAnswer {
        match self {
            Self::Conflict { .. } => BuildAnswer::Conflict,
            Self::Full { .. } => BuildAnswer::StoreFull,
            Self::TooLarge { .. } => BuildAnswer::TooLarge,
            Self::Late { .. } => BuildAnswer::Late,
            Self::CpuSooner { .. } => BuildAnswer::CpuSooner,
        }
    }
}

/// The jobs, and the pipeline that runs them.
#[derive(Debug)]
pub struct Service {
    inner: Arc<Mutex<Inner>>,
    pipeline: Pipeline,
    /// When it started, in seconds of Unix time.
    started: f64,
    /// The GPU's memory, where there is a GPU to read.
    gpu: Option<GpuMeter>,
}

#[derive(Debug)]
struct Inner {
    jobs: Jobs,
    /// What lets a new job in, and learns from those that end.
    gate: Gate,
    /// The pipeline's jobs in flight, for the forecast.
    flights: Flights,
    /// The jobs let in and not ended, by their number in the pipeline.
    watches: HashMap<u64, Watch>,
    /// A task per running job, recording its outcome.
    recorders: JoinSet<()>,
    /// What the metrics count.
    counted: Counted,
}

impl Service {
    /// Starts the pipeline on `store` and `builder`, as
    /// [`Pipeline::start`] does.
    ///
    /// # Panics
    ///
    /// When called outside a Tokio runtime.
    pub fn start<B: Builder>(store: Store, builder: B, settings: Settings) -> Self {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "bytes, saturating; a limit below 0 leaves the pipeline its least budget"
        )]
        let host_memory = settings.cpu_memory_limit as u64;
        let pipeline = Pipeline::start(store, builder, host_memory);
        Self {
            inner: Arc::new(Mutex::new(Inner {
                jobs: Jobs::new(settings),
                gate: Gate::new(&settings),
                flights: pipeline.flights(),
                watches: HashMap::new(),
                recorders: JoinSet::new(),
                counted: Counted::default(),
            })),
            pipeline,
            started: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0.0, |since| since.as_secs_f64()),
            gpu: None,
        }
    }

    /// The same service, its metrics reporting the GPU's memory as `meter`
    /// reads it.
    #[must_use]
    pub fn with_gpu_meter(self, meter: GpuMeter) -> Self {
        Self {
            gpu: Some(meter),
            ..self
        }
    }

    /// Takes `request`: returns its job id, of the job held for it or of a
    /// new one, queued.
    ///
    /// A request the service accepts but no build can take, an m below 0
    /// or an ef past 32 bits, gets a job that has failed, as the service
    /// fails it.
    ///
    /// # Errors
    ///
    /// The [`Refusal`], when the request is not taken.
    ///
    /// # Panics
    ///
    /// When called outside a Tokio runtime, or after a thread panicked
    /// holding the job store, a bug.
    pub fn submit(&self, request: BuildRequest) -> Result<JobId, Refusal> {
        let mut guard = self.inner.lock().expect(POISONED);
        let inner = &mut *guard;
        while inner.recorders.try_join_next().is_some() {}
        let vector_path = request.vector_path.clone();
        let timeout = inner.jobs.settings.client_timeout;
        let recall_check = inner.jobs.settings.recall_check;
        let mut started = false;
        let taken = inner
            .jobs
            .take(request, Instant::now(), |job_id, request, now| {
                started = true;
                let job = match request.job() {
                    Ok(job) => Job {
                        recall_check,
                        ..job
                    },
                    Err(error) => {
                        tracing::info!(%job_id, vector_path, %error, "job failed at once");
                        count_failure(&mut inner.counted, &error);
                        return Ok(State::ended(Err(error), now));
                    }
                };
                // Under the store's lock, so the next request's forecast
                // counts this job.
                let in_flight = inner.flights.in_flight();
                let admitted = inner
                    .gate
                    .admit(job_id, &vector_path, &job, &in_flight, now)?;
                tracing::info!(
                    %job_id,
                    vector_path,
                    forecast_s = admitted.ends_in.as_secs_f64(),
                    "job queued"
                );
                let learned = job.clone();
                let (taken, forecast) = (now, admitted.ends_in);
                let handle = self.pipeline.submit(job);
                let id = handle.id();
                let canceler = handle.canceler();
                let reason = Arc::new(OnceLock::new());
                inner.watches.insert(
                    id,
                    Watch {
                        job_id: job_id.clone(),
                        vector_path: vector_path.clone(),
                        admitted: now,
                        canceler: canceler.clone(),
                        reason: Arc::clone(&reason),
                    },
                );
                let stop = canceler.clone();
                let shared = Arc::clone(&self.inner);
                let (job_id, vector_path) = (job_id.clone(), vector_path.clone());
                // The data node's wait starts as this request returns; a
                // timeout too long to count to never passes.
                let deadline = now.checked_add(timeout);
                inner.recorders.spawn(async move {
                    let outcome = handle.outcome();
                    tokio::pin!(outcome);
                    let outcome = tokio::select! {
                        outcome = &mut outcome => outcome,
                        () = past(deadline) => {
                            let past = JobError::PastDeadline {
                                seconds: timeout.as_secs(),
                            };
                            if reason.set(past).is_ok() {
                                tracing::info!(%job_id, vector_path, "past the data node's timeout: canceling the job");
                                stop.cancel();
                            }
                            // An upload that sent its last request first
                            // has stored the index.
                            outcome.await
                        }
                    };
                    // A cancel says why, when the service made it.
                    let outcome = outcome.map_err(|error| match (error, reason.get()) {
                        (JobError::Canceled, Some(why)) => why.clone(),
                        (error, _) => error,
                    });
                    let ended = Instant::now();
                    log_end(&job_id, &vector_path, &outcome);
                    let times = Times {
                        taken,
                        forecast,
                        ended,
                    };
                    let mut inner = shared.lock().expect(POISONED);
                    inner.record_end(id, &job_id, &learned, times, outcome);
                });
                Ok(State::Running { canceler, id })
            });
        let answer = match &taken {
            Ok(_) if started => BuildAnswer::Accepted,
            Ok(_) => BuildAnswer::Held,
            Err(refusal) => refusal.answer(),
        };
        inner.counted.build_requests.add(answer, 1);
        match &taken {
            Ok(job_id) if !started => tracing::info!(%job_id, vector_path, "job held already"),
            Ok(_) => {}
            Err(refusal) => tracing::warn!(%refusal, "build request refused"),
        }
        taken
    }

    /// The status of the job `job_id`, or `None` when no job of that id is
    /// held: never taken, or expired.
    ///
    /// # Panics
    ///
    /// After a thread panicked holding the job store, a bug.
    #[must_use]
    pub fn status(&self, job_id: &JobId) -> Option<Status> {
        self.inner
            .lock()
            .expect(POISONED)
            .jobs
            .status(job_id, Instant::now())
    }

    /// Counts a build request refused before it reached the service: a
    /// body the API could not read.
    ///
    /// # Panics
    ///
    /// After a thread panicked holding the job store, a bug.
    pub fn count_invalid(&self) {
        let mut inner = self.inner.lock().expect(POISONED);
        inner.counted.build_requests.add(BuildAnswer::Invalid, 1);
    }

    /// Every metric's value now, but the requests answered, which are the
    /// API's to count.
    ///
    /// # Panics
    ///
    /// After a thread panicked holding the job store, the pipeline's job
    /// list or the GPU meter's readings, a bug.
    #[must_use]
    pub fn metrics(&self) -> Snapshot {
        // Read before the lock is taken, as it reads a file.
        let resident_memory = memory::resident();
        let gpu_memory = self.gpu.as_ref().and_then(GpuMeter::memory);
        let now = Instant::now();
        let inner = self.inner.lock().expect(POISONED);
        let in_flight = inner.flights.in_flight();
        let mut jobs_in_flight = Family::default();
        // A job the service dropped has ended, though it stays in the
        // pipeline until its stage lets it go.
        let running = |id| {
            let watch: Option<&Watch> = inner.watches.get(&id);
            watch.is_some_and(|watch| inner.jobs.runs(&watch.job_id, id))
        };
        for flight in in_flight.iter().filter(|flight| running(flight.id)) {
            jobs_in_flight.add(flight.stage.into(), 1);
        }
        let ratios = inner.gate.ratios();
        let mut stage_model_ratio = Family::new(1.0);
        for (stage, ratio) in [
            (TimedStage::Download, ratios.download),
            (TimedStage::Build, ratios.build),
            (TimedStage::Projection, ratios.projection),
            (TimedStage::RecallCheck, ratios.check),
            (TimedStage::Upload, ratios.upload),
        ] {
            stage_model_ratio.set(stage, ratio);
        }
        let (host_budget, host_budget_used) = inner.flights.host_budget();
        let settings = &inner.jobs.settings;
        Snapshot {
            counted: inner.counted.clone(),
            jobs_in_flight,
            jobs_held: inner.jobs.held(now),
            jobs_held_limit: u64::try_from(settings.max_jobs).unwrap_or(u64::MAX),
            stage_model_ratio,
            drain_seconds: inner.gate.drain(now, &in_flight).as_secs_f64(),
            host_budget,
            host_budget_used,
            resident_memory,
            start_time_seconds: self.started,
            gpu_memory,
            ..Snapshot::default()
        }
    }

    /// Every job held, in the order they were taken: the debug list of
    /// `GET /_jobs`.
    ///
    /// # Panics
    ///
    /// After a thread panicked holding the job store, a bug.
    #[must_use]
    pub fn jobs(&self) -> Vec<Listed> {
        self.inner.lock().expect(POISONED).jobs.list(Instant::now())
    }

    /// Cancels every running job, then waits for the pipeline to stop.
    /// Queued jobs end at once, and downloads and uploads stop, the uploads
    /// aborted so they store nothing; but a build under way cannot be
    /// stopped, so this returns only once it ends, to be thrown away. A
    /// stopping container may not have that long: the binary bounds the
    /// wait. The data node builds the segments of the jobs canceled this
    /// way itself.
    ///
    /// # Panics
    ///
    /// After a thread panicked holding the job store, a bug.
    pub async fn close(self) {
        let recorders = {
            let mut inner = self.inner.lock().expect(POISONED);
            for canceler in inner.jobs.running() {
                canceler.cancel();
            }
            std::mem::take(&mut inner.recorders)
        };
        self.pipeline.close().await;
        recorders.join_all().await;
    }
}

/// Returns once `deadline` has passed; never when there is none.
async fn past(deadline: Option<Instant>) {
    match deadline {
        Some(deadline) => tokio::time::sleep_until(deadline.into()).await,
        None => std::future::pending().await,
    }
}

/// A job let in, as the service watches it until it ends.
#[derive(Debug)]
struct Watch {
    job_id: JobId,
    vector_path: String,
    /// When it was let in: the data node's wait starts about then.
    admitted: Instant,
    canceler: Canceler,
    /// Why it was canceled, set by whoever cancels it first.
    reason: Arc<OnceLock<JobError>>,
}

impl Inner {
    /// Records how the job `job_id`, number `id` in the pipeline, ended:
    /// what the forecast learns from it, its status, and its count, once;
    /// then forecasts the others again.
    fn record_end(
        &mut self,
        id: u64,
        job_id: &JobId,
        job: &Job,
        times: Times,
        outcome: Result<Built, JobError>,
    ) {
        if let Ok(built) = &outcome {
            self.gate.learn(job, &built.timings);
        }
        // A job the service dropped first is ended, and counted, already;
        // and the job of its id may by now be another run of the request,
        // taken once the dropped one expired, which this report is not of.
        if self.jobs.runs(job_id, id) {
            match &outcome {
                Ok(built) => count_completion(&mut self.counted, job, built, times),
                Err(error) => count_failure(&mut self.counted, error),
            }
        }
        self.jobs.end(job_id, id, outcome, times.ended);
        self.watches.remove(&id);
        // What this one taught, and the room it left, may move the others'
        // forecasts.
        self.recheck(times.ended);
    }

    /// Forecasts the jobs not yet building again, and cancels those that
    /// would now miss the data node's timeout. Each fails at once: one
    /// waiting in the build slot would otherwise end only when the builder
    /// takes it, and none of them can store an index any more, the builder
    /// and the upload each checking for the cancel before they start.
    fn recheck(&mut self, now: Instant) {
        let overdue = self.gate.overdue(now, &self.flights.in_flight(), |id| {
            self.watches.get(&id).map(|watch| watch.admitted)
        });
        for id in overdue {
            let Some(watch) = self.watches.get(&id) else {
                continue;
            };
            let reason = JobError::WouldMissDeadline {
                seconds: self.gate.timeout_seconds(),
            };
            if watch.reason.set(reason.clone()).is_ok() {
                tracing::info!(
                    job_id = %watch.job_id,
                    vector_path = watch.vector_path,
                    "forecast to miss the data node's timeout: canceling the job"
                );
                watch.canceler.cancel();
                // The pipeline's report, when it comes, finds it ended.
                let job_id = watch.job_id.clone();
                if self.jobs.runs(&job_id, id) {
                    count_failure(&mut self.counted, &reason);
                }
                self.jobs.end(&job_id, id, Err(reason), now);
            }
        }
    }
}

/// When a job was taken, the seconds it was forecast to take from then,
/// and when it ended.
#[derive(Debug, Clone, Copy)]
struct Times {
    taken: Instant,
    forecast: Duration,
    ended: Instant,
}

/// Counts a job that ended with its index stored: its vectors, its times
/// and its recall.
fn count_completion(counted: &mut Counted, job: &Job, built: &Built, times: Times) {
    let seconds = |span: Span| span.duration().as_secs_f64();
    let timings = &built.timings;
    counted.jobs_ended.add(Outcome::Completed, 1);
    counted
        .vectors_built
        .add(job.data_type.into(), u64::from(job.doc_count));
    let took = times.ended.saturating_duration_since(times.taken);
    counted.job_seconds.observe(took.as_secs_f64());
    let queued = timings
        .download
        .start
        .saturating_duration_since(times.taken);
    counted.queue_seconds.observe(queued.as_secs_f64());
    let stages = [
        (TimedStage::Download, Some(timings.download)),
        (TimedStage::Build, Some(timings.build)),
        (TimedStage::Projection, timings.projection),
        (TimedStage::RecallCheck, timings.check),
        (TimedStage::Upload, Some(timings.upload)),
    ];
    for (stage, span) in stages {
        if let Some(span) = span {
            counted.stage_seconds.observe(stage, seconds(span));
        }
    }
    // A forecast of nothing has no ratio: the histogram leaves out the
    // infinity.
    counted
        .forecast_ratio
        .observe(took.as_secs_f64() / times.forecast.as_secs_f64());
    if let Some(checked) = &built.recall {
        counted.recall.observe(checked.recall);
    }
}

/// Counts a job that failed; one failed for its recall, with the recall.
fn count_failure(counted: &mut Counted, error: &JobError) {
    counted.jobs_ended.add(error.outcome(), 1);
    if let JobError::LowRecall { recall, .. } = error {
        counted.recall.observe(*recall);
    }
}

/// A job's end in the logs: its status, and at `DEBUG` its stages' times.
fn log_end(job_id: &JobId, vector_path: &str, outcome: &Result<Built, JobError>) {
    match outcome {
        Ok(built) => {
            let checked = built.recall.as_ref();
            tracing::info!(
                %job_id,
                vector_path,
                file_name = built.file_name,
                recall = checked.map(|c| c.recall),
                recall_queries = checked.map(|c| c.queries),
                recall_k = checked.map(|c| c.k),
                "job completed"
            );
            let seconds = |span: crate::Span| span.duration().as_secs_f64();
            tracing::debug!(
                %job_id,
                download_s = seconds(built.timings.download),
                build_s = seconds(built.timings.build),
                projection_s = built.timings.projection.map(seconds),
                recall_check_s = built.timings.check.map(seconds),
                upload_s = seconds(built.timings.upload),
                total_s = built
                    .timings
                    .upload
                    .end
                    .duration_since(built.timings.download.start)
                    .as_secs_f64(),
                "job stage times"
            );
        }
        Err(error) => tracing::warn!(%job_id, vector_path, %error, "job failed"),
    }
}

/// What to do with a request.
#[derive(Debug, PartialEq, Eq)]
enum Admission {
    /// Its job is held: answer with its id.
    Held(JobId),
    /// Start a job of this id.
    New(JobId),
}

/// A job held, as `GET /_jobs` lists it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Listed {
    /// Its id.
    pub job_id: JobId,
    /// Its state.
    pub status: Status,
    /// The request's vector path.
    pub vector_path: String,
    /// The request's tenant id.
    pub tenant_id: String,
}

/// The jobs held, by id.
#[derive(Debug)]
struct Jobs {
    settings: Settings,
    held: HashMap<JobId, Held>,
    /// The number the next job taken gets: jobs are listed in that order.
    next: u64,
}

#[derive(Debug)]
struct Held {
    request: BuildRequest,
    state: State,
    number: u64,
}

#[derive(Debug)]
enum State {
    /// In the pipeline, as its job number `id`.
    Running {
        canceler: Canceler,
        id: u64,
    },
    Ended {
        status: Status,
        at: Instant,
    },
}

impl State {
    /// Whether this is the run that is the pipeline's job number `id`,
    /// not ended.
    fn runs(&self, id: u64) -> bool {
        matches!(self, Self::Running { id: run, .. } if *run == id)
    }

    fn ended(outcome: Result<Built, JobError>, at: Instant) -> Self {
        let status = match outcome {
            Ok(built) => Status::Completed {
                file_name: built.file_name,
            },
            Err(error) => Status::Failed {
                error_message: error.to_string(),
            },
        };
        Self::Ended { status, at }
    }
}

impl Jobs {
    fn new(settings: Settings) -> Self {
        Self {
            settings,
            held: HashMap::new(),
            next: 0,
        }
    }

    /// Decides on `request`, in the service's order: the job held for it,
    /// then the store's size, then the memory estimate. A job that has
    /// expired counts as gone.
    fn admit(&mut self, request: &BuildRequest, now: Instant) -> Result<Admission, Refusal> {
        let job_id = request.job_id();
        self.expire(&job_id, now);
        if let Some(held) = self.held.get(&job_id) {
            if held.request == *request {
                return Ok(Admission::Held(job_id));
            }
            return Err(Refusal::Conflict {
                job_id,
                vector_path: request.vector_path.clone(),
            });
        }
        if self.held.len() >= self.settings.max_jobs {
            self.held
                .retain(|_, held| !held.expired(self.settings.ttl, now));
        }
        if self.held.len() >= self.settings.max_jobs {
            return Err(Refusal::Full {
                job_id,
                vector_path: request.vector_path.clone(),
            });
        }
        let (gpu, cpu) = estimate(request);
        if gpu > self.settings.gpu_memory_limit || cpu > self.settings.cpu_memory_limit {
            return Err(Refusal::TooLarge {
                job_id,
                vector_path: request.vector_path.clone(),
            });
        }
        Ok(Admission::New(job_id))
    }

    /// Takes `request`: its job held, or a new one, held in the state
    /// `start` puts it in, unless `start` refuses it. The check and the
    /// start are one call on one borrow, so two identical requests at once
    /// start one job.
    fn take(
        &mut self,
        request: BuildRequest,
        now: Instant,
        start: impl FnOnce(&JobId, &BuildRequest, Instant) -> Result<State, Refusal>,
    ) -> Result<JobId, Refusal> {
        let job_id = match self.admit(&request, now)? {
            Admission::Held(job_id) => return Ok(job_id),
            Admission::New(job_id) => job_id,
        };
        let state = start(&job_id, &request, now)?;
        let number = self.next;
        self.next += 1;
        self.held.insert(
            job_id.clone(),
            Held {
                request,
                state,
                number,
            },
        );
        Ok(job_id)
    }

    /// Whether the job `job_id` is held and runs as the pipeline's job
    /// number `id`: not ended, and not a later run of the same request,
    /// taken once the one before had ended and expired.
    fn runs(&self, job_id: &JobId, id: u64) -> bool {
        self.held
            .get(job_id)
            .is_some_and(|held| held.state.runs(id))
    }

    /// The jobs held that have not expired.
    fn held(&self, now: Instant) -> u64 {
        let held = self.held.values();
        let live = held.filter(|held| !held.expired(self.settings.ttl, now));
        u64::try_from(live.count()).unwrap_or(u64::MAX)
    }

    /// Records the end of the job `job_id` running as the pipeline's job
    /// number `id`. A job that has ended stays as it ended, and so does a
    /// later run of the same request: the pipeline reports a job the
    /// service dropped only when its stage lets it go, by when the dropped
    /// one may have expired and its request been taken again.
    fn end(&mut self, job_id: &JobId, id: u64, outcome: Result<Built, JobError>, at: Instant) {
        // A running job is never removed, so its own run finds it there.
        if let Some(held) = self.held.get_mut(job_id)
            && held.state.runs(id)
        {
            held.state = State::ended(outcome, at);
        }
    }

    fn status(&mut self, job_id: &JobId, now: Instant) -> Option<Status> {
        self.expire(job_id, now);
        self.held.get(job_id).map(Held::status)
    }

    /// The jobs not expired, in the order they were taken.
    fn list(&mut self, now: Instant) -> Vec<Listed> {
        self.held
            .retain(|_, held| !held.expired(self.settings.ttl, now));
        let mut held: Vec<(&JobId, &Held)> = self.held.iter().collect();
        held.sort_by_key(|(_, held)| held.number);
        held.into_iter()
            .map(|(job_id, held)| Listed {
                job_id: job_id.clone(),
                status: held.status(),
                vector_path: held.request.vector_path.clone(),
                tenant_id: held.request.tenant_id.clone(),
            })
            .collect()
    }

    fn running(&self) -> impl Iterator<Item = &Canceler> {
        self.held.values().filter_map(|held| match &held.state {
            State::Running { canceler, .. } => Some(canceler),
            State::Ended { .. } => None,
        })
    }

    /// Drops the job `job_id` if it has expired.
    fn expire(&mut self, job_id: &JobId, now: Instant) {
        if self
            .held
            .get(job_id)
            .is_some_and(|held| held.expired(self.settings.ttl, now))
        {
            self.held.remove(job_id);
        }
    }
}

impl Held {
    fn status(&self) -> Status {
        match &self.state {
            State::Running { .. } => Status::Running,
            State::Ended { status, .. } => status.clone(),
        }
    }

    /// Ended `ttl` ago or longer.
    fn expired(&self, ttl: Duration, now: Instant) -> bool {
        match self.state {
            State::Running { .. } => false,
            State::Ended { at, .. } => now.saturating_duration_since(at) >= ttl,
        }
    }
}

/// The upstream service's estimate of the GPU and CPU memory a request
/// takes, in bytes (`calculate_memory_requirements`), computed in the same
/// order: the integer terms exactly, then rounded to f64 where Python
/// turns them into floats.
fn estimate(request: &BuildRequest) -> (f64, f64) {
    // EDGE_SIZE, MEMORY_OVERHEAD_FACTOR and GPU_SCALING_FACTOR.
    const EDGE_SIZE: i128 = 8;
    const OVERHEAD: f64 = 1.1;
    const GPU_SCALING: f64 = 0.5;
    let size: u8 = match request.data_type {
        DataType::Float => 4,
        DataType::HalfFloat => 2,
        DataType::Byte | DataType::Binary => 1,
    };
    let (dimension, doc_count) = (request.dimension, request.doc_count);
    // dimension * size + m * EDGE_SIZE: within i128 for any u64 and i64.
    let per_vector = i128::from(dimension) * i128::from(size) + i128::from(request.m) * EDGE_SIZE;
    #[expect(
        clippy::cast_precision_loss,
        reason = "Python's int-to-float conversion, rounded to nearest as `as` rounds"
    )]
    let (per_vector, doc_count_f, vectors) = (
        per_vector as f64,
        doc_count as f64,
        // dimension * doc_count * size: the product of two u64 fits u128,
        // and times 2 or 4 a float is exact, so this rounds once, as the
        // exact integer converted does.
        (u128::from(dimension) * u128::from(doc_count)) as f64 * f64::from(size),
    );
    // The index's CPU memory; the GPU's is half of it.
    let index = per_vector * OVERHEAD * doc_count_f;
    (index * GPU_SCALING, index + vectors)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::time::{Duration, Instant};

    use serde_json::{Map, Value};
    use tokio::task::JoinSet;

    use super::{
        Admission, Inner, Jobs, Refusal, Settings, State, Status, Times, count_failure, estimate,
    };
    use crate::gate::Gate;
    use crate::metrics::{Counted, Outcome};
    use crate::request::{BuildRequest, JobId};
    use crate::{
        Built, Canceler, DataType, Flights, JobError, RecallCheck, SpaceType, Span, Timings,
    };

    /// `tools/oracle/capacity.py`'s record of the service's estimates.
    const CAPACITY: &str = include_str!("../tests/data/capacity.json");

    const TTL: Duration = Duration::from_mins(30);

    /// The pipeline's number of the one run the store's own tests give
    /// each job.
    const RUN: u64 = 1;

    /// A job running as the pipeline's job number `id`.
    fn running(id: u64) -> State {
        State::Running {
            canceler: Canceler::detached(),
            id,
        }
    }

    fn request(path: &str, dimension: u64) -> BuildRequest {
        BuildRequest {
            container_name: "bucket".into(),
            vector_path: path.into(),
            doc_id_path: "a/_0_f.knndid".into(),
            tenant_id: String::new(),
            dimension,
            doc_count: 10,
            data_type: DataType::Float,
            space_type: SpaceType::L2,
            ef_construction: 100,
            ef_search: 100,
            m: 16,
            project_link_to_base_layer: false,
            other_algorithm_parameters: Map::new(),
            skip_stored_vectors: false,
        }
    }

    fn jobs(max_jobs: usize) -> Jobs {
        Jobs::new(Settings {
            max_jobs,
            ..Settings::default()
        })
    }

    /// Takes `request` at `now` as a new job, running.
    fn start(jobs: &mut Jobs, request: &BuildRequest, now: Instant) -> JobId {
        let mut started = false;
        let job_id = jobs
            .take(request.clone(), now, |_, _, _| {
                started = true;
                Ok(running(RUN))
            })
            .unwrap();
        assert!(started, "{} held already", request.vector_path);
        job_id
    }

    fn built(file_name: &str, at: Instant) -> Built {
        let span = Span { start: at, end: at };
        Built {
            file_name: file_name.into(),
            timings: Timings {
                download: span,
                build: span,
                projection: None,
                check: None,
                upload: span,
                check_wait: Duration::ZERO,
            },
            recall: None,
        }
    }

    /// A recorded case's data type.
    fn data_type(name: &Value) -> DataType {
        match name.as_str().unwrap() {
            "float" => DataType::Float,
            "half_float" => DataType::HalfFloat,
            "byte" => DataType::Byte,
            "binary" => DataType::Binary,
            other => panic!("data type {other}"),
        }
    }

    #[test]
    fn estimates_as_the_service_does() {
        let table: Value = serde_json::from_str(CAPACITY).unwrap();
        let limits = table["limits"].as_array().unwrap();
        let estimates = table["estimates"].as_array().unwrap();
        assert!(estimates.len() > 400, "{} estimates", estimates.len());
        let mut decided_by_gpu = 0;
        for case in estimates {
            let mut request = request("a/_0_f.knnvec", case["dimension"].as_u64().unwrap());
            request.doc_count = case["doc_count"].as_u64().unwrap();
            request.m = case["m"].as_i64().unwrap();
            request.data_type = data_type(&case["data_type"]);
            let (gpu, cpu) = estimate(&request);
            let want = (case["gpu"].as_f64().unwrap(), case["cpu"].as_f64().unwrap());
            assert_eq!(
                (gpu.to_bits(), cpu.to_bits()),
                (want.0.to_bits(), want.1.to_bits()),
                "{case}: {gpu} {cpu}"
            );
            for (pair, fits) in limits.iter().zip(case["fits"].as_array().unwrap()) {
                let gpu_memory_limit = pair["gpu"].as_f64().unwrap();
                let cpu_memory_limit = pair["cpu"].as_f64().unwrap();
                let mut jobs = Jobs::new(Settings {
                    gpu_memory_limit,
                    cpu_memory_limit,
                    ..Settings::default()
                });
                let admitted = jobs.admit(&request, Instant::now());
                assert_eq!(
                    matches!(admitted, Ok(Admission::New(_))),
                    fits.as_bool().unwrap(),
                    "{case} under {pair}: {admitted:?}"
                );
                if let Err(refusal) = admitted {
                    assert!(matches!(refusal, Refusal::TooLarge { .. }), "{refusal:?}");
                }
                if gpu > gpu_memory_limit && cpu <= cpu_memory_limit {
                    decided_by_gpu += 1;
                }
            }
        }
        assert!(decided_by_gpu > 0, "no case where the GPU limit decides");
    }

    /// Python's test, `estimate > limit`, at its edge: a request whose
    /// estimate equals a limit fits.
    #[test]
    fn takes_an_estimate_equal_to_a_limit() {
        let table: Value = serde_json::from_str(CAPACITY).unwrap();
        for case in table["at_limits"].as_array().unwrap() {
            let mut request = request("a/_0_f.knnvec", case["dimension"].as_u64().unwrap());
            request.doc_count = case["doc_count"].as_u64().unwrap();
            request.m = case["m"].as_i64().unwrap();
            request.data_type = data_type(&case["data_type"]);
            let mut jobs = Jobs::new(Settings {
                gpu_memory_limit: case["limits"]["gpu"].as_f64().unwrap(),
                cpu_memory_limit: case["limits"]["cpu"].as_f64().unwrap(),
                ..Settings::default()
            });
            let admitted = jobs.admit(&request, Instant::now());
            assert_eq!(
                matches!(admitted, Ok(Admission::New(_))),
                case["fits"].as_bool().unwrap(),
                "{case}: {admitted:?}"
            );
        }
    }

    #[test]
    fn defaults_to_the_services_settings() {
        let table: Value = serde_json::from_str(CAPACITY).unwrap();
        let defaults = &table["defaults"];
        let ttl = Duration::from_secs(defaults["ttl_seconds"].as_u64().unwrap());
        assert_eq!(
            Settings::default(),
            Settings {
                max_jobs: usize::try_from(defaults["max_jobs"].as_u64().unwrap()).unwrap(),
                ttl,
                gpu_memory_limit: defaults["gpu_memory_limit"].as_f64().unwrap(),
                cpu_memory_limit: defaults["cpu_memory_limit"].as_f64().unwrap(),
                // k-NN's own defaults (KNNSettings at 3.9.0.0).
                client_timeout: Duration::from_secs(3_600),
                poll_interval: Duration::from_secs(5),
                cpu_build: None,
                device: None,
                recall_check: RecallCheck::default(),
            }
        );
    }

    #[test]
    fn refuses_with_the_services_messages() {
        let table: Value = serde_json::from_str(CAPACITY).unwrap();
        let refusals = &table["refusals"];
        let now = Instant::now();
        let mut full = jobs(1);
        start(&mut full, &request("a/_0_held.knnvec", 8), now);
        let refused = full
            .admit(
                &request(refusals["full"]["vector_path"].as_str().unwrap(), 8),
                now,
            )
            .unwrap_err();
        assert_eq!(refused.status(), 507);
        assert_eq!(refused.to_string(), refusals["full"]["message"]);

        let mut large = request(refusals["too_large"]["vector_path"].as_str().unwrap(), 768);
        large.doc_count = 100_000_000;
        let refused = jobs(1).admit(&large, now).unwrap_err();
        assert_eq!(refused.status(), 507);
        assert_eq!(refused.to_string(), refusals["too_large"]["message"]);
    }

    #[test]
    fn answers_a_retry_with_the_job_held() {
        let now = Instant::now();
        let mut jobs = jobs(10);
        let first = request("a/_0_f.knnvec", 8);
        let job_id = start(&mut jobs, &first, now);
        assert_eq!(job_id, first.job_id());
        let held = Ok(Admission::Held(job_id.clone()));
        assert_eq!(jobs.admit(&first, now), held);
        jobs.end(&job_id, RUN, Err(JobError::Canceled), now);
        assert_eq!(jobs.admit(&first, now), held, "a failed job is held too");
        assert_eq!(
            jobs.status(&job_id, now),
            Some(Status::Failed {
                error_message: "the job was canceled".into()
            })
        );
    }

    #[test]
    fn starts_one_job_however_often_it_is_taken() {
        let now = Instant::now();
        let mut jobs = jobs(10);
        let request = request("a/_0_f.knnvec", 8);
        let job_id = start(&mut jobs, &request, now);
        let mut starts = 0;
        for _ in 0..3 {
            let taken = jobs.take(request.clone(), now, |_, _, _| {
                starts += 1;
                Ok(running(RUN))
            });
            assert_eq!(taken, Ok(job_id.clone()));
        }
        assert_eq!(starts, 0);
        assert_eq!(jobs.running().count(), 1);
    }

    #[test]
    fn refuses_the_same_path_with_other_parameters() {
        let now = Instant::now();
        let mut jobs = jobs(10);
        let first = request("a/_0_f.knnvec", 8);
        start(&mut jobs, &first, now);
        let mut other = first.clone();
        other
            .other_algorithm_parameters
            .insert("x".into(), 1.into());
        let refused = jobs.admit(&other, now).unwrap_err();
        assert_eq!(
            refused,
            Refusal::Conflict {
                job_id: first.job_id(),
                vector_path: first.vector_path.clone(),
            }
        );
        assert_eq!(refused.status(), 409);
        // Another tenant is another job.
        let mut tenant = other.clone();
        tenant.tenant_id = "b".into();
        assert!(matches!(jobs.admit(&tenant, now), Ok(Admission::New(_))));
    }

    #[test]
    fn expires_a_job_ttl_after_it_ends() {
        let start_at = Instant::now();
        let mut jobs = jobs(10);
        let first = request("a/_0_f.knnvec", 8);
        let job_id = start(&mut jobs, &first, start_at);
        // Running, long past the TTL: held.
        let later = start_at + TTL * 10;
        assert_eq!(jobs.status(&job_id, later), Some(Status::Running));
        jobs.end(&job_id, RUN, Ok(built("_0_f.faiss", later)), later);
        let completed = Some(Status::Completed {
            file_name: "_0_f.faiss".into(),
        });
        let before = (later + TTL).checked_sub(Duration::from_nanos(1)).unwrap();
        assert_eq!(jobs.status(&job_id, before), completed);
        assert_eq!(jobs.status(&job_id, later + TTL), None);
        // Gone, so the same path with other parameters is a new job.
        let mut other = first.clone();
        other.dimension = 16;
        assert_eq!(
            jobs.admit(&other, later + TTL),
            Ok(Admission::New(first.job_id()))
        );
    }

    #[test]
    fn counts_only_jobs_not_expired_toward_the_size() {
        let now = Instant::now();
        let mut jobs = jobs(2);
        let first = start(&mut jobs, &request("a/_0_a.knnvec", 8), now);
        start(&mut jobs, &request("a/_0_b.knnvec", 8), now);
        let third = request("a/_0_c.knnvec", 8);
        assert!(matches!(jobs.admit(&third, now), Err(Refusal::Full { .. })));
        // A retry of a job held is answered all the same.
        assert!(matches!(
            jobs.admit(&request("a/_0_a.knnvec", 8), now),
            Ok(Admission::Held(_))
        ));
        jobs.end(&first, RUN, Err(JobError::Canceled), now);
        let before = (now + TTL).checked_sub(Duration::from_nanos(1)).unwrap();
        assert!(matches!(
            jobs.admit(&third, before),
            Err(Refusal::Full { .. })
        ));
        assert_eq!(jobs.held(before), 2);
        assert_eq!(jobs.held(now + TTL), 1, "an expired job is not counted");
        assert!(matches!(
            jobs.admit(&third, now + TTL),
            Ok(Admission::New(_))
        ));
    }

    #[test]
    fn checks_the_size_before_the_estimate() {
        let now = Instant::now();
        let mut jobs = jobs(1);
        start(&mut jobs, &request("a/_0_a.knnvec", 8), now);
        let mut large = request("a/_0_b.knnvec", 768);
        large.doc_count = 100_000_000;
        assert!(matches!(jobs.admit(&large, now), Err(Refusal::Full { .. })));
    }

    #[test]
    fn records_how_a_job_ended() {
        let now = Instant::now();
        let mut jobs = jobs(10);
        let job_id = start(&mut jobs, &request("a/_0_f.knnvec", 8), now);
        assert!(jobs.runs(&job_id, RUN));
        assert!(!jobs.runs(&job_id, RUN + 1));
        assert!(!jobs.runs(&JobId::from("no-such-job"), RUN));
        // Another run's end is not this one's.
        jobs.end(&job_id, RUN + 1, Err(JobError::Stopped), now);
        assert_eq!(jobs.status(&job_id, now), Some(Status::Running));
        jobs.end(&job_id, RUN, Ok(built("_0_f.faiss", now)), now);
        let completed = Some(Status::Completed {
            file_name: "_0_f.faiss".into(),
        });
        assert_eq!(jobs.status(&job_id, now), completed);
        assert!(!jobs.runs(&job_id, RUN));
        // A second end changes nothing.
        jobs.end(&job_id, RUN, Err(JobError::Stopped), now);
        assert_eq!(jobs.status(&job_id, now), completed);
        assert_eq!(jobs.running().count(), 0);
        assert_eq!(jobs.status(&JobId::from("no-such-job"), now), None);
    }

    /// The jobs `inner` counts as having ended as `outcome`.
    fn ended_as(inner: &Inner, outcome: Outcome) -> u64 {
        let mut counts = inner.counted.jobs_ended.iter();
        *counts.find(|(known, _)| *known == outcome).unwrap().1
    }

    #[test]
    fn keeps_a_job_taken_again_apart_from_its_dropped_run() {
        let taken = Instant::now();
        let settings = Settings::default();
        let mut inner = Inner {
            jobs: Jobs::new(settings),
            gate: Gate::new(&settings),
            flights: Flights::idle(),
            watches: HashMap::new(),
            recorders: JoinSet::new(),
            counted: Counted::default(),
        };
        let request = request("a/_0_f.knnvec", 8);
        let job = request.job().unwrap();
        let take = |jobs: &mut Jobs, id, now| {
            let mut started = false;
            let job_id = jobs.take(request.clone(), now, |_, _, _| {
                started = true;
                Ok(running(id))
            });
            assert!(started, "run {id} held already");
            job_id.unwrap()
        };
        // The first run, dropped as the forecast drops one: ended and
        // counted at once, the pipeline holding it still.
        let job_id = take(&mut inner.jobs, 1, taken);
        let late = JobError::WouldMissDeadline { seconds: 4 };
        count_failure(&mut inner.counted, &late);
        inner.jobs.end(&job_id, 1, Err(late.clone()), taken);
        // Expired before the pipeline let it go, and its request taken
        // again: the same job id, running.
        let again = taken + settings.ttl;
        assert_eq!(take(&mut inner.jobs, 2, again), job_id);
        let times = Times {
            taken,
            forecast: Duration::from_secs(1),
            ended: again,
        };
        // The first run's report, when it comes, is not of the job running
        // now: it ends nothing and counts nothing.
        inner.record_end(1, &job_id, &job, times, Err(late));
        assert_eq!(inner.jobs.status(&job_id, again), Some(Status::Running));
        assert!(inner.jobs.runs(&job_id, 2));
        assert_eq!(ended_as(&inner, Outcome::ForecastLate), 1);
        // The second run's own end is counted.
        let stored = built("_0_f.faiss", again);
        inner.record_end(2, &job_id, &job, times, Ok(stored));
        let completed = Some(Status::Completed {
            file_name: "_0_f.faiss".into(),
        });
        assert_eq!(inner.jobs.status(&job_id, again), completed);
        assert_eq!(ended_as(&inner, Outcome::ForecastLate), 1);
        assert_eq!(ended_as(&inner, Outcome::Completed), 1);
    }

    #[test]
    fn lists_jobs_in_the_order_taken() {
        let now = Instant::now();
        let mut jobs = jobs(10);
        let paths = ["a/_0_c.knnvec", "a/_0_a.knnvec", "a/_0_b.knnvec"];
        let ids: Vec<JobId> = paths
            .iter()
            .map(|path| start(&mut jobs, &request(path, 8), now))
            .collect();
        jobs.end(&ids[1], RUN, Err(JobError::Canceled), now);
        let listed = jobs.list(now);
        let order: Vec<&str> = listed.iter().map(|job| job.vector_path.as_str()).collect();
        assert_eq!(order, paths);
        assert_eq!(listed[0].job_id, ids[0]);
        assert_eq!(listed[0].status, Status::Running);
        assert!(matches!(listed[1].status, Status::Failed { .. }));
        // An expired job is not listed; a running one stays.
        let listed = jobs.list(now + TTL);
        let order: Vec<&str> = listed.iter().map(|job| job.vector_path.as_str()).collect();
        assert_eq!(order, [paths[0], paths[2]]);
    }

    #[test]
    fn names_each_task_status() {
        assert_eq!(Status::Running.task_status(), "RUNNING_INDEX_BUILD");
        let completed = Status::Completed {
            file_name: String::new(),
        };
        assert_eq!(completed.task_status(), "COMPLETED_INDEX_BUILD");
        let failed = Status::Failed {
            error_message: String::new(),
        };
        assert_eq!(failed.task_status(), "FAILED_INDEX_BUILD");
    }
}
