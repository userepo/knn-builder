//! Jobs flow through three stages, one job at a time in each: the
//! download, the build on the builder's own thread, then the write and
//! upload. So the next job downloads while one builds, and the last one
//! uploads meanwhile. A stage takes the next stage's one slot before it
//! takes up a job, so a finished job waits where it is until the next
//! stage has room, and at most three jobs hold host memory: one
//! downloading, one building, one uploading, or one in each stage's slot
//! while the stage ahead finishes.
//!
//! Those jobs share a budget of host memory too: a job takes its
//! buffers' share (vectors, doc ids, graph; not the transfers' own
//! buffers) before it downloads and gives it back when it ends, so
//! a large job waits until the ones ahead of it have made room. A job
//! larger than the whole budget takes all of it, and runs alone.
//!
//! [`Pipeline::in_flight`] tells where each job is and since when, for a
//! forecast of when each will end.

use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use knn_store::{Cancel, Store};
use tokio::runtime::Handle;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc, oneshot};
use tokio::task::JoinHandle;

use crate::cost::Fitted;
use crate::load::{Dataset, load};
use crate::write::write_index;
use crate::{Builder, Job, JobError};

/// The stages, running.
#[derive(Debug)]
pub struct Pipeline {
    queue: mpsc::UnboundedSender<Queued>,
    board: Arc<Board>,
    /// The number the next job submitted gets.
    next: AtomicU64,
    downloads: JoinHandle<()>,
    builds: thread::JoinHandle<()>,
    uploads: JoinHandle<()>,
}

impl Pipeline {
    /// Starts the stages: the download and upload stages on the current
    /// Tokio runtime, which must be multi-threaded (the transfers block
    /// threads of their own and run their requests on it), and `builder`
    /// on a thread of its own. The jobs' buffers, their vectors, doc ids
    /// and graphs, take `host_memory` bytes at most at once, counted in
    /// whole MiB. The transfers' own buffers come on top: the one upload's
    /// parts in memory, up to [`knn_store::Transfers`]'s
    /// `upload_concurrency` plus two, of `part_size` or more (80 MiB at
    /// least by default).
    ///
    /// # Panics
    ///
    /// When called outside a Tokio runtime.
    pub fn start<B: Builder>(store: Store, builder: B, host_memory: u64) -> Self {
        let runtime = Handle::current();
        let (queue, queued) = mpsc::unbounded_channel();
        let (to_build, loaded) = mpsc::channel(1);
        let (to_upload, graphed) = mpsc::channel(1);
        let budget = Budget::new(host_memory);
        let downloads = runtime.spawn(downloads(store.clone(), queued, to_build, budget));
        let builds = {
            let runtime = runtime.clone();
            thread::spawn(move || builds(builder, loaded, &to_upload, &runtime))
        };
        let uploads = runtime.spawn(uploads(store, graphed));
        Self {
            queue,
            board: Arc::default(),
            next: AtomicU64::new(0),
            downloads,
            builds,
            uploads,
        }
    }

    /// Queues `job`, after the jobs already queued.
    ///
    /// # Panics
    ///
    /// After a thread panicked while it held the list of jobs in flight,
    /// a bug.
    #[must_use]
    pub fn submit(&self, job: Job) -> JobHandle {
        let cancel = Cancel::new();
        let claim = Arc::new(Claim::default());
        let (report, outcome) = oneshot::channel();
        // On the board and in the queue under one lock, so both hold the
        // jobs in one order.
        let mut board = self.board.0.lock().expect(BOARD_POISONED);
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        let track = Board::track(&mut board, id, &job, &cancel);
        // Stages that stopped drop the job, whose outcome is then
        // `Stopped`.
        _ = self.queue.send(Queued {
            job,
            cancel: cancel.clone(),
            claim: Arc::clone(&claim),
            report,
            track,
        });
        drop(board);
        JobHandle {
            id,
            cancel,
            claim,
            outcome,
        }
    }

    /// Every job that has not ended, in the order they were submitted,
    /// which is the order each stage takes them in.
    ///
    /// # Panics
    ///
    /// After a thread panicked while it held the list, a bug.
    #[must_use]
    pub fn in_flight(&self) -> Vec<InFlight> {
        self.board.in_flight()
    }

    /// What tells the jobs in flight from elsewhere, such as from a task
    /// that does not hold the pipeline.
    #[must_use]
    pub fn flights(&self) -> Flights {
        Flights(Arc::clone(&self.board))
    }

    /// Takes no more jobs, and waits until every job taken has ended.
    pub async fn close(self) {
        let Self {
            queue,
            board: _,
            next: _,
            downloads,
            builds,
            uploads,
        } = self;
        drop(queue);
        // Each stage ends when the one before it does. A stage that
        // panicked outside a job ends too; its jobs end `Stopped`.
        _ = downloads.await;
        _ = tokio::task::spawn_blocking(move || builds.join()).await;
        _ = uploads.await;
    }
}

/// A job in the pipeline.
#[derive(Debug)]
pub struct JobHandle {
    id: u64,
    cancel: Cancel,
    claim: Arc<Claim>,
    outcome: oneshot::Receiver<Result<Built, JobError>>,
}

impl JobHandle {
    /// The job's number: the order it was submitted in, as
    /// [`InFlight::id`] reports it.
    #[must_use]
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Cancels the job, as [`Canceler::cancel`] does.
    pub fn cancel(&self) {
        self.canceler().cancel();
    }

    /// What cancels the job from elsewhere, such as while another task
    /// awaits its [`JobHandle::outcome`].
    #[must_use]
    pub fn canceler(&self) -> Canceler {
        Canceler {
            cancel: self.cancel.clone(),
            claim: Arc::clone(&self.claim),
        }
    }

    /// Waits for the job to end.
    ///
    /// # Errors
    ///
    /// The job's failure.
    pub async fn outcome(self) -> Result<Built, JobError> {
        let Self {
            id: _,
            cancel,
            claim,
            outcome,
        } = self;
        // A job withdrawn from the queue has ended; the download stage
        // drops it when it gets there.
        let withdrawn = async {
            cancel.canceled().await;
            if !claim.is_withdrawn() {
                std::future::pending::<()>().await;
            }
        };
        tokio::select! {
            report = outcome => report.unwrap_or(Err(JobError::Stopped)),
            () = withdrawn => Err(JobError::Canceled),
        }
    }
}

/// Cancels one job; clones cancel the same job.
#[derive(Debug, Clone)]
pub struct Canceler {
    cancel: Cancel,
    claim: Arc<Claim>,
}

impl Canceler {
    /// Cancels the job: one still queued ends at once and never starts, a
    /// download or upload stops at once, an upload aborts so nothing is
    /// stored, and a build, which cannot be stopped, is thrown away when it
    /// ends. The job then ends with [`JobError::Canceled`], unless its
    /// upload had already sent its last request.
    pub fn cancel(&self) {
        // Withdrawn before the flag is set, so a wait on the flag finds the
        // claim settled.
        self.claim.withdraw();
        self.cancel.cancel();
    }

    /// A canceler of no job, for tests of what holds one.
    #[cfg(test)]
    pub(crate) fn detached() -> Self {
        Self {
            cancel: Cancel::new(),
            claim: Arc::default(),
        }
    }
}

/// Whether the download stage has started a job, or a cancel withdrew it
/// from the queue first. Exactly one of the two happens.
#[derive(Debug, Default)]
struct Claim(AtomicU8);

impl Claim {
    const QUEUED: u8 = 0;
    const STARTED: u8 = 1;
    const WITHDRAWN: u8 = 2;

    /// Marks the job started. Returns false when a cancel withdrew it
    /// first.
    fn start(&self) -> bool {
        self.settle(Self::STARTED)
    }

    /// Withdraws the job, unless it has started.
    fn withdraw(&self) {
        self.settle(Self::WITHDRAWN);
    }

    fn settle(&self, state: u8) -> bool {
        self.0
            .compare_exchange(Self::QUEUED, state, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
    }

    fn is_withdrawn(&self) -> bool {
        self.0.load(Ordering::SeqCst) == Self::WITHDRAWN
    }
}

/// A job that ended with its index stored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Built {
    /// The index file's name: the status response's `file_name`.
    pub file_name: String,
    /// When each stage worked on it.
    pub timings: Timings,
}

/// When each stage worked on a job.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Timings {
    /// Reading the blobs' sizes and downloading them.
    pub download: Span,
    /// Building the graph.
    pub build: Span,
    /// Writing and uploading the index.
    pub upload: Span,
}

/// A stage's time with a job.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    /// Started.
    pub start: Instant,
    /// Ended.
    pub end: Instant,
}

impl Span {
    /// From `start` until now.
    fn since(start: Instant) -> Self {
        Self {
            start,
            end: Instant::now(),
        }
    }

    /// How long.
    #[must_use]
    pub fn duration(&self) -> Duration {
        self.end - self.start
    }
}

/// Where a job is in the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// Waiting to download: in the queue, for the build slot, or for room
    /// in the budget.
    Queued,
    /// Downloading.
    Downloading,
    /// Downloaded, in the build slot until the builder takes it.
    Loaded,
    /// Building.
    Building,
    /// Built, in the upload slot until the upload stage takes it.
    Graphed,
    /// Writing and uploading its index.
    Uploading,
}

/// A job that has not ended, as [`Pipeline::in_flight`] reports it.
#[derive(Debug, Clone)]
pub struct InFlight {
    /// Its number, as [`JobHandle::id`] gives it.
    pub id: u64,
    /// The job.
    pub job: Job,
    /// Its stage.
    pub stage: Stage,
    /// When it entered that stage.
    pub since: Instant,
    /// Canceled: a queued job will not start, a download or upload stops,
    /// a build runs to its end and the job ends there.
    pub canceled: bool,
}

/// The jobs in flight of a pipeline, as [`Pipeline::flights`] gives
/// them; clones tell the same.
#[derive(Debug, Clone)]
pub struct Flights(Arc<Board>);

impl Flights {
    /// As [`Pipeline::in_flight`].
    ///
    /// # Panics
    ///
    /// After a thread panicked while it held the list, a bug.
    #[must_use]
    pub fn in_flight(&self) -> Vec<InFlight> {
        self.0.in_flight()
    }
}

/// The jobs submitted and not ended, in order.
#[derive(Debug, Default)]
struct Board(Mutex<Vec<Arc<Progress>>>);

/// One job's place, from its submission until it ends.
#[derive(Debug)]
struct Progress {
    id: u64,
    job: Job,
    cancel: Cancel,
    /// Its stage and since when; `None` once it has ended.
    stage: Mutex<Option<(Stage, Instant)>>,
}

const BOARD_POISONED: &str = "a thread panicked holding the pipeline's job list";

impl Board {
    /// Puts `job` on the `board`, locked, after dropping the jobs that
    /// have ended: it holds no more than the jobs in flight and those that
    /// ended since the last job came.
    fn track(board: &mut Vec<Arc<Progress>>, id: u64, job: &Job, cancel: &Cancel) -> Track {
        Self::prune(board);
        let progress = Arc::new(Progress {
            id,
            job: job.clone(),
            cancel: cancel.clone(),
            stage: Mutex::new(Some((Stage::Queued, Instant::now()))),
        });
        board.push(Arc::clone(&progress));
        Track(progress)
    }

    fn prune(board: &mut Vec<Arc<Progress>>) {
        board.retain(|progress| progress.stage.lock().expect(BOARD_POISONED).is_some());
    }

    fn in_flight(&self) -> Vec<InFlight> {
        let mut jobs = self.0.lock().expect(BOARD_POISONED);
        Self::prune(&mut jobs);
        jobs.iter()
            .filter_map(|progress| {
                let (stage, since) = (*progress.stage.lock().expect(BOARD_POISONED))?;
                Some(InFlight {
                    id: progress.id,
                    job: progress.job.clone(),
                    stage,
                    since,
                    canceled: progress.cancel.is_canceled(),
                })
            })
            .collect()
    }
}

/// A job's entry on the board, carried with it through the stages: it
/// ends however the job does, its stage dropping it.
#[derive(Debug)]
struct Track(Arc<Progress>);

impl Track {
    fn enter(&self, stage: Stage) {
        *self.0.stage.lock().expect(BOARD_POISONED) = Some((stage, Instant::now()));
    }
}

impl Drop for Track {
    fn drop(&mut self) {
        // A poisoned lock is left as it is: the board's next look panics.
        if let Ok(mut stage) = self.0.stage.lock() {
            *stage = None;
        }
    }
}

/// A job waiting to download.
struct Queued {
    job: Job,
    cancel: Cancel,
    claim: Arc<Claim>,
    report: oneshot::Sender<Result<Built, JobError>>,
    track: Track,
}

impl Queued {
    fn fail(self, error: JobError) {
        // Off the board before its end is told, so whoever hears of it
        // sees it gone.
        let Self { report, track, .. } = self;
        drop(track);
        // The handle may be gone; the job ends all the same.
        _ = report.send(Err(error));
    }
}

/// The host memory the jobs' buffers may take at once, in MiB.
pub(crate) struct Budget {
    mib: Arc<Semaphore>,
    total: u32,
}

impl Budget {
    const MIB: u64 = 1 << 20;

    fn new(bytes: u64) -> Self {
        let total = Self::total(bytes);
        Self {
            mib: Arc::new(Semaphore::new(total as usize)),
            total,
        }
    }

    /// The MiB a budget of `bytes` holds: whole MiB, at least one, and a
    /// count one permit can take.
    pub(crate) fn total(bytes: u64) -> u32 {
        let max = u32::try_from(Semaphore::MAX_PERMITS).unwrap_or(u32::MAX);
        u32::try_from(bytes / Self::MIB)
            .unwrap_or(u32::MAX)
            .clamp(1, max)
    }

    /// `job`'s share of a budget of `total` MiB: its buffers rounded up to
    /// whole MiB, or the whole budget when they are larger.
    pub(crate) fn share(job: &Job, total: u32) -> u32 {
        let mib = job.host_bytes().div_ceil(Self::MIB);
        u32::try_from(mib).map_or(total, |mib| mib.clamp(1, total))
    }

    /// Waits for `job`'s share.
    async fn take(&self, job: &Job) -> OwnedSemaphorePermit {
        Arc::clone(&self.mib)
            .acquire_many_owned(Self::share(job, self.total))
            .await
            .expect("the budget is never closed")
    }
}

/// A job downloaded, waiting to build.
struct Loaded {
    queued: Queued,
    dataset: Dataset,
    download: Span,
    /// Its share of the budget, given back when it ends.
    memory: OwnedSemaphorePermit,
}

/// A job built, waiting to upload.
struct Graphed {
    loaded: Loaded,
    graph: Vec<u32>,
    degree: usize,
    build: Span,
}

/// The download stage.
async fn downloads(
    store: Store,
    mut queued: mpsc::UnboundedReceiver<Queued>,
    to_build: mpsc::Sender<Loaded>,
    budget: Budget,
) {
    while let Some(job) = queued.recv().await {
        // Its handle has ended it; no need to wait for a slot.
        if job.claim.is_withdrawn() {
            job.fail(JobError::Canceled);
            continue;
        }
        // The build slot: a download starts once the builder has taken
        // the last one.
        let Ok(slot) = to_build.reserve().await else {
            job.fail(JobError::Stopped);
            continue;
        };
        // Room for its buffers; a cancel meanwhile ends it, as it would in
        // the queue.
        let memory = tokio::select! {
            memory = budget.take(&job.job) => memory,
            () = job.cancel.canceled() => {
                job.fail(JobError::Canceled);
                continue;
            }
        };
        // From here on a cancel stops the transfers instead.
        if !job.claim.start() {
            job.fail(JobError::Canceled);
            continue;
        }
        job.track.enter(Stage::Downloading);
        let start = Instant::now();
        match load(&store, &job.job, &job.cancel).await {
            Ok(dataset) => {
                let download = Span::since(start);
                job.track.enter(Stage::Loaded);
                slot.send(Loaded {
                    queued: job,
                    dataset,
                    download,
                    memory,
                });
            }
            Err(error) => job.fail(error),
        }
    }
}

/// The build stage, on the builder's thread.
fn builds<B: Builder>(
    mut builder: B,
    mut loaded: mpsc::Receiver<Loaded>,
    to_upload: &mpsc::Sender<Graphed>,
    runtime: &Handle,
) {
    loop {
        // The upload slot, taken before the next job leaves the build
        // slot: until its graph has somewhere to go, that job waits there,
        // and the download stage behind it.
        let Ok(slot) = runtime.block_on(to_upload.reserve()) else {
            // The upload stage is gone; the jobs left end `Stopped`. Closed,
            // the build slot stops the download stage at once instead of
            // feeding it every queued job.
            loaded.close();
            while let Some(job) = loaded.blocking_recv() {
                job.queued.fail(JobError::Stopped);
            }
            return;
        };
        let Some(mut job) = loaded.blocking_recv() else {
            return;
        };
        if job.queued.cancel.is_canceled() {
            job.queued.fail(JobError::Canceled);
            continue;
        }
        job.queued.track.enter(Stage::Building);
        let start = Instant::now();
        // Defense in depth at the job boundary: a panicking build fails
        // its job, not the stage.
        let built = panic::catch_unwind(AssertUnwindSafe(|| build(&mut builder, &job)))
            .unwrap_or(Err(JobError::Panicked { stage: "build" }));
        match built {
            Ok((graph, degree)) => {
                // Not in the file, so not needed past the build.
                if job.queued.job.skip_stored_vectors {
                    job.dataset.vectors.release_values();
                }
                let span = Span::since(start);
                job.queued.track.enter(Stage::Graphed);
                slot.send(Graphed {
                    build: span,
                    loaded: job,
                    graph,
                    degree,
                });
            }
            Err(error) => job.queued.fail(error),
        }
    }
}

/// A job's graph and its degree.
fn build<B: Builder>(builder: &mut B, job: &Loaded) -> Result<(Vec<u32>, usize), JobError> {
    let request = &job.queued.job;
    // Another process may have taken device memory since the job was
    // admitted: fail it now rather than out of memory partway.
    if let Some(free) = builder.free_device_memory() {
        let needed = Fitted::device_bytes_needed(request);
        if needed > free {
            return Err(JobError::DeviceMemory {
                needed_mib: needed.div_ceil(1 << 20),
                free_mib: free >> 20,
            });
        }
    }
    let params = request.graph_params().map_err(JobError::Params)?;
    builder
        .build(&job.dataset.vectors, request.dimension as usize, &params)
        .map_err(|e| JobError::Build(e.to_string()))
}

/// The upload stage.
async fn uploads(store: Store, mut graphed: mpsc::Receiver<Graphed>) {
    while let Some(job) = graphed.recv().await {
        let Graphed {
            loaded:
                Loaded {
                    queued,
                    dataset,
                    download,
                    memory,
                },
            graph,
            degree,
            build,
        } = job;
        queued.track.enter(Stage::Uploading);
        let start = Instant::now();
        let (store, request, cancel) = (store.clone(), queued.job.clone(), queued.cancel.clone());
        // The inputs and the graph are freed as the write ends.
        let written = tokio::task::spawn_blocking(move || {
            write_index(&store, &request, &dataset, &graph, degree, &cancel)
        })
        .await;
        let outcome = match written {
            Ok(Ok(())) => Ok(Built {
                file_name: queued.job.file_name(),
                timings: Timings {
                    download,
                    build,
                    upload: Span::since(start),
                },
            }),
            Ok(Err(error)) => Err(error),
            Err(join) if join.is_panic() => Err(JobError::Panicked { stage: "upload" }),
            // The runtime is shutting down.
            Err(_) => Err(JobError::Stopped),
        };
        // The buffers are freed; the share goes back before the job ends,
        // so the next one may start downloading at once.
        drop(memory);
        // Off the board before its end is told, as in `Queued::fail`.
        let Queued { report, track, .. } = queued;
        drop(track);
        _ = report.send(outcome);
    }
}

#[cfg(test)]
mod tests {
    use super::Budget;
    use crate::job::tests::job;

    #[test]
    fn counts_the_budget_in_whole_mib() {
        assert_eq!(Budget::new(0).total, 1, "never none");
        assert_eq!(Budget::new((3 << 20) + 5).total, 3);
        assert_eq!(Budget::new(32_000_000_000).total, 30_517);
        assert!(Budget::new(u64::MAX).total > 30_517);
    }

    #[tokio::test]
    async fn gives_a_job_its_share_or_all_of_it() {
        let budget = Budget::new(4 << 20);
        // About 0.18 MB, rounded up.
        let small = job("bucket", 600, 16);
        assert_eq!(budget.take(&small).await.num_permits(), 1);
        // Between 1 and 2 MiB: rounded up, to 2.
        let mid = job("bucket", 4_000, 32);
        assert!((1 << 20..2 << 20).contains(&mid.host_bytes()));
        assert_eq!(budget.take(&mid).await.num_permits(), 2);
        // About 31 MB: larger than the budget, so all of it.
        let large = job("bucket", 80_000, 64);
        assert_eq!(budget.take(&large).await.num_permits(), 4);
        // Given back when dropped.
        assert_eq!(budget.mib.available_permits(), 4);
    }
}
