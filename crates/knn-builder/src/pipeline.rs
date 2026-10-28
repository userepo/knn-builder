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

use std::panic::{self, AssertUnwindSafe};
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use knn_gpu::BuildParams;
use knn_store::{Cancel, Store};
use tokio::runtime::Handle;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc, oneshot};
use tokio::task::JoinHandle;

use crate::load::{Dataset, load};
use crate::write::write_index;
use crate::{Builder, Job, JobError};

/// The stages, running.
#[derive(Debug)]
pub struct Pipeline {
    queue: mpsc::UnboundedSender<Queued>,
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
            downloads,
            builds,
            uploads,
        }
    }

    /// Queues `job`, after the jobs already queued.
    #[must_use]
    pub fn submit(&self, job: Job) -> JobHandle {
        let cancel = Cancel::new();
        let claim = Arc::new(Claim::default());
        let (report, outcome) = oneshot::channel();
        // Stages that stopped drop the job, whose outcome is then
        // `Stopped`.
        _ = self.queue.send(Queued {
            job,
            cancel: cancel.clone(),
            claim: Arc::clone(&claim),
            report,
        });
        JobHandle {
            cancel,
            claim,
            outcome,
        }
    }

    /// Takes no more jobs, and waits until every job taken has ended.
    pub async fn close(self) {
        let Self {
            queue,
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
    cancel: Cancel,
    claim: Arc<Claim>,
    outcome: oneshot::Receiver<Result<Built, JobError>>,
}

impl JobHandle {
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

/// A job waiting to download.
struct Queued {
    job: Job,
    cancel: Cancel,
    claim: Arc<Claim>,
    report: oneshot::Sender<Result<Built, JobError>>,
}

impl Queued {
    fn fail(self, error: JobError) {
        // The handle may be gone; the job ends all the same.
        _ = self.report.send(Err(error));
    }
}

/// The host memory the jobs' buffers may take at once, in MiB.
struct Budget {
    mib: Arc<Semaphore>,
    total: u32,
}

impl Budget {
    const MIB: u64 = 1 << 20;

    fn new(bytes: u64) -> Self {
        // Whole MiB, at least one, and a count one permit can take.
        let max = u32::try_from(Semaphore::MAX_PERMITS).unwrap_or(u32::MAX);
        let total = u32::try_from(bytes / Self::MIB)
            .unwrap_or(u32::MAX)
            .clamp(1, max);
        Self {
            mib: Arc::new(Semaphore::new(total as usize)),
            total,
        }
    }

    /// Waits for `job`'s share: its buffers rounded up to whole MiB, or
    /// the whole budget when they are larger.
    async fn take(&self, job: &Job) -> OwnedSemaphorePermit {
        let mib = job.host_bytes().div_ceil(Self::MIB);
        let share = u32::try_from(mib).map_or(self.total, |mib| mib.clamp(1, self.total));
        Arc::clone(&self.mib)
            .acquire_many_owned(share)
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
        let start = Instant::now();
        match load(&store, &job.job, &job.cancel).await {
            Ok(dataset) => slot.send(Loaded {
                queued: job,
                dataset,
                download: Span::since(start),
                memory,
            }),
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
                slot.send(Graphed {
                    build: Span::since(start),
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
    let params = BuildParams::new(
        request.doc_count,
        request.dimension,
        request.m,
        request.space_type.build_metric(),
    )
    .map_err(JobError::Params)?;
    let graph = builder
        .build(&job.dataset.vectors, request.dimension as usize, &params)
        .map_err(|e| JobError::Build(e.to_string()))?;
    Ok((graph, params.graph_degree as usize))
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
        _ = queued.report.send(outcome);
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
