//! Jobs flow through three stages, one job at a time in each: the
//! download, the build on the builder's own thread, then the write and
//! upload. So the next job downloads while one builds, and the last one
//! uploads meanwhile. A stage takes the next stage's one slot before it
//! takes up a job, so a finished job waits where it is until the next
//! stage has room, and at most three jobs hold host memory: one
//! downloading, one building, one uploading, or one in each stage's slot
//! while the stage ahead finishes.

use std::panic::{self, AssertUnwindSafe};
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use knn_gpu::BuildParams;
use knn_store::{Cancel, Store};
use tokio::runtime::Handle;
use tokio::sync::{mpsc, oneshot};
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
    /// on a thread of its own.
    ///
    /// # Panics
    ///
    /// When called outside a Tokio runtime.
    pub fn start<B: Builder>(store: Store, builder: B) -> Self {
        let runtime = Handle::current();
        let (queue, queued) = mpsc::unbounded_channel();
        let (to_build, loaded) = mpsc::channel(1);
        let (to_upload, graphed) = mpsc::channel(1);
        let downloads = runtime.spawn(downloads(store.clone(), queued, to_build));
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

/// A job downloaded, waiting to build.
struct Loaded {
    queued: Queued,
    dataset: Dataset,
    download: Span,
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
        _ = queued.report.send(outcome);
    }
}
