//! The service: the HTTP API OpenSearch k-NN calls to start a remote index
//! build and poll its status, on port 1025 as the upstream service
//! listens, set up from the environment as [`knn_builder::config`] reads
//! it, with S3 as [`knn_store::Settings::from_env`] reads it.
//!
//! It builds on the GPU, so it needs the `gpu` feature; built without it,
//! it says so and exits.
//!
//! On SIGTERM or Ctrl-C it stops taking requests, cancels its jobs and
//! waits a few seconds for them to end: a build under way cannot stop,
//! and a stopping container has little time. A build still running then
//! is left behind; the data node builds that segment itself.

use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use knn_builder::api;
use knn_builder::config::Config;
use knn_builder::service::{Device, Service};
use knn_store::{Settings, Store, Transfers};
use tokio::net::TcpListener;
use tracing_subscriber::filter::Targets;
use tracing_subscriber::layer::SubscriberExt as _;
use tracing_subscriber::util::SubscriberInitExt as _;

/// The crates whose logs `LOG_LEVEL` sets.
const OWN_CRATES: [&str; 4] = ["knn_builder", "knn_store", "knn_gpu", "knn_write"];

/// The port the upstream service listens on.
const PORT: u16 = 1025;

/// How long a stop waits for the jobs to end once canceled. Docker gives
/// a stopping container 10 s.
const SHUTDOWN_GRACE: Duration = Duration::from_secs(5);

#[tokio::main(flavor = "multi_thread")]
async fn main() -> ExitCode {
    let mut config = match Config::from_env() {
        Ok(config) => config,
        Err(error) => {
            eprintln!("knn-builder: {error}");
            return ExitCode::FAILURE;
        }
    };
    // `LOG_LEVEL` for this service's own logs, INFO for the libraries',
    // as the upstream service sets its own loggers and leaves the root at
    // INFO.
    let own = OWN_CRATES.map(|target| (target, config.log_level));
    let filter = Targets::new()
        .with_default(tracing::Level::INFO)
        .with_targets(own);
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(filter)
        .init();
    let builder = match builder() {
        Ok(builder) => builder,
        Err(error) => {
            tracing::error!(%error, "no builder");
            return ExitCode::FAILURE;
        }
    };
    // Read once the build's resources exist, so their memory is not
    // counted free.
    config.service.device = match device() {
        Ok(device) => {
            tracing::info!(
                free_mib = device.free_memory >> 20,
                shared_memory_per_block = device.shared_memory_per_block,
                "GPU"
            );
            Some(device)
        }
        Err(error) => {
            tracing::error!(%error, "cannot read the GPU");
            return ExitCode::FAILURE;
        }
    };
    let listener = match TcpListener::bind(("0.0.0.0", PORT)).await {
        Ok(listener) => listener,
        Err(error) => {
            tracing::error!(%error, port = PORT, "cannot listen");
            return ExitCode::FAILURE;
        }
    };
    let store = Store::new(&Settings::from_env(), Transfers::default()).await;
    let service = Arc::new(Service::start(store, builder, config.service));
    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        port = PORT,
        basic_auth = config.auth.is_some(),
        "listening"
    );
    let app = api::app(Arc::clone(&service), config.auth);
    if let Err(error) = axum::serve(listener, axum::ServiceExt::into_make_service(app))
        .with_graceful_shutdown(stop())
        .await
    {
        tracing::error!(%error, "the server failed");
    }
    // The app, and the server's clones of it, are gone.
    let Some(service) = Arc::into_inner(service) else {
        tracing::warn!("stopped without canceling the jobs: the service is still shared");
        return ExitCode::SUCCESS;
    };
    if tokio::time::timeout(SHUTDOWN_GRACE, service.close())
        .await
        .is_err()
    {
        tracing::warn!("stopped with a build still under way");
    }
    ExitCode::SUCCESS
}

/// The GPU.
#[cfg(feature = "gpu")]
fn builder() -> Result<knn_gpu::Gpu, String> {
    knn_gpu::Gpu::new().map_err(|error| error.to_string())
}

/// No GPU without the `gpu` feature.
#[cfg(not(feature = "gpu"))]
fn builder() -> Result<knn_builder::FakeBuilder, String> {
    Err("built without the gpu feature, so there is no GPU to build on".to_owned())
}

/// The GPU's free memory and the shared memory a block may take.
#[cfg(feature = "gpu")]
fn device() -> Result<Device, String> {
    let memory = knn_gpu::device_memory().map_err(|error| error.to_string())?;
    let shared_memory_per_block =
        knn_gpu::shared_memory_per_block().map_err(|error| error.to_string())?;
    Ok(Device {
        free_memory: u64::try_from(memory.free).unwrap_or(u64::MAX),
        shared_memory_per_block,
    })
}

/// No GPU without the `gpu` feature: never called, the builder having
/// failed first.
#[cfg(not(feature = "gpu"))]
fn device() -> Result<Device, String> {
    Err("built without the gpu feature".to_owned())
}

/// A SIGTERM, or Ctrl-C. A signal that cannot be listened for never comes.
async fn stop() {
    let ctrl_c = async {
        if tokio::signal::ctrl_c().await.is_err() {
            std::future::pending::<()>().await;
        }
    };
    #[cfg(unix)]
    let terminate = async {
        use tokio::signal::unix::{SignalKind, signal};
        match signal(SignalKind::terminate()) {
            Ok(mut terminate) => {
                terminate.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        () = ctrl_c => {}
        () = terminate => {}
    }
    tracing::info!("stopping");
}
