use std::env;
use std::error::Error as StdError;
use std::num::NonZero;
use std::thread;
use std::time::Duration;

use aws_config::meta::region::ProvideRegion;
use aws_config::profile::ProfileFileRegionProvider;
use aws_config::retry::RetryConfig;
use aws_config::timeout::TimeoutConfig;
use aws_config::{BehaviorVersion, ConfigLoader, SdkConfig};
use aws_sdk_s3::Client;
use aws_sdk_s3::config::Region;
use aws_sdk_s3::config::http::HttpResponse;
use aws_sdk_s3::error::{ProvideErrorMetadata, SdkError};
use aws_smithy_http_client::tls::{self, rustls_provider::CryptoMode};
use tokio::runtime::Handle;

use crate::Error;

const MIB: usize = 1 << 20;

/// S3's smallest part of a multipart upload, but for the last.
const MIN_PART: usize = 5 * MIB;

/// The region boto3 signs S3 requests with when none is set.
const FALLBACK_REGION: &str = "us-east-1";

/// Where S3 is, from the environment variables the upstream service
/// reads. Credentials come from the AWS SDK's default chain, as boto3's
/// do: environment, profile, web identity, container or instance role.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Settings {
    /// `S3_ENDPOINT_URL`: an endpoint other than AWS's, such as
    /// LocalStack's. Requests to it use path-style URLs, as boto3's do,
    /// so the bucket name need not resolve as a host name.
    pub endpoint_url: Option<String>,
    /// `AWS_DEFAULT_REGION`. When unset with an endpoint, the profile's
    /// region, else `us-east-1`, as boto3 picks it. When unset without
    /// one, the SDK's chain: `AWS_REGION`, the profile, then the instance
    /// metadata, which gives an EC2 instance its own region (the Rust SDK
    /// does not follow S3's redirects to the bucket's region, as boto3
    /// does) and takes a few seconds to time out anywhere else; failing
    /// all of them, `us-east-1`.
    pub region: Option<String>,
}

impl Settings {
    /// Reads `S3_ENDPOINT_URL` and `AWS_DEFAULT_REGION`; an empty value
    /// counts as unset.
    #[must_use]
    pub fn from_env() -> Self {
        let var = |name| env::var(name).ok().filter(|value| !value.is_empty());
        Self {
            endpoint_url: var("S3_ENDPOINT_URL"),
            region: var("AWS_DEFAULT_REGION"),
        }
    }
}

/// How blobs are split up and how many requests run at once.
///
/// The defaults keep the upstream service's concurrency (its #142):
/// eight requests at least, more with more CPUs. Ranges and parts are
/// 8 MiB, the AWS CRT client's default, where the service uses 50 MiB;
/// the size is what an upload holds in memory per request in flight.
/// A store raises a value below its floor to it: 1 for each, 5 MiB for
/// the part size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Transfers {
    /// Bytes per ranged GET. A blob no larger comes in one GET, which
    /// checks the object's checksum when it has one.
    pub range_size: usize,
    /// GETs in flight per blob, each on a thread of its own that decodes
    /// its range.
    pub download_concurrency: usize,
    /// The smallest part of a multipart upload: 5 MiB at least, S3's
    /// minimum. A file no larger goes up in one PUT; one over 10,000
    /// parts of this size, S3's limit, gets larger parts.
    pub part_size: usize,
    /// Parts in flight per upload. The upload holds up to two more
    /// parts in memory: the one being written and one waiting.
    pub upload_concurrency: usize,
}

impl Default for Transfers {
    fn default() -> Self {
        Self {
            range_size: 8 * MIB,
            download_concurrency: per_cpu(5, 8),
            part_size: 8 * MIB,
            upload_concurrency: per_cpu(1, 2),
        }
    }
}

impl Transfers {
    /// Each value raised to its floor.
    fn floored(self) -> Self {
        Self {
            range_size: self.range_size.max(1),
            download_concurrency: self.download_concurrency.max(1),
            part_size: self.part_size.max(MIN_PART),
            upload_concurrency: self.upload_concurrency.max(1),
        }
    }
}

/// `max(8, floor(cpus x numerator / denominator))`, the service's
/// `get_transfer_concurrency` with factor 0.625 for downloads and 0.5 for
/// uploads. It counts the CPUs this process may use, where the service
/// counts the host's.
fn per_cpu(numerator: usize, denominator: usize) -> usize {
    let cpus = thread::available_parallelism().map_or(1, NonZero::get);
    (cpus.saturating_mul(numerator) / denominator).max(8)
}

/// A blob's size and what a transfer needs to know about it, from a
/// `HeadObject`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Blob {
    /// Bytes.
    pub size: u64,
    /// The `ETag`. Every GET of the blob asks for this one, so a blob
    /// replaced mid-download fails the download instead of mixing two.
    pub e_tag: Option<String>,
    /// The KMS key the blob is encrypted with, under SSE-KMS. The
    /// service encrypts the index file it uploads with the same key.
    pub kms_key: Option<String>,
}

/// An S3 client and the transfer settings, shared by every job.
///
/// Downloads and uploads block their caller and run their requests on
/// the Tokio runtime the store was created on, which must be
/// multi-threaded, or driven by another thread meanwhile.
///
/// Cloning is cheap: clones share the client and its connections.
#[derive(Debug, Clone)]
pub struct Store {
    pub(crate) client: Client,
    pub(crate) transfers: Transfers,
    /// The runtime the requests run on, so downloads and uploads can be
    /// driven from blocking threads.
    pub(crate) runtime: Handle,
}

impl Store {
    /// Creates the client from `settings` and the SDK's default
    /// credential chain, with four attempts per request, as boto3's
    /// `max_attempts` of 3 retries allows, and boto3's 60-second read
    /// timeout. Values in `transfers` below their floor are raised to it.
    ///
    /// # Panics
    ///
    /// When called outside a Tokio runtime.
    pub async fn new(settings: &Settings, transfers: Transfers) -> Self {
        let region = match (&settings.region, &settings.endpoint_url) {
            (Some(region), _) => Some(Region::new(region.clone())),
            // An endpoint of its own routes by URL, so the region only
            // signs: boto3's order, without asking the instance metadata.
            (None, Some(_)) => Some(
                ProfileFileRegionProvider::new()
                    .region()
                    .await
                    .unwrap_or(Region::from_static(FALLBACK_REGION)),
            ),
            // The loader runs the SDK's region chain, with the HTTPS
            // client from `loader` for the instance metadata.
            (None, None) => None,
        };
        let mut loader = loader();
        if let Some(region) = region {
            loader = loader.region(region);
        }
        Self::with_config(&loader.load().await, settings, transfers)
    }

    /// A store on an SDK configuration from [`loader`].
    pub(crate) fn with_config(
        config: &SdkConfig,
        settings: &Settings,
        transfers: Transfers,
    ) -> Self {
        let mut s3 = aws_sdk_s3::config::Builder::from(config);
        if config.region().is_none() {
            s3 = s3.region(Region::from_static(FALLBACK_REGION));
        }
        if let Some(url) = &settings.endpoint_url {
            s3 = s3.endpoint_url(url).force_path_style(true);
        }
        Self {
            client: Client::from_conf(s3.build()),
            transfers: transfers.floored(),
            runtime: Handle::current(),
        }
    }

    /// The transfer settings, floors applied.
    #[must_use]
    pub fn transfers(&self) -> Transfers {
        self.transfers
    }

    /// The S3 client, for tests that set up and inspect buckets.
    #[doc(hidden)]
    #[must_use]
    pub fn sdk_client(&self) -> &Client {
        &self.client
    }

    /// Reads a blob's size, `ETag` and KMS key.
    ///
    /// # Errors
    ///
    /// Returns [`Error::S3`] when the `HeadObject` fails, the blob missing
    /// among other causes, or its response has no size.
    pub async fn head(&self, bucket: &str, key: &str) -> Result<Blob, Error> {
        let call = "HeadObject";
        let head = self
            .client
            .head_object()
            .bucket(bucket)
            .key(key)
            .send()
            .await
            .map_err(|e| s3_error(call, bucket, key, &e))?;
        let size = head
            .content_length()
            .and_then(|size| u64::try_from(size).ok())
            .ok_or_else(|| Error::S3 {
                call,
                bucket: bucket.to_owned(),
                key: key.to_owned(),
                message: "the response has no valid Content-Length".to_owned(),
            })?;
        Ok(Blob {
            size,
            e_tag: head.e_tag().map(str::to_owned),
            kms_key: head.ssekms_key_id().map(str::to_owned),
        })
    }
}

/// The SDK settings every store starts from: rustls on ring, four
/// attempts per request, the SDK's 3.1-second connect timeout and boto3's
/// 60-second read timeout, the longest a request waits from its start to
/// its response's first byte. Without it, a response that never comes
/// would hold a job forever.
pub(crate) fn loader() -> ConfigLoader {
    let https = aws_smithy_http_client::Builder::new()
        .tls_provider(tls::Provider::Rustls(CryptoMode::Ring))
        .build_https();
    aws_config::defaults(BehaviorVersion::v2026_01_12())
        .http_client(https)
        .retry_config(RetryConfig::standard().with_max_attempts(4))
        .timeout_config(
            TimeoutConfig::builder()
                .connect_timeout(Duration::from_millis(3_100))
                .read_timeout(Duration::from_secs(60))
                .build(),
        )
}

/// [`Error::S3`] for a failed request: S3's error code, the HTTP status
/// and S3's message for an error response, the chain of causes for any
/// other failure, such as a connection refused.
pub(crate) fn s3_error<E>(
    call: &'static str,
    bucket: &str,
    key: &str,
    error: &SdkError<E, HttpResponse>,
) -> Error
where
    E: ProvideErrorMetadata + StdError + 'static,
{
    let message = match error {
        SdkError::ServiceError(service) => {
            let status = service.raw().status().as_u16();
            let reply = service.err();
            match (reply.code(), reply.message()) {
                (Some(code), Some(text)) => format!("{code} (HTTP {status}): {text}"),
                (Some(code), None) => format!("{code} (HTTP {status})"),
                _ => format!("HTTP {status}: {}", causes(reply)),
            }
        }
        _ => causes(error),
    };
    Error::S3 {
        call,
        bucket: bucket.to_owned(),
        key: key.to_owned(),
        message,
    }
}

/// [`Error::S3`] for a response body cut off.
pub(crate) fn body_error(bucket: &str, key: &str, error: &(dyn StdError + 'static)) -> Error {
    Error::S3 {
        call: "GetObject",
        bucket: bucket.to_owned(),
        key: key.to_owned(),
        message: causes(error),
    }
}

/// An error and its sources, colon-separated. The SDK's own
/// `DisplayErrorContext` adds the error's `Debug` form, raw response
/// included, too long for a job's error message.
fn causes(error: &(dyn StdError + 'static)) -> String {
    let mut text = error.to_string();
    let mut source = error.source();
    while let Some(cause) = source {
        text.push_str(": ");
        text.push_str(&cause.to_string());
        source = cause.source();
    }
    text
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use aws_config::retry::RetryConfig;
    use aws_config::timeout::TimeoutConfig;

    use super::{MIB, Settings, Store, Transfers};
    use crate::Error;
    use crate::testing::{FakeS3Server, test_loader, test_store};

    #[test]
    fn raises_transfers_to_their_floors() {
        let zero = Transfers {
            range_size: 0,
            download_concurrency: 0,
            part_size: MIB,
            upload_concurrency: 0,
        };
        assert_eq!(
            zero.floored(),
            Transfers {
                range_size: 1,
                download_concurrency: 1,
                part_size: 5 * MIB,
                upload_concurrency: 1,
            }
        );
        let defaults = Transfers::default();
        assert_eq!(defaults.floored(), defaults);
    }

    #[tokio::test]
    async fn keeps_the_service_settings() {
        let transfers = Transfers {
            part_size: MIB,
            ..Transfers::default()
        };
        let store = test_store(None, transfers).await;
        assert_eq!(store.transfers().part_size, 5 * MIB);
        let s3 = store.client.config();
        let timeouts = s3.timeout_config().unwrap();
        assert_eq!(timeouts.read_timeout(), Some(Duration::from_secs(60)));
        assert_eq!(
            timeouts.connect_timeout(),
            Some(Duration::from_millis(3_100))
        );
        assert_eq!(s3.retry_config().unwrap().max_attempts(), 4);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn gives_up_on_a_response_that_never_comes() {
        let silent = FakeS3Server::silent();
        // The service's timeout, shortened, and one attempt.
        let config = test_loader()
            .timeout_config(
                TimeoutConfig::builder()
                    .read_timeout(Duration::from_secs(1))
                    .build(),
            )
            .retry_config(RetryConfig::standard().with_max_attempts(1))
            .load()
            .await;
        let settings = Settings {
            endpoint_url: Some(silent.endpoint.clone()),
            region: None,
        };
        let store = Store::with_config(&config, &settings, Transfers::default());
        let start = Instant::now();
        let error = store.head("bucket", "vectors.knnvec").await.unwrap_err();
        assert!(
            matches!(
                error,
                Error::S3 {
                    call: "HeadObject",
                    ..
                }
            ),
            "{error}"
        );
        assert!(start.elapsed() < Duration::from_secs(10));
    }
}
