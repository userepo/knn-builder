use std::collections::HashMap;
use std::env;
use std::error::Error as StdError;
use std::num::NonZero;
use std::sync::{Arc, Mutex};
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

/// The header S3 names a bucket's region in when a request reaches the
/// bucket through another region.
const BUCKET_REGION: &str = "x-amz-bucket-region";

/// The redirected clients' lock is held for a lookup or an insert,
/// neither of which can panic; a poisoned lock is a bug.
const REDIRECTED_POISONED: &str = "redirected clients lock poisoned: a holder panicked";

/// Clients kept for redirected buckets, at most: a client is small, but a
/// service sent ever new buckets would otherwise keep one for each
/// forever. Past the cap the set starts over, and an old bucket's
/// redirect is followed again.
const REDIRECTED_BUCKETS: usize = 256;

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
    /// metadata, which gives an EC2 instance its own region and takes a
    /// few seconds to time out anywhere else; failing all of them,
    /// `us-east-1`. A bucket in another region is reached through S3's
    /// redirect to it, as boto3 reaches it.
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
/// A bucket in another region than the client's is reached as boto3
/// reaches it: S3 answers a `HeadObject` that comes through the wrong
/// region with a 301, or a 400 for a request signed for the wrong one,
/// naming the bucket's region in a header (or in a `HeadBucket`'s, which
/// always carries it), and the store keeps a client for that region per
/// bucket, for the head's retry and every later request to the bucket.
/// A download or upload redirected itself, for a bucket moved since its
/// head, fails; the next head of the bucket follows the new redirect.
///
/// Downloads and uploads block their caller and run their requests on
/// the Tokio runtime the store was created on, which must be
/// multi-threaded, or driven by another thread meanwhile.
///
/// Cloning is cheap: clones share the clients and their connections.
#[derive(Debug, Clone)]
pub struct Store {
    /// The client for every bucket S3 has not redirected.
    client: Client,
    /// A client per bucket S3 redirected, signing for its region.
    redirected: Arc<Mutex<HashMap<String, Client>>>,
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
            redirected: Arc::default(),
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

    /// The client for `bucket`: the one for the region S3 redirected it
    /// to, else the store's own.
    pub(crate) fn client_for(&self, bucket: &str) -> Client {
        let redirected = self.redirected.lock().expect(REDIRECTED_POISONED);
        redirected
            .get(bucket)
            .cloned()
            .unwrap_or_else(|| self.client.clone())
    }

    /// A client signing for `region`, the store's own in every other
    /// setting, kept for `bucket`.
    fn redirect(&self, bucket: &str, region: &str) -> Client {
        let config = self
            .client
            .config()
            .to_builder()
            .region(Region::new(region.to_owned()))
            .build();
        let client = Client::from_conf(config);
        let mut redirected = self.redirected.lock().expect(REDIRECTED_POISONED);
        if redirected.len() >= REDIRECTED_BUCKETS && !redirected.contains_key(bucket) {
            redirected.clear();
        }
        redirected.insert(bucket.to_owned(), client.clone());
        client
    }

    /// Reads a blob's size, `ETag` and KMS key, through the bucket's
    /// region once S3 names it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::S3`] when the `HeadObject` fails, the blob missing
    /// among other causes, when S3 redirects it to a region it names
    /// nowhere, to the one it was signed for, or to another region
    /// twice, or when its response has no size.
    pub async fn head(&self, bucket: &str, key: &str) -> Result<Blob, Error> {
        let call = "HeadObject";
        let mut client = self.client_for(bucket);
        // The region a redirect was followed to, once one was.
        let mut followed: Option<String> = None;
        let head = loop {
            let error = match client.head_object().bucket(bucket).key(key).send().await {
                Ok(head) => break head,
                Err(error) => error,
            };
            let Some(response) = redirect(&error) else {
                return Err(s3_error(call, bucket, key, &error));
            };
            // The redirect names the region, or, for a 301 without it, a
            // `HeadBucket` does, as boto3 reads them in that order. A 400
            // without it is some other fault: S3 names the region on a
            // 400 for the wrong one, and a round trip to find out would be
            // spent on every other 400.
            let named = match named_region(response) {
                Some(region) => Some(region),
                None if response.status().as_u16() == 301 => {
                    head_bucket_region(&client, bucket).await
                }
                None => None,
            };
            // A redirect to the region signed for is some other fault too.
            let region = match named {
                Some(region) if !signs_for(&client, &region) => region,
                _ => return Err(s3_error(call, bucket, key, &error)),
            };
            // One redirect, as boto3 allows a request: the region S3
            // names is the bucket's, so a second redirect is some other
            // fault.
            if let Some(region) = &followed {
                return Err(after_redirect(s3_error(call, bucket, key, &error), region));
            }
            client = self.redirect(bucket, &region);
            followed = Some(region);
        };
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
                // S3's redirect to a bucket's region: a `HeadObject`'s
                // response carries no body, so no code.
                _ if status == 301 => {
                    "PermanentRedirect (HTTP 301): the bucket is in another region".to_owned()
                }
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

/// The response to a request that reached the bucket through another
/// region: a 301, or a 400 for a request signed for the wrong one.
fn redirect<E>(error: &SdkError<E, HttpResponse>) -> Option<&HttpResponse> {
    let SdkError::ServiceError(service) = error else {
        return None;
    };
    let response = service.raw();
    matches!(response.status().as_u16(), 301 | 400).then_some(response)
}

/// The bucket's region, when the response names it in S3's header.
fn named_region(response: &HttpResponse) -> Option<String> {
    response.headers().get(BUCKET_REGION).map(str::to_owned)
}

/// The bucket's region from a `HeadBucket`, whose response names it
/// whether the request got through or was redirected; `None` when
/// neither response names it or the request failed otherwise.
async fn head_bucket_region(client: &Client, bucket: &str) -> Option<String> {
    match client.head_bucket().bucket(bucket).send().await {
        Ok(output) => output.bucket_region().map(str::to_owned),
        Err(SdkError::ServiceError(service)) => named_region(service.raw()),
        Err(_) => None,
    }
}

/// `error`, its message saying the redirect already followed.
fn after_redirect(mut error: Error, region: &str) -> Error {
    if let Error::S3 { message, .. } = &mut error {
        message.push_str(", after a redirect to ");
        message.push_str(region);
    }
    error
}

/// Whether `client` signs its requests for `region`.
fn signs_for(client: &Client, region: &str) -> bool {
    client.config().region().map(AsRef::as_ref) == Some(region)
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
    use std::io::Write;
    use std::time::{Duration, Instant};

    use aws_config::retry::RetryConfig;
    use aws_config::timeout::TimeoutConfig;
    use aws_sdk_s3::Client;
    use tokio::runtime::Runtime;

    use super::{MIB, REDIRECTED_BUCKETS, Settings, Store, Transfers};
    use crate::testing::{Elsewhere, FakeS3Server, Signed, test_loader, test_store};
    use crate::{Cancel, Error};

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

    /// Twelve bytes in a bucket in `region`, where the test store signs for
    /// us-east-1; a request signed for another region is answered with
    /// `status`, naming `names` as the bucket's region in turn.
    fn elsewhere(region: &str, status: u16, names: &[&str]) -> FakeS3Server {
        FakeS3Server::serving_elsewhere(
            vec![7; 12],
            Elsewhere {
                region: region.to_owned(),
                status,
                names: names.iter().map(|&name| name.to_owned()).collect(),
            },
        )
    }

    fn signed(method: &str, path: &str, region: &str) -> Signed {
        Signed {
            method: method.to_owned(),
            path: path.to_owned(),
            region: Some(region.to_owned()),
        }
    }

    fn region(client: &Client) -> &str {
        client.config().region().map_or("", AsRef::as_ref)
    }

    const VECTORS: &str = "/bucket/vectors.knnvec";
    const IDS: &str = "/bucket/doc_ids.knndid";
    const INDEX: &str = "/bucket/index.faiss";

    #[test]
    fn follows_a_redirect_to_the_buckets_region() {
        for status in [301, 400] {
            let fake = elsewhere("eu-west-1", status, &["eu-west-1"]);
            let runtime = Runtime::new().unwrap();
            let store = runtime.block_on(test_store(Some(&fake), Transfers::default()));
            let blob = runtime
                .block_on(store.head("bucket", "vectors.knnvec"))
                .unwrap();
            assert_eq!(blob.size, 12, "status {status}");
            // Every later request to the bucket goes through its region,
            // from the store's clones too.
            let mut out = [0_u8; 12];
            store
                .clone()
                .download(
                    "bucket",
                    "vectors.knnvec",
                    &blob,
                    &mut out,
                    u8::from_le_bytes,
                    &Cancel::new(),
                )
                .unwrap();
            assert_eq!(out, [7; 12]);
            runtime
                .block_on(store.head("bucket", "doc_ids.knndid"))
                .unwrap();
            let mut upload = store.upload("bucket", "index.faiss", 12, None, &Cancel::new());
            upload.write_all(&[7; 12]).unwrap();
            upload.finish().unwrap();
            assert_eq!(
                fake.signed(),
                [
                    signed("HEAD", VECTORS, "us-east-1"),
                    signed("HEAD", VECTORS, "eu-west-1"),
                    signed("GET", VECTORS, "eu-west-1"),
                    signed("HEAD", IDS, "eu-west-1"),
                    signed("PUT", INDEX, "eu-west-1"),
                ],
                "status {status}"
            );
            assert_eq!(region(&store.client_for("bucket")), "eu-west-1");
            // Other buckets keep the store's region.
            assert_eq!(region(&store.client_for("other")), "us-east-1");
        }
    }

    #[test]
    fn asks_the_bucket_when_the_redirect_names_no_region() {
        let fake = elsewhere("eu-west-1", 301, &["", "eu-west-1"]);
        let runtime = Runtime::new().unwrap();
        let store = runtime.block_on(test_store(Some(&fake), Transfers::default()));
        let blob = runtime
            .block_on(store.head("bucket", "vectors.knnvec"))
            .unwrap();
        assert_eq!(blob.size, 12);
        assert_eq!(
            fake.signed(),
            [
                signed("HEAD", VECTORS, "us-east-1"),
                signed("HEAD", "/bucket/", "us-east-1"),
                signed("HEAD", VECTORS, "eu-west-1"),
            ]
        );
        assert_eq!(region(&store.client_for("bucket")), "eu-west-1");
    }

    #[test]
    fn keeps_a_redirect_that_names_no_region() {
        let fake = elsewhere("eu-west-1", 301, &[]);
        let runtime = Runtime::new().unwrap();
        let store = runtime.block_on(test_store(Some(&fake), Transfers::default()));
        let error = runtime
            .block_on(store.head("bucket", "vectors.knnvec"))
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "HeadObject s3://bucket/vectors.knnvec: \
             PermanentRedirect (HTTP 301): the bucket is in another region"
        );
        assert_eq!(
            fake.signed(),
            [
                signed("HEAD", VECTORS, "us-east-1"),
                signed("HEAD", "/bucket/", "us-east-1"),
            ]
        );
        assert_eq!(region(&store.client_for("bucket")), "us-east-1");
    }

    #[test]
    fn keeps_a_redirect_to_the_region_signed_for() {
        // A 400 naming the region the request was signed for is some
        // other fault: no new client, no retry.
        let fake = elsewhere("eu-west-1", 400, &["us-east-1"]);
        let runtime = Runtime::new().unwrap();
        let store = runtime.block_on(test_store(Some(&fake), Transfers::default()));
        let error = runtime
            .block_on(store.head("bucket", "vectors.knnvec"))
            .unwrap_err();
        assert!(error.to_string().contains("HTTP 400"), "{error}");
        assert_eq!(fake.signed(), [signed("HEAD", VECTORS, "us-east-1")]);
        assert_eq!(region(&store.client_for("bucket")), "us-east-1");
    }

    #[test]
    fn keeps_a_400_from_the_region_followed() {
        // A redirect, then a 400 naming the region followed: some other
        // fault in the bucket's own region, not a second redirect.
        let fake = elsewhere("ap-south-1", 400, &["eu-west-1"]);
        let runtime = Runtime::new().unwrap();
        let store = runtime.block_on(test_store(Some(&fake), Transfers::default()));
        let error = runtime
            .block_on(store.head("bucket", "vectors.knnvec"))
            .unwrap_err();
        let message = error.to_string();
        assert!(message.contains("HTTP 400"), "{message}");
        assert!(!message.contains("redirect"), "{message}");
        assert_eq!(
            fake.signed(),
            [
                signed("HEAD", VECTORS, "us-east-1"),
                signed("HEAD", VECTORS, "eu-west-1"),
            ]
        );
    }

    #[test]
    fn keeps_a_400_that_names_no_region() {
        // No HeadBucket for it: S3 names the region on a 400 for the
        // wrong one.
        let fake = elsewhere("eu-west-1", 400, &[]);
        let runtime = Runtime::new().unwrap();
        let store = runtime.block_on(test_store(Some(&fake), Transfers::default()));
        let error = runtime
            .block_on(store.head("bucket", "vectors.knnvec"))
            .unwrap_err();
        assert!(error.to_string().contains("HTTP 400"), "{error}");
        assert_eq!(fake.signed(), [signed("HEAD", VECTORS, "us-east-1")]);
    }

    #[test]
    fn follows_one_redirect_only() {
        // The region named is not the bucket's, and the retry is sent on
        // to a third: the error says which redirect was followed.
        let fake = elsewhere("ap-south-1", 301, &["eu-west-1", "ap-southeast-2"]);
        let runtime = Runtime::new().unwrap();
        let store = runtime.block_on(test_store(Some(&fake), Transfers::default()));
        let error = runtime
            .block_on(store.head("bucket", "vectors.knnvec"))
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "HeadObject s3://bucket/vectors.knnvec: \
             PermanentRedirect (HTTP 301): the bucket is in another region, \
             after a redirect to eu-west-1"
        );
        assert_eq!(
            fake.signed(),
            [
                signed("HEAD", VECTORS, "us-east-1"),
                signed("HEAD", VECTORS, "eu-west-1"),
            ]
        );
        assert_eq!(region(&store.client_for("bucket")), "eu-west-1");
    }

    #[tokio::test]
    async fn keeps_a_bounded_set_of_redirected_clients() {
        let store = test_store(None, Transfers::default()).await;
        for i in 0..REDIRECTED_BUCKETS {
            store.redirect(&format!("bucket-{i}"), "eu-west-1");
        }
        assert_eq!(region(&store.client_for("bucket-0")), "eu-west-1");
        // One past the cap starts the set over, with the new bucket in it.
        store.redirect("bucket-more", "eu-west-1");
        assert_eq!(region(&store.client_for("bucket-more")), "eu-west-1");
        assert_eq!(region(&store.client_for("bucket-0")), "us-east-1");
        assert_eq!(store.redirected.lock().unwrap().len(), 1);
        // A bucket already in a full set is replaced in place.
        for i in 1..REDIRECTED_BUCKETS {
            store.redirect(&format!("bucket-{i}"), "eu-west-1");
        }
        store.redirect("bucket-more", "ap-south-1");
        assert_eq!(region(&store.client_for("bucket-more")), "ap-south-1");
        assert_eq!(store.redirected.lock().unwrap().len(), REDIRECTED_BUCKETS);
    }
}
