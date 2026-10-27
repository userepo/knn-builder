//! Transfers against a real S3 API: LocalStack in CI. Ignored by default;
//! run them with the service's own settings pointing at LocalStack:
//!
//! ```text
//! S3_ENDPOINT_URL=http://localhost:4566 AWS_DEFAULT_REGION=us-east-1 \
//! AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test \
//! cargo test -p knn-store -- --ignored
//! ```
//!
//! Each test makes a bucket of its own and removes it when it passes.

// The clippy.toml test allowances cover `#[test]` functions, not the
// helpers beside them; in a test crate panicking is the failure mechanism
// everywhere.
#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    reason = "test crate"
)]

use std::io::Write;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use aws_sdk_s3::Client;
use aws_sdk_s3::primitives::ByteStream;
use aws_sdk_s3::types::{
    BucketLocationConstraint, CreateBucketConfiguration, ServerSideEncryption,
};
use half::f16;
use knn_store::{Cancel, Error, Settings, Store, Transfers, Upload};

const NEEDS_S3: &str =
    "needs LocalStack: set S3_ENDPOINT_URL and credentials (see the file's docs)";
const MIB: usize = 1 << 20;

/// Small ranges and parts, so modest blobs take several.
fn transfers() -> Transfers {
    Transfers {
        range_size: 1_000_003,
        download_concurrency: 4,
        part_size: 5 * MIB,
        upload_concurrency: 3,
    }
}

struct S3 {
    store: Store,
    bucket: String,
}

impl S3 {
    async fn new(test: &str) -> Self {
        let settings = Settings::from_env();
        assert!(settings.endpoint_url.is_some(), "{NEEDS_S3}");
        let store = Store::new(&settings, transfers()).await;
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let bucket = format!("knn-store-{test}-{nanos}");
        let client = store.sdk_client();
        let region = client.config().region().unwrap().to_string();
        let mut create = client.create_bucket().bucket(&bucket);
        if region != "us-east-1" {
            create = create.create_bucket_configuration(
                CreateBucketConfiguration::builder()
                    .location_constraint(BucketLocationConstraint::from(region.as_str()))
                    .build(),
            );
        }
        create.send().await.unwrap();
        Self { store, bucket }
    }

    /// The store's own client, to set up and inspect the bucket.
    fn client(&self) -> &Client {
        self.store.sdk_client()
    }

    /// Deletes the bucket and everything in it.
    async fn remove(self) {
        let client = self.client();
        let uploads = client
            .list_multipart_uploads()
            .bucket(&self.bucket)
            .send()
            .await
            .unwrap();
        for upload in uploads.uploads() {
            client
                .abort_multipart_upload()
                .bucket(&self.bucket)
                .key(upload.key().unwrap())
                .upload_id(upload.upload_id().unwrap())
                .send()
                .await
                .unwrap();
        }
        let objects = client
            .list_objects_v2()
            .bucket(&self.bucket)
            .send()
            .await
            .unwrap();
        for object in objects.contents() {
            client
                .delete_object()
                .bucket(&self.bucket)
                .key(object.key().unwrap())
                .send()
                .await
                .unwrap();
        }
        client
            .delete_bucket()
            .bucket(&self.bucket)
            .send()
            .await
            .unwrap();
    }

    async fn put(&self, key: &str, bytes: Vec<u8>) {
        self.client()
            .put_object()
            .bucket(&self.bucket)
            .key(key)
            .body(ByteStream::from(bytes))
            .send()
            .await
            .unwrap();
    }

    async fn get(&self, key: &str) -> Option<Vec<u8>> {
        let object = self
            .client()
            .get_object()
            .bucket(&self.bucket)
            .key(key)
            .send()
            .await
            .ok()?;
        Some(object.body.collect().await.unwrap().to_vec())
    }

    async fn pending_uploads(&self) -> usize {
        self.client()
            .list_multipart_uploads()
            .bucket(&self.bucket)
            .send()
            .await
            .unwrap()
            .uploads()
            .len()
    }

    /// Waits until no multipart upload is pending, for up to five
    /// seconds: a dropped `Upload` aborts on its own time.
    async fn settled_uploads(&self) -> usize {
        let mut pending = self.pending_uploads().await;
        for _ in 0..50 {
            if pending == 0 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
            pending = self.pending_uploads().await;
        }
        pending
    }

    /// Downloads `len` values of `key` on a blocking thread.
    async fn download<T, D, const N: usize>(
        &self,
        key: &str,
        len: usize,
        decode: D,
    ) -> Result<Vec<T>, Error>
    where
        T: Default + Clone + Send + 'static,
        D: Fn([u8; N]) -> T + Copy + Send + Sync + 'static,
    {
        let blob = self.store.head(&self.bucket, key).await?;
        let (store, bucket, key) = (self.store.clone(), self.bucket.clone(), key.to_owned());
        tokio::task::spawn_blocking(move || {
            let mut out = vec![T::default(); len];
            store
                .download(&bucket, &key, &blob, &mut out, decode, &Cancel::new())
                .map(|()| out)
        })
        .await
        .unwrap()
    }

    /// Starts an upload of a file of `size` bytes.
    fn start(&self, key: &str, size: usize) -> Upload {
        self.store
            .upload(&self.bucket, key, size as u64, None, &Cancel::new())
    }

    /// Uploads `bytes` as a file of `size`, written in uneven chunks on a
    /// blocking thread, and finishes the upload.
    async fn upload(
        &self,
        key: &str,
        size: u64,
        bytes: Vec<u8>,
        kms_key: Option<&str>,
    ) -> Result<(), Error> {
        let mut upload = self
            .store
            .upload(&self.bucket, key, size, kms_key, &Cancel::new());
        tokio::task::spawn_blocking(move || {
            for chunk in bytes.chunks(777_777) {
                if upload.write_all(chunk).is_err() {
                    break;
                }
            }
            upload.finish()
        })
        .await
        .unwrap()
    }
}

fn le<const N: usize, T>(values: &[T], to: fn(T) -> [u8; N]) -> Vec<u8>
where
    T: Copy,
{
    values.iter().flat_map(|&v| to(v)).collect()
}

fn bytes(len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| u8::try_from((i * 31 + i / 4099) % 251).unwrap())
        .collect()
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs LocalStack"]
async fn downloads_values_in_ranges() {
    let s3 = S3::new("ranges").await;
    // 1,500,001 values: six ranges of 250,000 (1,000,003 bytes rounded
    // down to whole values) and one of a single value.
    let values: Vec<f32> = (0..1_500_001_u32)
        .map(|i| f32::from(u16::try_from(i % 65_521).unwrap()) * 0.37 - 9_000.0)
        .collect();
    s3.put("vectors.knnvec", le(&values, f32::to_le_bytes))
        .await;
    let blob = s3.store.head(&s3.bucket, "vectors.knnvec").await.unwrap();
    assert_eq!(blob.size, 4 * 1_500_001);
    assert_eq!(blob.kms_key, None);

    let out = s3
        .download("vectors.knnvec", values.len(), f32::from_le_bytes)
        .await
        .unwrap();
    assert!(out == values, "fp32 values differ");

    // The same blob into fp16, converted as each range arrives.
    let halves = s3
        .download("vectors.knnvec", values.len(), |b| {
            f16::from_f32(f32::from_le_bytes(b)).to_bits()
        })
        .await
        .unwrap();
    let expected: Vec<u16> = values.iter().map(|&v| f16::from_f32(v).to_bits()).collect();
    assert!(halves == expected, "fp16 values differ");
    s3.remove().await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs LocalStack"]
async fn downloads_a_small_blob_in_one_get() {
    let s3 = S3::new("small").await;
    let ids: Vec<i32> = (0..1_000).map(|i| i * 7 - 300).collect();
    s3.put("ids.knndid", le(&ids, i32::to_le_bytes)).await;
    let out = s3
        .download("ids.knndid", ids.len(), i32::from_le_bytes)
        .await
        .unwrap();
    assert_eq!(out, ids);
    s3.remove().await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs LocalStack"]
async fn refuses_a_blob_of_another_size() {
    let s3 = S3::new("size").await;
    s3.put("ids.knndid", bytes(4_000)).await;
    let error = s3
        .download("ids.knndid", 999, i32::from_le_bytes)
        .await
        .unwrap_err();
    assert_eq!(
        error,
        Error::Size {
            bucket: s3.bucket.clone(),
            key: "ids.knndid".to_owned(),
            expected: 3_996,
            actual: 4_000,
        }
    );
    s3.remove().await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs LocalStack"]
async fn fails_on_a_missing_blob() {
    let s3 = S3::new("missing").await;
    let error = s3
        .store
        .head(&s3.bucket, "nowhere.knnvec")
        .await
        .unwrap_err();
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
    s3.remove().await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs LocalStack"]
async fn fails_when_the_blob_changes_after_head() {
    let s3 = S3::new("changed").await;
    s3.put("vectors.knnvec", vec![1; 3 * MIB]).await;
    let blob = s3.store.head(&s3.bucket, "vectors.knnvec").await.unwrap();
    s3.put("vectors.knnvec", vec![2; 3 * MIB]).await;
    let (store, bucket) = (s3.store.clone(), s3.bucket.clone());
    let error = tokio::task::spawn_blocking(move || {
        let mut out = vec![0_u8; 3 * MIB];
        store.download(
            &bucket,
            "vectors.knnvec",
            &blob,
            &mut out,
            u8::from_le_bytes,
            &Cancel::new(),
        )
    })
    .await
    .unwrap()
    .unwrap_err();
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
    s3.remove().await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs LocalStack"]
async fn uploads_a_small_file_in_one_put() {
    let s3 = S3::new("put").await;
    let file = bytes(100_000);
    s3.upload("index.faiss", 100_000, file.clone(), None)
        .await
        .unwrap();
    assert!(
        s3.get("index.faiss").await == Some(file),
        "stored file differs"
    );
    assert_eq!(s3.pending_uploads().await, 0);
    s3.remove().await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs LocalStack"]
async fn uploads_a_large_file_in_parts() {
    let s3 = S3::new("parts").await;
    let file = bytes(2 * 5 * MIB + 12_345);
    s3.upload("index.faiss", file.len() as u64, file.clone(), None)
        .await
        .unwrap();
    assert!(
        s3.get("index.faiss").await == Some(file),
        "stored file differs"
    );
    let first = s3
        .client()
        .head_object()
        .bucket(&s3.bucket)
        .key("index.faiss")
        .part_number(1)
        .send()
        .await
        .unwrap();
    assert_eq!(first.parts_count(), Some(3));
    assert_eq!(s3.pending_uploads().await, 0);
    s3.remove().await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs LocalStack"]
async fn aborts_a_short_upload() {
    let s3 = S3::new("short").await;
    for (size, written) in [(1_000, 500), (12 * MIB, 7 * MIB)] {
        let error = s3
            .upload("index.faiss", size as u64, bytes(written), None)
            .await
            .unwrap_err();
        assert_eq!(
            error,
            Error::Incomplete {
                bucket: s3.bucket.clone(),
                key: "index.faiss".to_owned(),
                written: written as u64,
                size: size as u64,
            }
        );
        assert_eq!(s3.get("index.faiss").await, None);
        assert_eq!(s3.pending_uploads().await, 0);
    }
    s3.remove().await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs LocalStack"]
async fn aborts_an_upload_written_past_its_size() {
    let s3 = S3::new("long").await;
    // Within a part; one part exactly and two, every part sent before the
    // extra byte; and a part and some.
    for size in [1_000, 5 * MIB, 10 * MIB, 6 * MIB] {
        let mut upload = s3.start("index.faiss", size);
        let error = tokio::task::spawn_blocking(move || {
            upload.write_all(&bytes(size)).unwrap();
            // The byte past the size fails its write.
            assert!(upload.write_all(&[0]).is_err());
            upload.finish().unwrap_err()
        })
        .await
        .unwrap();
        assert_eq!(
            error,
            Error::Incomplete {
                bucket: s3.bucket.clone(),
                key: "index.faiss".to_owned(),
                written: size as u64 + 1,
                size: size as u64,
            }
        );
        assert_eq!(s3.get("index.faiss").await, None);
        assert_eq!(s3.pending_uploads().await, 0);
    }
    s3.remove().await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs LocalStack"]
async fn aborts_a_dropped_upload() {
    let s3 = S3::new("dropped").await;
    // Mid-file, and with every byte written. Each writes at least two
    // parts: the task takes the first only once the multipart upload
    // exists, so it exists when the `Upload` is dropped, and its end
    // shows. A drop of a one-PUT file shows nothing to wait for;
    // `aborts_on_request` covers that path, which is the drop's plus a
    // wait.
    for (size, written) in [(24 * MIB, 17 * MIB), (10 * MIB, 10 * MIB)] {
        let mut upload = s3.start("index.faiss", size);
        tokio::task::spawn_blocking(move || upload.write_all(&bytes(written)).unwrap())
            .await
            .unwrap();
        assert_eq!(s3.settled_uploads().await, 0, "{written} of {size}");
        assert_eq!(s3.get("index.faiss").await, None, "{written} of {size}");
    }
    s3.remove().await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs LocalStack"]
async fn aborts_on_request() {
    let s3 = S3::new("abort").await;
    // Every byte written, in one PUT (one part, and none) and in parts:
    // `abort` returns once the task is done, and nothing was stored.
    for size in [5 * MIB, 0, 10 * MIB] {
        let mut upload = s3.start("index.faiss", size);
        tokio::task::spawn_blocking(move || {
            upload.write_all(&bytes(size)).unwrap();
            upload.abort()
        })
        .await
        .unwrap()
        .unwrap();
        assert_eq!(s3.get("index.faiss").await, None, "{size} bytes");
        assert_eq!(s3.pending_uploads().await, 0, "{size} bytes");
    }
    s3.remove().await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs LocalStack"]
async fn cancels_an_upload() {
    let s3 = S3::new("cancel").await;
    // In parts, two of them sent; and in one PUT, before it goes out.
    for (size, written) in [(24 * MIB, 11 * MIB), (1_000, 500)] {
        let cancel = Cancel::new();
        let mut upload = s3
            .store
            .upload(&s3.bucket, "index.faiss", size as u64, None, &cancel);
        let mut upload = tokio::task::spawn_blocking(move || {
            upload.write_all(&bytes(written)).unwrap();
            upload
        })
        .await
        .unwrap();
        cancel.cancel();
        // The writer carries on, and learns of it at a part or at
        // `finish`.
        let error = tokio::task::spawn_blocking(move || {
            _ = upload.write_all(&bytes(size - written));
            upload.finish().unwrap_err()
        })
        .await
        .unwrap();
        assert_eq!(
            error,
            Error::Canceled {
                bucket: s3.bucket.clone(),
                key: "index.faiss".to_owned(),
            },
            "{size} bytes"
        );
        assert_eq!(s3.get("index.faiss").await, None, "{size} bytes");
        assert_eq!(s3.pending_uploads().await, 0, "{size} bytes");
    }
    s3.remove().await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs LocalStack"]
async fn fails_when_the_upload_is_aborted_elsewhere() {
    let s3 = S3::new("elsewhere").await;
    let mut upload = s3.start("index.faiss", 24 * MIB);
    // Three parts in: the multipart upload exists.
    let mut upload = tokio::task::spawn_blocking(move || {
        upload.write_all(&bytes(15 * MIB)).unwrap();
        upload
    })
    .await
    .unwrap();
    let uploads = s3
        .client()
        .list_multipart_uploads()
        .bucket(&s3.bucket)
        .send()
        .await
        .unwrap();
    s3.client()
        .abort_multipart_upload()
        .bucket(&s3.bucket)
        .key("index.faiss")
        .upload_id(uploads.uploads()[0].upload_id().unwrap())
        .send()
        .await
        .unwrap();

    // The parts after it fail, and so does the task's own abort.
    let error = tokio::task::spawn_blocking(move || {
        _ = upload.write_all(&bytes(9 * MIB));
        upload.finish().unwrap_err()
    })
    .await
    .unwrap();
    let Error::NotAborted { cause, .. } = &error else {
        panic!("{error}");
    };
    assert!(
        matches!(
            **cause,
            Error::S3 {
                call: "UploadPart",
                ..
            }
        ),
        "{error}"
    );
    assert_eq!(s3.get("index.faiss").await, None);
    assert_eq!(s3.pending_uploads().await, 0);
    s3.remove().await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs LocalStack"]
async fn encrypts_with_the_blob_kms_key() {
    let s3 = S3::new("kms").await;
    let key = "arn:aws:kms:us-east-1:000000000000:key/7d1ba2b8-1d14-4c2a-9a8c-0f2d1c3e4b5a";
    s3.client()
        .put_object()
        .bucket(&s3.bucket)
        .key("vectors.knnvec")
        .body(ByteStream::from(bytes(4_000)))
        .server_side_encryption(ServerSideEncryption::AwsKms)
        .ssekms_key_id(key)
        .send()
        .await
        .unwrap();
    let blob = s3.store.head(&s3.bucket, "vectors.knnvec").await.unwrap();
    assert_eq!(blob.kms_key.as_deref(), Some(key));

    for size in [100_000, 6 * MIB] {
        s3.upload(
            "index.faiss",
            size as u64,
            bytes(size),
            blob.kms_key.as_deref(),
        )
        .await
        .unwrap();
        let index = s3.store.head(&s3.bucket, "index.faiss").await.unwrap();
        assert_eq!(index.kms_key.as_deref(), Some(key), "{size} bytes");
    }
    s3.remove().await;
}
