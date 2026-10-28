//! What the LocalStack tests share: a bucket of a test's own on the S3 the
//! service's settings point at, and the doc ids the tests store.

// Each test crate uses some of it.
#![allow(dead_code, reason = "shared by test crates that each use a part")]

use std::time::{SystemTime, UNIX_EPOCH};

use aws_sdk_s3::Client;
use aws_sdk_s3::primitives::ByteStream;
use aws_sdk_s3::types::{
    BucketLocationConstraint, CreateBucketConfiguration, ServerSideEncryption,
};
use knn_store::{Settings, Store, Transfers};

pub const NEEDS_S3: &str = "needs LocalStack: set S3_ENDPOINT_URL and credentials (see the README)";

/// A bucket of the test's own, and a store on its S3.
pub struct S3 {
    pub store: Store,
    pub bucket: String,
}

impl S3 {
    pub async fn new(test: &str) -> Self {
        let settings = Settings::from_env();
        assert!(settings.endpoint_url.is_some(), "{NEEDS_S3}");
        let store = Store::new(&settings, Transfers::default()).await;
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let bucket = format!("knn-builder-{test}-{nanos}");
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

    pub fn client(&self) -> &Client {
        self.store.sdk_client()
    }

    /// Deletes the bucket and everything in it.
    pub async fn remove(self) {
        let client = self.client();
        for key in self.keys().await {
            client
                .delete_object()
                .bucket(&self.bucket)
                .key(key)
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

    /// The keys of the objects stored.
    pub async fn keys(&self) -> Vec<String> {
        let objects = self
            .client()
            .list_objects_v2()
            .bucket(&self.bucket)
            .send()
            .await
            .unwrap();
        objects
            .contents()
            .iter()
            .map(|object| object.key().unwrap().to_owned())
            .collect()
    }

    pub async fn put(&self, key: &str, bytes: Vec<u8>, kms_key: Option<&str>) {
        let mut put = self
            .client()
            .put_object()
            .bucket(&self.bucket)
            .key(key)
            .body(ByteStream::from(bytes));
        if let Some(kms_key) = kms_key {
            put = put
                .server_side_encryption(ServerSideEncryption::AwsKms)
                .ssekms_key_id(kms_key);
        }
        put.send().await.unwrap();
    }

    pub async fn get(&self, key: &str) -> Option<Vec<u8>> {
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

    pub async fn kms_key(&self, key: &str) -> Option<String> {
        self.store.head(&self.bucket, key).await.unwrap().kms_key
    }

    pub async fn pending_uploads(&self) -> usize {
        self.client()
            .list_multipart_uploads()
            .bucket(&self.bucket)
            .send()
            .await
            .unwrap()
            .uploads()
            .len()
    }
}

/// Doc ids counting down from 9,999.
pub fn ids(count: u32) -> Vec<i32> {
    (0..i32::try_from(count).unwrap())
        .map(|i| 9_999 - i)
        .collect()
}

pub fn le_ids(ids: &[i32]) -> Vec<u8> {
    ids.iter().flat_map(|v| v.to_le_bytes()).collect()
}
