use std::fmt;
use std::io::{self, Write};
use std::mem;
use std::pin::pin;

use aws_sdk_s3::Client;
use aws_sdk_s3::primitives::ByteStream;
use aws_sdk_s3::types::{
    ChecksumAlgorithm, CompletedMultipartUpload, CompletedPart, ServerSideEncryption,
};
use futures_util::StreamExt;
use futures_util::stream::FuturesUnordered;
use tokio::sync::{mpsc, oneshot};

use crate::store::{Store, s3_error};
use crate::{Cancel, Error};

/// S3's limit on parts per upload.
const MAX_PARTS: u64 = 10_000;

impl Store {
    /// Starts uploading a file of `size` bytes to `bucket`/`key`, written
    /// through the [`Upload`] returned, encrypted with `kms_key` under
    /// SSE-KMS when one is given. A file of up to
    /// [`Transfers::part_size`] bytes goes up in one `PutObject` once
    /// written; a larger one in a multipart upload, each part sent as soon
    /// as it is written, [`Transfers::upload_concurrency`] at a time. The
    /// file, or each part, carries a CRC32 checksum, as the service's
    /// uploads do.
    ///
    /// `cancel` gives the upload up from another thread: it takes no more
    /// parts, so the writer's next part or [`Upload::finish`] fails with
    /// [`Error::Canceled`], and it is aborted once the parts in flight end,
    /// so none lands after the abort. Nothing is stored, unless the cancel
    /// comes after the last request went out.
    ///
    /// The requests run on the runtime the store was created on.
    ///
    /// [`Transfers::part_size`]: crate::Transfers::part_size
    /// [`Transfers::upload_concurrency`]: crate::Transfers::upload_concurrency
    #[must_use]
    pub fn upload(
        &self,
        bucket: &str,
        key: &str,
        size: u64,
        kms_key: Option<&str>,
        cancel: &Cancel,
    ) -> Upload {
        let part_size = part_size(size, self.transfers.part_size);
        let (parts, received) = mpsc::channel(1);
        let (done, writer_done) = oneshot::channel();
        let (result_sender, result) = oneshot::channel();
        let object = Object {
            client: self.client.clone(),
            bucket: bucket.to_owned(),
            key: key.to_owned(),
            size,
            kms_key: kms_key.map(str::to_owned),
            cancel: cancel.clone(),
        };
        let concurrency = self.transfers.upload_concurrency;
        self.runtime.spawn(async move {
            let outcome = if size <= part_size as u64 {
                object.put_whole(received, writer_done).await
            } else {
                object.put_parts(concurrency, received, writer_done).await
            };
            // The `Upload` may be gone, dropped before it finished.
            _ = result_sender.send(outcome);
        });
        Upload {
            bucket: bucket.to_owned(),
            key: key.to_owned(),
            size,
            written: 0,
            part_size,
            part: Vec::with_capacity(capped(size, part_size)),
            state: State::Writing {
                parts,
                done,
                result,
            },
        }
    }
}

/// The part size for a file of `size` bytes: `smallest`, or more when the
/// file would take over [`MAX_PARTS`] parts of that size.
fn part_size(size: u64, smallest: usize) -> usize {
    let least = size.div_ceil(MAX_PARTS);
    usize::try_from(least).map_or(usize::MAX, |least| least.max(smallest))
}

/// `size` as a `usize`, at most `cap`.
fn capped(size: u64, cap: usize) -> usize {
    usize::try_from(size).map_or(cap, |size| size.min(cap))
}

/// The writing end of an upload from [`Store::upload`]: a [`Write`] that
/// cuts the file into parts and hands each to the upload's task once
/// full. It holds one part; the task holds the parts in flight and one
/// waiting, and a writer ahead of the upload waits for room.
///
/// Call [`Upload::finish`] once the file is written. The file is stored
/// only when exactly the size declared was written, without error, and
/// `finish` called; otherwise the upload is aborted and nothing stored.
/// [`Upload::abort`] gives up and waits for the abort; a dropped `Upload`
/// aborts without waiting. A runtime that shuts down mid-upload leaves
/// its parts to the bucket's lifecycle rules.
///
/// Its methods block: it belongs on a blocking thread, such as one from
/// `tokio::task::spawn_blocking`, and they panic in an async context.
pub struct Upload {
    bucket: String,
    key: String,
    size: u64,
    /// Bytes written; on a write past the size, the bytes tried.
    written: u64,
    part_size: usize,
    part: Vec<u8>,
    state: State,
}

/// Where an upload stands, as its writer sees it.
enum State {
    /// Taking writes.
    Writing {
        /// Parts to the task; dropped, it ends the parts.
        parts: mpsc::Sender<Vec<u8>>,
        /// The writer's byte count, sent once it stops writing. The task
        /// stores the file only when the count is the size; dropped
        /// unsent, as when the `Upload` is, the task aborts.
        done: oneshot::Sender<u64>,
        result: oneshot::Receiver<Result<(), Error>>,
    },
    /// The task's result: the file stored, or the upload aborted.
    Settled(Result<(), Error>),
}

impl Upload {
    /// Sends the last part, and waits until the file is stored or the
    /// upload aborted.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Incomplete`] when the bytes written, or tried, are
    /// not the size declared; [`Error::S3`] when a request failed;
    /// [`Error::Canceled`] when the upload was canceled;
    /// [`Error::NotAborted`] when aborting failed too; or
    /// [`Error::Stopped`] when the task ended without a result.
    pub fn finish(mut self) -> Result<(), Error> {
        if self.written == self.size && !self.part.is_empty() {
            // A task that stopped taking parts has failed; `settle` says
            // why.
            self.send();
        }
        self.settle()
    }

    /// Gives the upload up, whatever was written, and waits until it is
    /// aborted: nothing is stored. Dropping the `Upload` does the same
    /// without waiting.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotAborted`] when the abort failed, leaving parts
    /// in the bucket, and [`Error::Stopped`] when the task ended without a
    /// result.
    pub fn abort(mut self) -> Result<(), Error> {
        if let State::Writing {
            parts,
            done,
            result,
        } = mem::replace(&mut self.state, State::Settled(Ok(())))
        {
            // Without the writer's count, the task stores nothing.
            drop(done);
            drop(parts);
            let outcome = result.blocking_recv().unwrap_or_else(|_| {
                Err(Error::Stopped {
                    bucket: self.bucket.clone(),
                    key: self.key.clone(),
                })
            });
            self.state = State::Settled(outcome);
        }
        match &self.state {
            State::Settled(Err(error @ (Error::NotAborted { .. } | Error::Stopped { .. }))) => {
                Err(error.clone())
            }
            // An incomplete upload or a failed request, aborted. The task
            // cannot store the file without the count, so a success here
            // only follows `finish`, which consumes the `Upload`.
            _ => Ok(()),
        }
    }

    /// Hands the full part to the task. Returns false when the task has
    /// stopped taking parts: it failed, and has aborted or is aborting.
    fn send(&mut self) -> bool {
        let State::Writing { parts, .. } = &self.state else {
            return false;
        };
        let next = capped(self.size.saturating_sub(self.written), self.part_size);
        let part = mem::replace(&mut self.part, Vec::with_capacity(next));
        parts.blocking_send(part).is_ok()
    }

    /// Gives the task the writer's count and waits for its result, once.
    fn settle(&mut self) -> Result<(), Error> {
        let stopped = Error::Stopped {
            bucket: self.bucket.clone(),
            key: self.key.clone(),
        };
        let outcome = match mem::replace(&mut self.state, State::Settled(Err(stopped.clone()))) {
            State::Settled(outcome) => outcome,
            State::Writing {
                parts,
                done,
                result,
            } => {
                // A task that has stopped no longer listens.
                _ = done.send(self.written);
                drop(parts);
                result.blocking_recv().unwrap_or(Err(stopped))
            }
        };
        self.state = State::Settled(outcome.clone());
        outcome
    }

    /// Settles the upload after a failed write, and returns the error the
    /// write returns.
    fn failed(&mut self) -> io::Error {
        let error = self.settle().err().unwrap_or_else(|| Error::Incomplete {
            bucket: self.bucket.clone(),
            key: self.key.clone(),
            written: self.written,
            size: self.size,
        });
        io::Error::other(error)
    }
}

impl Write for Upload {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if matches!(self.state, State::Settled(_)) {
            return Err(self.failed());
        }
        let past = self.written.saturating_add(buf.len() as u64);
        if past > self.size {
            self.written = past;
            return Err(self.failed());
        }
        let room = self.part_size - self.part.len();
        let (taken, _) = buf.split_at(buf.len().min(room));
        self.part.extend_from_slice(taken);
        self.written += taken.len() as u64;
        if self.part.len() == self.part_size && !self.send() {
            return Err(self.failed());
        }
        Ok(taken.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// The object and progress, not the part's bytes.
impl fmt::Debug for Upload {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Upload")
            .field("bucket", &self.bucket)
            .field("key", &self.key)
            .field("size", &self.size)
            .field("written", &self.written)
            .field("part_size", &self.part_size)
            .finish_non_exhaustive()
    }
}

/// The object an upload's task stores.
struct Object {
    client: Client,
    bucket: String,
    key: String,
    size: u64,
    kms_key: Option<String>,
    cancel: Cancel,
}

impl Object {
    /// Gathers the file, then stores it in one `PutObject`.
    async fn put_whole(
        &self,
        mut received: mpsc::Receiver<Vec<u8>>,
        writer_done: oneshot::Receiver<u64>,
    ) -> Result<(), Error> {
        let mut canceled = pin!(self.cancel.canceled());
        // One part, which becomes the body as it is.
        let mut body = Vec::new();
        loop {
            let part = tokio::select! {
                part = received.recv() => part,
                () = &mut canceled => return Err(self.canceled()),
            };
            let Some(part) = part else { break };
            if body.is_empty() {
                body = part;
            } else {
                body.extend_from_slice(&part);
            }
        }
        self.check_done(writer_done, body.len() as u64).await?;
        if self.cancel.is_canceled() {
            return Err(self.canceled());
        }
        let (encryption, kms_key) = self.encryption();
        self.client
            .put_object()
            .bucket(&self.bucket)
            .key(&self.key)
            .body(ByteStream::from(body))
            .checksum_algorithm(ChecksumAlgorithm::Crc32)
            .set_server_side_encryption(encryption)
            .set_ssekms_key_id(kms_key)
            .send()
            .await
            .map_err(|e| s3_error("PutObject", &self.bucket, &self.key, &e))?;
        Ok(())
    }

    /// Stores the file in a multipart upload, aborted on any error.
    async fn put_parts(
        &self,
        concurrency: usize,
        received: mpsc::Receiver<Vec<u8>>,
        writer_done: oneshot::Receiver<u64>,
    ) -> Result<(), Error> {
        let (encryption, kms_key) = self.encryption();
        let created = self
            .client
            .create_multipart_upload()
            .bucket(&self.bucket)
            .key(&self.key)
            .checksum_algorithm(ChecksumAlgorithm::Crc32)
            .set_server_side_encryption(encryption)
            .set_ssekms_key_id(kms_key)
            .send()
            .await
            .map_err(|e| s3_error("CreateMultipartUpload", &self.bucket, &self.key, &e))?;
        let id = created.upload_id().ok_or_else(|| Error::S3 {
            call: "CreateMultipartUpload",
            bucket: self.bucket.clone(),
            key: self.key.clone(),
            message: "the response has no UploadId".to_owned(),
        })?;
        match self
            .upload_parts(id, concurrency, received, writer_done)
            .await
        {
            Ok(()) => Ok(()),
            Err(error) => Err(self.abort(id, error).await),
        }
    }

    /// Uploads the parts as they arrive, `concurrency` at a time, then
    /// completes the upload. After a failed part or a cancel it takes no
    /// more and waits for those in flight, so none lands after the abort
    /// that follows: S3 may keep a part an abort raced.
    async fn upload_parts(
        &self,
        id: &str,
        concurrency: usize,
        mut received: mpsc::Receiver<Vec<u8>>,
        writer_done: oneshot::Receiver<u64>,
    ) -> Result<(), Error> {
        let mut canceled = pin!(self.cancel.canceled());
        let mut in_flight = FuturesUnordered::new();
        let mut uploaded = Vec::new();
        let mut failure = None;
        let mut ended = false;
        let mut number = 0;
        // Until no part can come and none is in flight. While none has
        // failed, the cancel branch stays open, so the loop cannot rely on
        // `select!` running out of branches.
        while !((ended || failure.is_some()) && in_flight.is_empty()) {
            let room = failure.is_none() && !ended && in_flight.len() < concurrency;
            tokio::select! {
                part = received.recv(), if room => match part {
                    Some(body) => {
                        number += 1;
                        in_flight.push(self.upload_part(id, number, body));
                    }
                    None => ended = true,
                },
                Some(part) = in_flight.next(), if !in_flight.is_empty() => match part {
                    Ok(part) => uploaded.push(part),
                    Err(error) => {
                        // The writer's next part fails, and it stops.
                        received.close();
                        failure.get_or_insert(error);
                    }
                },
                () = &mut canceled, if failure.is_none() => {
                    received.close();
                    failure = Some(self.canceled());
                },
            }
        }
        if let Some(error) = failure {
            return Err(error);
        }
        let received = uploaded.iter().map(|&(_, len)| len).sum();
        self.check_done(writer_done, received).await?;
        if self.cancel.is_canceled() {
            return Err(self.canceled());
        }
        uploaded.sort_by_key(|(part, _)| part.part_number());
        let parts = uploaded.into_iter().map(|(part, _)| part).collect();
        self.client
            .complete_multipart_upload()
            .bucket(&self.bucket)
            .key(&self.key)
            .upload_id(id)
            .multipart_upload(
                CompletedMultipartUpload::builder()
                    .set_parts(Some(parts))
                    .build(),
            )
            .send()
            .await
            .map_err(|e| s3_error("CompleteMultipartUpload", &self.bucket, &self.key, &e))?;
        Ok(())
    }

    /// Uploads one part. Returns it as `CompleteMultipartUpload` lists it,
    /// with its length.
    async fn upload_part(
        &self,
        id: &str,
        number: i32,
        body: Vec<u8>,
    ) -> Result<(CompletedPart, u64), Error> {
        let len = body.len() as u64;
        let uploaded = self
            .client
            .upload_part()
            .bucket(&self.bucket)
            .key(&self.key)
            .upload_id(id)
            .part_number(number)
            .body(ByteStream::from(body))
            .checksum_algorithm(ChecksumAlgorithm::Crc32)
            .send()
            .await
            .map_err(|e| s3_error("UploadPart", &self.bucket, &self.key, &e))?;
        let part = CompletedPart::builder()
            .part_number(number)
            .set_e_tag(uploaded.e_tag().map(str::to_owned))
            .set_checksum_crc32(uploaded.checksum_crc32().map(str::to_owned))
            .build();
        Ok((part, len))
    }

    /// Waits for the writer's count, and checks that it and the bytes
    /// received are both the size. A writer dropped before it finished
    /// never stores the file, however much of it arrived.
    async fn check_done(
        &self,
        writer_done: oneshot::Receiver<u64>,
        received: u64,
    ) -> Result<(), Error> {
        let written = match writer_done.await {
            Ok(written) if written == self.size && received == self.size => return Ok(()),
            Ok(written) => written,
            Err(_) => received,
        };
        Err(Error::Incomplete {
            bucket: self.bucket.clone(),
            key: self.key.clone(),
            written,
            size: self.size,
        })
    }

    fn canceled(&self) -> Error {
        Error::Canceled {
            bucket: self.bucket.clone(),
            key: self.key.clone(),
        }
    }

    /// Aborts the upload after `error`, so S3 drops its parts.
    async fn abort(&self, id: &str, error: Error) -> Error {
        let aborted = self
            .client
            .abort_multipart_upload()
            .bucket(&self.bucket)
            .key(&self.key)
            .upload_id(id)
            .send()
            .await;
        match aborted {
            Ok(_) => error,
            Err(e) => Error::NotAborted {
                cause: Box::new(error),
                message: s3_error("AbortMultipartUpload", &self.bucket, &self.key, &e).to_string(),
            },
        }
    }

    /// SSE-KMS with the key, when there is one.
    fn encryption(&self) -> (Option<ServerSideEncryption>, Option<String>) {
        match &self.kms_key {
            Some(key) => (Some(ServerSideEncryption::AwsKms), Some(key.clone())),
            None => (None, None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{MAX_PARTS, part_size};

    #[test]
    fn grows_parts_past_the_part_limit() {
        let mib = 1 << 20;
        assert_eq!(part_size(0, 8 * mib), 8 * mib);
        assert_eq!(part_size(3 << 30, 8 * mib), 8 * mib);
        // Exactly 10,000 parts of the smallest size, then one byte more.
        let most = MAX_PARTS * 8 * mib as u64;
        assert_eq!(part_size(most, 8 * mib), 8 * mib);
        assert_eq!(part_size(most + 1, 8 * mib), 8 * mib + 1);
        // 1 TiB takes parts of 110 MB and change.
        let tib = 1_u64 << 40;
        let size = part_size(tib, 8 * mib);
        assert_eq!(size as u64, tib.div_ceil(MAX_PARTS));
        assert!((size as u64) * MAX_PARTS >= tib);
    }
}
