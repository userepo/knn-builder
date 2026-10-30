//! The download stage's work: a job's blobs into host memory.

use knn_gpu::fp16;
use knn_store::{Blob, Cancel, Store};

use crate::{DataType, Job, JobError, Vectors};

/// A job's inputs in host memory.
pub(crate) struct Dataset {
    pub(crate) vectors: Vectors,
    pub(crate) ids: Vec<i32>,
    /// The KMS key the index is encrypted with, when its inputs are.
    pub(crate) kms_key: Option<String>,
}

/// How a vector blob decodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Encoding {
    /// fp32 values, kept.
    F32,
    /// fp16 values, kept.
    F16,
    /// fp32 values, converted to fp16 as they arrive.
    F32ToF16,
    /// int8 values, kept.
    I8,
    /// Binary vectors' packed bits, kept.
    Bits,
}

/// The vector blob's encoding, from the request's data type and the
/// blob's size, as the service's `_determine_streaming_buffer` decides it.
pub(crate) fn encoding(job: &Job, size: u64) -> Result<Encoding, JobError> {
    let values = u64::from(job.doc_count) * u64::from(job.dimension);
    // Saturating: no blob is that large, so no size matches.
    let (fp16, fp32) = (values.saturating_mul(2), values.saturating_mul(4));
    match job.data_type {
        // Of another size, the download refuses it.
        DataType::Float => Ok(Encoding::F32),
        DataType::Byte => Ok(Encoding::I8),
        DataType::Binary => Ok(Encoding::Bits),
        DataType::HalfFloat if size == fp16 => Ok(Encoding::F16),
        DataType::HalfFloat if size == fp32 => Ok(Encoding::F32ToF16),
        DataType::HalfFloat => Err(JobError::HalfFloatSize {
            size,
            fp16,
            fp32,
            doc_count: job.doc_count,
            dimension: job.dimension,
        }),
    }
}

/// Reads both blobs' sizes and KMS keys, then downloads them, vectors
/// first, on a blocking thread.
pub(crate) async fn load(store: &Store, job: &Job, cancel: &Cancel) -> Result<Dataset, JobError> {
    let vectors = store
        .head(&job.bucket, &job.vector_path)
        .await
        .map_err(JobError::download)?;
    // The service settles the vector buffer, refusing a half_float blob of
    // neither size, before it touches the doc ids.
    let encoding = encoding(job, vectors.size)?;
    let ids = store
        .head(&job.bucket, &job.doc_id_path)
        .await
        .map_err(JobError::download)?;
    // The first key the service finds: the vectors', else the doc ids'.
    let kms_key = vectors.kms_key.clone().or_else(|| ids.kms_key.clone());
    let (store, job, cancel) = (store.clone(), job.clone(), cancel.clone());
    let fetched =
        tokio::task::spawn_blocking(move || fetch(&store, &job, &vectors, &ids, encoding, &cancel))
            .await;
    let (vectors, ids) = match fetched {
        Ok(fetched) => fetched?,
        Err(join) if join.is_panic() => return Err(JobError::Panicked { stage: "download" }),
        // The runtime is shutting down.
        Err(_) => return Err(JobError::Stopped),
    };
    Ok(Dataset {
        vectors,
        ids,
        kms_key,
    })
}

/// Downloads the vectors, decoded as `encoding` says, then the doc ids.
fn fetch(
    store: &Store,
    job: &Job,
    vectors: &Blob,
    ids: &Blob,
    encoding: Encoding,
    cancel: &Cancel,
) -> Result<(Vectors, Vec<i32>), JobError> {
    let values = u64::from(job.doc_count) * u64::from(job.dimension);
    let (bucket, key) = (job.bucket.as_str(), job.vector_path.as_str());
    let vectors = match encoding {
        Encoding::F32 => {
            let mut out = zeroed(values)?;
            store
                .download(bucket, key, vectors, &mut out, f32::from_le_bytes, cancel)
                .map_err(JobError::download)?;
            Vectors::F32(out)
        }
        Encoding::F16 => {
            let mut out = zeroed(values)?;
            store
                .download(bucket, key, vectors, &mut out, u16::from_le_bytes, cancel)
                .map_err(JobError::download)?;
            Vectors::F16(out)
        }
        Encoding::F32ToF16 => {
            let mut out = zeroed(values)?;
            let decode = |b| fp16::f32_to_f16(f32::from_le_bytes(b));
            store
                .download(bucket, key, vectors, &mut out, decode, cancel)
                .map_err(JobError::download)?;
            Vectors::F16(out)
        }
        Encoding::I8 => {
            let mut out = zeroed(values)?;
            store
                .download(bucket, key, vectors, &mut out, i8::from_le_bytes, cancel)
                .map_err(JobError::download)?;
            Vectors::I8(out)
        }
        Encoding::Bits => {
            // dimension / 8 bytes a vector, not one a value.
            let mut out = zeroed(u64::from(job.doc_count) * (u64::from(job.dimension) / 8))?;
            store
                .download(bucket, key, vectors, &mut out, u8::from_le_bytes, cancel)
                .map_err(JobError::download)?;
            Vectors::Bits(out)
        }
    };
    let mut doc_ids = zeroed(u64::from(job.doc_count))?;
    store
        .download(
            bucket,
            &job.doc_id_path,
            ids,
            &mut doc_ids,
            i32::from_le_bytes,
            cancel,
        )
        .map_err(JobError::download)?;
    Ok((vectors, doc_ids))
}

/// `len` zero values, or [`JobError::HostMemory`] when they cannot be
/// allocated. A fallible reservation checks first, so a buffer too large
/// fails its job where `vec!` would abort the process; the buffer itself
/// then comes from `vec!`, whose fresh zero pages need no pass over them.
/// A `resize` writes every byte before the download does: for 3 GB, a
/// pass of 0.8 to 1.1 s under WSL (0.5 s on Windows), of which 0.2 to
/// 0.6 s (0.27 s) is saved end to end, the rest being page faults the
/// download's writes then take instead.
fn zeroed<T: Clone + Default>(len: u64) -> Result<Vec<T>, JobError> {
    let too_large = || JobError::HostMemory { values: len };
    let len = usize::try_from(len).map_err(|_| too_large())?;
    let mut probe: Vec<T> = Vec::new();
    probe.try_reserve_exact(len).map_err(|_| too_large())?;
    drop(probe);
    // Should memory run out between the two, this aborts, as the
    // download's first writes to the pages would under Linux's overcommit.
    Ok(vec![T::default(); len])
}

#[cfg(test)]
mod tests {
    use super::{Encoding, encoding, zeroed};
    use crate::job::tests::job;
    use crate::{DataType, JobError};

    #[test]
    fn keeps_byte_blobs_as_int8() {
        let mut job = job("bucket", 1_000, 8);
        job.data_type = DataType::Byte;
        // Of any size: the download checks it, as for float.
        assert_eq!(encoding(&job, 8_000), Ok(Encoding::I8));
        assert_eq!(encoding(&job, 32_000), Ok(Encoding::I8));
    }

    #[test]
    fn tells_half_float_blobs_apart_by_size() {
        let mut job = job("bucket", 1_000, 8);
        // A float blob of any size is fp32; the download checks its size.
        assert_eq!(encoding(&job, 32_000), Ok(Encoding::F32));
        assert_eq!(encoding(&job, 7), Ok(Encoding::F32));

        job.data_type = DataType::HalfFloat;
        assert_eq!(encoding(&job, 16_000), Ok(Encoding::F16));
        assert_eq!(encoding(&job, 32_000), Ok(Encoding::F32ToF16));
        let error = encoding(&job, 24_000).unwrap_err();
        assert_eq!(
            error.to_string(),
            "half_float vector blob of 24000 bytes matches neither fp16 (16000) nor fp32 (32000) for doc_count=1000, dimension=8"
        );
    }

    #[test]
    fn refuses_buffers_too_large_for_memory() {
        assert_eq!(
            zeroed::<f32>(u64::MAX).unwrap_err(),
            JobError::HostMemory { values: u64::MAX }
        );
        assert_eq!(zeroed::<f32>(3).unwrap(), [0.0; 3]);
    }
}
