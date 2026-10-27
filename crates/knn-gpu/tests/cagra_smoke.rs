//! Checks the GPU toolchain end to end: cuVS builds a CAGRA graph over
//! random vectors with the upstream service's parameters for m = 16 (graph
//! degree and intermediate graph degree 64, IVF-PQ graph build), and a
//! search for vectors of the dataset finds each one as its own nearest
//! neighbor.
//!
//! Needs the `gpu` feature, a CUDA GPU and the conda environment in
//! `environment.yml`.

use std::ffi::{c_int, c_void};
use std::marker::PhantomData;

use cuvs::Resources;
use cuvs::dlpack::{
    AsDlTensor, AsDlTensorMut, DLDevice, DLDeviceType, DLPackError, DLTensorView, DLTensorViewMut,
    DType,
};
use cuvs::neighbors::cagra::{Index, IndexParams, SearchParams};

type TestResult<T> = Result<T, Box<dyn std::error::Error>>;

const CUDA_SUCCESS: c_int = 0;
const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;

// The cuvs crate keeps its device tensor helpers private to its own
// tests, so this one allocates through the CUDA runtime, as the crate's
// CAGRA example does.
#[link(name = "cudart")]
unsafe extern "C" {
    fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> c_int;
    fn cudaFree(ptr: *mut c_void) -> c_int;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: c_int) -> c_int;
}

fn check_cuda(status: c_int, call: &str) -> TestResult<()> {
    if status == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!("{call} failed with CUDA error {status}").into())
    }
}

/// A row-major matrix in device memory, freed on drop.
struct DeviceMatrix<T: DType> {
    data: *mut c_void,
    shape: [i64; 2],
    len: usize,
    _marker: PhantomData<T>,
}

impl<T: DType> DeviceMatrix<T> {
    fn alloc(rows: usize, cols: usize) -> TestResult<Self> {
        let len = rows * cols;
        let mut data = std::ptr::null_mut();
        // SAFETY: `data` is a valid out-pointer; the size is the matrix's
        // byte length.
        let status = unsafe { cudaMalloc(&raw mut data, len * size_of::<T>()) };
        check_cuda(status, "cudaMalloc")?;
        Ok(Self {
            data,
            shape: [i64::try_from(rows)?, i64::try_from(cols)?],
            len,
            _marker: PhantomData,
        })
    }

    fn upload(host: &[T], rows: usize, cols: usize) -> TestResult<Self> {
        if host.len() != rows * cols {
            return Err(format!("{} host values for a {rows} x {cols} matrix", host.len()).into());
        }
        let matrix = Self::alloc(rows, cols)?;
        // SAFETY: both buffers hold `len` elements of `T`; the copy is
        // synchronous, so `host` outlives it.
        let status = unsafe {
            cudaMemcpy(
                matrix.data,
                host.as_ptr().cast(),
                matrix.len * size_of::<T>(),
                CUDA_MEMCPY_HOST_TO_DEVICE,
            )
        };
        check_cuda(status, "cudaMemcpy to device")?;
        Ok(matrix)
    }

    fn download(&self, host: &mut [T]) -> TestResult<()> {
        if host.len() != self.len {
            return Err(format!("a host buffer of {} for {} values", host.len(), self.len).into());
        }
        // SAFETY: both buffers hold `len` elements of `T`; the copy is
        // synchronous.
        let status = unsafe {
            cudaMemcpy(
                host.as_mut_ptr().cast(),
                self.data,
                self.len * size_of::<T>(),
                CUDA_MEMCPY_DEVICE_TO_HOST,
            )
        };
        check_cuda(status, "cudaMemcpy to host")
    }

    fn device() -> DLDevice {
        DLDevice {
            device_type: DLDeviceType::kDLCUDA,
            device_id: 0,
        }
    }
}

impl<T: DType> Drop for DeviceMatrix<T> {
    fn drop(&mut self) {
        // SAFETY: `data` came from `cudaMalloc` and is freed only here.
        // A failure cannot be reported from drop; the process ends soon.
        let _ = unsafe { cudaFree(self.data) };
    }
}

impl<T: DType> AsDlTensor for DeviceMatrix<T> {
    fn as_dl_tensor(&self) -> Result<DLTensorView<'_>, DLPackError> {
        // SAFETY: `data` points to a live device allocation of `shape`
        // elements of `T`, row-major, borrowed for the view's lifetime.
        unsafe {
            DLTensorView::from_raw_parts(
                self.data,
                Self::device(),
                &self.shape,
                None,
                T::dl_dtype(),
            )
        }
    }
}

impl<T: DType> AsDlTensorMut for DeviceMatrix<T> {
    fn as_dl_tensor_mut(&mut self) -> Result<DLTensorViewMut<'_>, DLPackError> {
        // SAFETY: as in `as_dl_tensor`, and the view borrows `self`
        // mutably, so nothing else reads the buffer meanwhile.
        unsafe {
            DLTensorViewMut::from_raw_parts(
                self.data,
                Self::device(),
                &self.shape,
                None,
                T::dl_dtype(),
            )
        }
    }
}

/// Uniform values in [0, 1) from a fixed-seed xorshift generator, so a
/// failure reproduces.
fn random_vectors(rows: usize, cols: usize) -> Vec<f32> {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    (0..rows * cols)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // The top 24 bits fill an f32 mantissa exactly.
            #[expect(clippy::cast_precision_loss, reason = "a 24-bit value is exact in f32")]
            let value = (state >> 40) as f32;
            value / 16_777_216.0
        })
        .collect()
}

const ROWS: usize = 10_000;
const DIM: usize = 32;
const QUERIES: usize = 16;
const K: usize = 10;
// m = 16 under the upstream rule: both degrees are m x 4.
const DEGREE: usize = 64;

/// Builds the graph and searches it for the first `QUERIES` rows of the
/// dataset; returns the neighbor ids and distances, `K` per query.
fn build_and_search_own_rows() -> TestResult<(Vec<u32>, Vec<f32>)> {
    let res = Resources::new()?;
    let host = random_vectors(ROWS, DIM);
    let dataset = DeviceMatrix::upload(&host, ROWS, DIM)?;

    let params = IndexParams::builder()
        .graph_degree(DEGREE)
        .intermediate_graph_degree(DEGREE)
        .ivf_pq()
        .build()?;
    let index = Index::build(&res, &params, &dataset)?;

    // The first rows of the dataset as queries: each is at distance 0
    // from itself.
    let first_rows = host.get(..QUERIES * DIM).ok_or("fewer rows than queries")?;
    let queries = DeviceMatrix::upload(first_rows, QUERIES, DIM)?;
    let mut neighbors = DeviceMatrix::<u32>::alloc(QUERIES, K)?;
    let mut distances = DeviceMatrix::<f32>::alloc(QUERIES, K)?;
    let search = SearchParams::builder().build()?;
    index.search(&res, &search, &queries, &mut neighbors, &mut distances)?;
    res.sync_stream()?;

    let mut found = vec![0_u32; QUERIES * K];
    let mut dist = vec![0_f32; QUERIES * K];
    neighbors.download(&mut found)?;
    distances.download(&mut dist)?;
    Ok((found, dist))
}

#[test]
fn builds_and_searches_a_cagra_graph() {
    let (found, dist) = build_and_search_own_rows().expect("the cuVS build or search failed");
    for query in 0..QUERIES {
        let row = &found[query * K..(query + 1) * K];
        let id = u32::try_from(query).unwrap();
        assert!(
            row.contains(&id),
            "query {query} not among its own neighbors: {row:?}"
        );
        assert!(
            dist[query * K] <= 1e-4,
            "query {query}: nearest distance {}",
            dist[query * K]
        );
    }
}
