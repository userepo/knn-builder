//! The CAGRA build on the GPU, through knn-gpu's C++ shim over cuVS's C++
//! API (`cpp/knn_cagra.cpp`), the API Faiss builds through, so every
//! setting reaches cuVS as the upstream service passes it
//! (`docs/params.md`).
//!
//! The shim owns what a build allocates on the device through C++ objects
//! and catches every exception, so a build that fails partway returns its
//! device memory as one that succeeds does, and reports a status.

use std::ffi::{CStr, c_char, c_int, c_void};
use std::fmt;
use std::ptr::{self, NonNull};

use crate::{BuildParams, Error, Metric};

/// The shim's C interface, `cpp/knn_cagra.h`.
mod ffi {
    use std::ffi::{c_char, c_int, c_void};
    use std::mem::offset_of;

    /// `knn_cagra_gpu`: one GPU's raft resources.
    #[repr(C)]
    pub(super) struct KnnCagraGpu {
        _opaque: [u8; 0],
    }

    pub(super) const F32: c_int = 0;
    pub(super) const F16: c_int = 1;

    pub(super) const L2: i32 = 0;
    pub(super) const INNER_PRODUCT: i32 = 1;

    pub(super) const OK: c_int = 0;
    pub(super) const RESOURCES: c_int = 1;
    pub(super) const DATASET: c_int = 2;
    pub(super) const BUILD: c_int = 3;
    pub(super) const GRAPH: c_int = 4;
    pub(super) const COPY: c_int = 5;
    pub(super) const ARGUMENTS: c_int = 6;

    /// `knn_cagra_params`.
    #[repr(C)]
    pub(super) struct Params {
        pub metric: i32,
        pub graph_degree: u32,
        pub intermediate_graph_degree: u32,
        pub n_lists: u32,
        pub kmeans_n_iters: u32,
        pub pq_bits: u32,
        pub pq_dim: u32,
        pub n_probes: u32,
        pub kmeans_trainset_fraction: f64,
        pub refinement_rate: f32,
        pub force_random_rotation: bool,
        pub conservative_memory_allocation: bool,
    }

    // The layout the shim asserts too.
    const _: () = {
        assert!(offset_of!(Params, metric) == 0);
        assert!(offset_of!(Params, graph_degree) == 4);
        assert!(offset_of!(Params, intermediate_graph_degree) == 8);
        assert!(offset_of!(Params, n_lists) == 12);
        assert!(offset_of!(Params, kmeans_n_iters) == 16);
        assert!(offset_of!(Params, pq_bits) == 20);
        assert!(offset_of!(Params, pq_dim) == 24);
        assert!(offset_of!(Params, n_probes) == 28);
        assert!(offset_of!(Params, kmeans_trainset_fraction) == 32);
        assert!(offset_of!(Params, refinement_rate) == 40);
        assert!(offset_of!(Params, force_random_rotation) == 44);
        assert!(offset_of!(Params, conservative_memory_allocation) == 45);
        assert!(size_of::<Params>() == 48);
    };

    unsafe extern "C" {
        pub(super) fn knn_cagra_gpu_create(
            gpu: *mut *mut KnnCagraGpu,
            message: *mut c_char,
            message_len: usize,
        ) -> c_int;

        pub(super) fn knn_cagra_gpu_destroy(gpu: *mut KnnCagraGpu);

        pub(super) fn knn_cagra_build(
            gpu: *mut KnnCagraGpu,
            vectors: *const c_void,
            dtype: c_int,
            rows: i64,
            dimension: i64,
            params: *const Params,
            graph: *mut u32,
            graph_len: usize,
            message: *mut c_char,
            message_len: usize,
        ) -> c_int;
    }

    // The CUDA runtime, which the build script links for cuVS.
    unsafe extern "C" {
        pub(super) fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> c_int;

        pub(super) fn cudaMalloc(pointer: *mut *mut c_void, size: usize) -> c_int;

        pub(super) fn cudaFree(pointer: *mut c_void) -> c_int;

        pub(super) fn cudaGetDevice(device: *mut c_int) -> c_int;

        pub(super) fn cudaDeviceGetAttribute(
            value: *mut c_int,
            attr: c_int,
            device: c_int,
        ) -> c_int;

        pub(super) fn cudaGetErrorString(error: c_int) -> *const c_char;
    }

    /// `cudaDevAttrMaxSharedMemoryPerBlockOptin`.
    pub(super) const MAX_SHARED_MEMORY_PER_BLOCK_OPTIN: c_int = 97;
}

/// [`Error::Cuvs`] for a CUDA runtime call that returned `status`, with
/// the runtime's text for it.
fn runtime_error(call: &'static str, status: c_int) -> Error {
    // SAFETY: the runtime returns a static NUL-terminated string for any
    // code, an unknown one included.
    let text = unsafe { CStr::from_ptr(ffi::cudaGetErrorString(status)) };
    Error::Cuvs {
        call,
        message: format!("{} (error {status})", text.to_string_lossy()),
    }
}

/// A device's memory, as the CUDA runtime counts it, in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceMemory {
    /// Free: not allocated by any process on the device, this one's
    /// builds included.
    pub free: usize,
    /// In all.
    pub total: usize,
}

/// The memory of the device current on the calling thread: the only one,
/// on a single-GPU host. The first call on a thread creates or binds the
/// device's primary context, as a build does.
///
/// # Errors
///
/// Returns [`Error::Cuvs`] naming `cudaMemGetInfo` when the runtime fails.
pub fn device_memory() -> Result<DeviceMemory, Error> {
    let (mut free, mut total) = (0, 0);
    // SAFETY: two valid out-pointers, written and not kept.
    let status = unsafe { ffi::cudaMemGetInfo(&raw mut free, &raw mut total) };
    if status == 0 {
        return Ok(DeviceMemory { free, total });
    }
    Err(runtime_error("cudaMemGetInfo", status))
}

/// The shared memory a block may take on the device current on the calling
/// thread, opting in past the default (`sharedMemPerBlockOptin`), in bytes:
/// what decides which IVF-PQ search kernel cuVS can run.
///
/// # Errors
///
/// Returns [`Error::Cuvs`] naming `cudaGetDevice` or
/// `cudaDeviceGetAttribute` when the runtime fails.
pub fn shared_memory_per_block() -> Result<usize, Error> {
    let mut device = 0;
    // SAFETY: a valid out-pointer, written and not kept.
    let status = unsafe { ffi::cudaGetDevice(&raw mut device) };
    if status != 0 {
        return Err(runtime_error("cudaGetDevice", status));
    }
    let mut bytes = 0;
    // SAFETY: a valid out-pointer, written and not kept; the attribute and
    // the device are the runtime's own values.
    let status = unsafe {
        ffi::cudaDeviceGetAttribute(
            &raw mut bytes,
            ffi::MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            device,
        )
    };
    if status != 0 {
        return Err(runtime_error("cudaDeviceGetAttribute", status));
    }
    usize::try_from(bytes).map_err(|_| Error::Cuvs {
        call: "cudaDeviceGetAttribute",
        message: format!("{bytes} bytes of shared memory a block"),
    })
}

/// Device memory held and left unused until dropped, so that less is free
/// for builds: what another process on the GPU would take. Under WSL2 the
/// driver pages device memory out to the host when a build needs more than
/// is free, so there a reservation lowers what [`device_memory`] reports
/// without keeping a build from the memory; a native Linux driver refuses
/// the build's allocation instead.
#[derive(Debug)]
pub struct Reservation {
    pointer: NonNull<c_void>,
    bytes: usize,
}

impl Reservation {
    /// Allocates `bytes` on the device current on the calling thread; 0
    /// holds nothing.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Cuvs`] naming `cudaMalloc` when the device cannot
    /// give that much.
    pub fn new(bytes: usize) -> Result<Self, Error> {
        if bytes == 0 {
            return Ok(Self {
                pointer: NonNull::dangling(),
                bytes,
            });
        }
        let mut pointer = ptr::null_mut();
        // SAFETY: a valid out-pointer, written and not kept.
        let status = unsafe { ffi::cudaMalloc(&raw mut pointer, bytes) };
        if status != 0 {
            return Err(runtime_error("cudaMalloc", status));
        }
        NonNull::new(pointer)
            .map(|pointer| Self { pointer, bytes })
            .ok_or_else(|| Error::Cuvs {
                call: "cudaMalloc",
                message: "a null pointer".to_owned(),
            })
    }

    /// The bytes held.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        if self.bytes > 0 {
            // SAFETY: the pointer came from cudaMalloc and is freed only
            // here. A failure leaves the memory held until the process
            // ends, which a drop cannot report.
            unsafe { ffi::cudaFree(self.pointer.as_ptr()) };
        }
    }
}

// SAFETY: device memory belongs to the context, not to the thread that
// allocated it; the pointer is only ever passed back to cudaFree.
unsafe impl Send for Reservation {}

/// The vectors in host memory, row-major.
#[derive(Clone, Copy)]
pub enum HostVectors<'a> {
    /// Float32 values.
    F32(&'a [f32]),
    /// IEEE half-precision bit patterns (see [`crate::fp16`]).
    F16(&'a [u16]),
}

impl HostVectors<'_> {
    fn len(&self) -> usize {
        match self {
            Self::F32(v) => v.len(),
            Self::F16(v) => v.len(),
        }
    }

    fn data(&self) -> *const c_void {
        match self {
            Self::F32(v) => v.as_ptr().cast(),
            Self::F16(v) => v.as_ptr().cast(),
        }
    }

    fn dtype(&self) -> c_int {
        match self {
            Self::F32(_) => ffi::F32,
            Self::F16(_) => ffi::F16,
        }
    }
}

/// The type and count only: a build's vectors run to millions of values.
impl fmt::Debug for HostVectors<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::F32(v) => write!(f, "F32({} values)", v.len()),
            Self::F16(v) => write!(f, "F16({} values)", v.len()),
        }
    }
}

/// A CAGRA graph in host memory: `degree` neighbor ids per vector,
/// row-major, row i for the i-th vector, in cuVS's order.
#[derive(Clone, PartialEq, Eq)]
pub struct Graph {
    neighbors: Vec<u32>,
    degree: usize,
}

impl Graph {
    /// The neighbor ids, row-major.
    #[must_use]
    pub fn neighbors(&self) -> &[u32] {
        &self.neighbors
    }

    /// The neighbor ids, row-major, without a copy.
    #[must_use]
    pub fn into_neighbors(self) -> Vec<u32> {
        self.neighbors
    }

    /// Neighbors per vector.
    #[must_use]
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Ids at or past the vector count: neighbors that name no vector.
    /// `knn-write` refuses a graph holding any; the upstream service would
    /// write them (`docs/faiss-layout.md`).
    #[must_use]
    pub fn out_of_range(&self) -> usize {
        let rows = self.neighbors.len().checked_div(self.degree).unwrap_or(0);
        self.neighbors
            .iter()
            .filter(|&&id| usize::try_from(id).map_or(true, |id| id >= rows))
            .count()
    }
}

/// The shape only, not the ids.
impl fmt::Debug for Graph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Graph")
            .field("rows", &self.neighbors.len().checked_div(self.degree))
            .field("degree", &self.degree)
            .finish()
    }
}

/// One GPU's cuVS resources, kept for the life of the process and reused
/// by every build on it.
///
/// Builds through one `Gpu` run one at a time (`build` takes `&mut self`).
/// The resources use the device current when the first build runs; a
/// `Gpu` moved to another thread must find that device current there too,
/// which holds on a single-GPU host, the only kind supported so far.
///
/// Builds on several threads at once, through separate `Gpu`s, are not
/// supported; run one build at a time in a process. The k-means that
/// trains the IVF-PQ lists (cuVS's `kmeans_balanced`) keeps cluster labels
/// and sizes in CUDA managed memory and reads them on the host. Where the
/// device does not offer concurrent managed access
/// (`cudaDevAttrConcurrentManagedAccess` is 0, as under WSL2), the driver
/// faults a host access to that memory while any kernel of the process
/// runs, so one build's read during another build's kernels ends the
/// process with SIGSEGV. Where the device does offer it, builds still race
/// on a static counter that the k-means updates without synchronization.
#[derive(Debug)]
pub struct Gpu {
    resources: Resources,
}

impl Gpu {
    /// Creates the resources, with a workspace (what cuVS plans its batches
    /// by) of a quarter of the device's memory, at most 2 GiB: so every GPU
    /// of 8 GB or more batches a build alike, and takes the device memory
    /// it takes on one.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Cuvs`] when they cannot be created.
    pub fn new() -> Result<Self, Error> {
        Ok(Self {
            resources: Resources::new()?,
        })
    }

    /// Builds the CAGRA graph of `vectors`, rows of `dimension`, as the
    /// upstream service does through Faiss: the vectors are copied to a
    /// padded device dataset, the graph is seeded by an IVF-PQ build and
    /// search with `params`, pruned to `params.graph_degree`, and copied
    /// back to the host.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Vectors`] when the values are not whole rows,
    /// [`Error::HostMemory`] when the graph would not fit in host memory,
    /// [`Error::Cuvs`] when cuVS fails (out of device memory, among
    /// others), and [`Error::Graph`] when its graph is not the n x degree
    /// array asked for. Device memory is released either way.
    pub fn build(
        &mut self,
        vectors: HostVectors<'_>,
        dimension: usize,
        params: &BuildParams,
    ) -> Result<Graph, Error> {
        let values = vectors.len();
        let not_rows = || Error::Vectors { values, dimension };
        let rows = values
            .checked_div(dimension)
            .filter(|&rows| rows > 0 && rows * dimension == values)
            .ok_or_else(not_rows)?;
        let shape = (
            i64::try_from(rows).map_err(|_| not_rows())?,
            i64::try_from(dimension).map_err(|_| not_rows())?,
        );
        let degree = params.graph_degree as usize;
        let (mut neighbors, len) = graph_buffer(rows, degree)?;
        let shim_params = shim_params(params);
        let mut message = Message::new();
        // SAFETY: the resources are live and used by this call alone
        // (`&mut self`); `vectors` holds rows x dimension values of the type
        // given, `neighbors` has room for `len` ids, rows x the degree
        // `shim_params` asks for, and `message` for its length. The shim
        // reads the vectors and parameters, writes only the graph and the
        // message, and keeps none of them.
        let status = unsafe {
            ffi::knn_cagra_build(
                self.resources.0.as_ptr(),
                vectors.data(),
                vectors.dtype(),
                shape.0,
                shape.1,
                &raw const shim_params,
                neighbors.as_mut_ptr(),
                len,
                message.as_mut_ptr(),
                message.len(),
            )
        };
        check(status, &message)?;
        // SAFETY: the capacity is at least `len`, and on success the shim
        // has copied exactly `len` ids into the buffer: it refuses a graph
        // of any other size before copying.
        unsafe { neighbors.set_len(len) };
        Ok(Graph { neighbors, degree })
    }
}

/// An empty buffer with room for rows x degree ids, and that count. It is
/// reserved, not written: a graph too large for host memory fails the
/// build instead of aborting the process, as `vec!` would.
fn graph_buffer(rows: usize, degree: usize) -> Result<(Vec<u32>, usize), Error> {
    let too_large = || Error::HostMemory { rows, degree };
    let len = rows.checked_mul(degree).ok_or_else(too_large)?;
    let mut buffer = Vec::new();
    buffer.try_reserve_exact(len).map_err(|_| too_large())?;
    Ok((buffer, len))
}

/// The shim's parameters: the request's, and the values the service fixes.
fn shim_params(params: &BuildParams) -> ffi::Params {
    ffi::Params {
        metric: match params.metric {
            Metric::L2 => ffi::L2,
            Metric::InnerProduct => ffi::INNER_PRODUCT,
        },
        graph_degree: params.graph_degree,
        intermediate_graph_degree: params.graph_degree,
        n_lists: params.n_lists,
        kmeans_n_iters: BuildParams::KMEANS_N_ITERS,
        pq_bits: BuildParams::PQ_BITS,
        pq_dim: params.pq_dim,
        n_probes: BuildParams::N_PROBES,
        kmeans_trainset_fraction: BuildParams::KMEANS_TRAINSET_FRACTION,
        refinement_rate: BuildParams::REFINEMENT_RATE,
        force_random_rotation: BuildParams::FORCE_RANDOM_ROTATION,
        conservative_memory_allocation: BuildParams::CONSERVATIVE_MEMORY_ALLOCATION,
    }
}

/// A buffer for the shim's error message.
struct Message([u8; 1024]);

impl Message {
    fn new() -> Self {
        Self([0; 1024])
    }

    fn as_mut_ptr(&mut self) -> *mut c_char {
        self.0.as_mut_ptr().cast()
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn text(&self) -> String {
        CStr::from_bytes_until_nul(&self.0).map_or_else(
            |_| "no message".to_owned(),
            |text| text.to_string_lossy().into_owned(),
        )
    }
}

/// Turns a shim status into a result: [`Error::Graph`] for a graph of the
/// wrong shape, [`Error::Cuvs`] naming the step for any other failure.
fn check(status: c_int, message: &Message) -> Result<(), Error> {
    let call = match status {
        ffi::OK => return Ok(()),
        ffi::GRAPH => {
            return Err(Error::Graph {
                detail: message.text(),
            });
        }
        ffi::RESOURCES => "raft::resources",
        ffi::DATASET => "make_device_padded_dataset",
        ffi::BUILD => "cagra::build",
        ffi::COPY => "raft::copy",
        ffi::ARGUMENTS => "knn_cagra_build",
        _ => "the cuVS shim",
    };
    Err(Error::Cuvs {
        call,
        message: message.text(),
    })
}

/// The shim's `knn_cagra_gpu`.
#[derive(Debug)]
struct Resources(NonNull<ffi::KnnCagraGpu>);

impl Resources {
    fn new() -> Result<Self, Error> {
        let mut handle = ptr::null_mut();
        let mut message = Message::new();
        // SAFETY: `handle` is a valid out-pointer and `message` a buffer of
        // its length.
        let status = unsafe {
            ffi::knn_cagra_gpu_create(&raw mut handle, message.as_mut_ptr(), message.len())
        };
        check(status, &message)?;
        NonNull::new(handle).map(Self).ok_or_else(|| Error::Cuvs {
            call: "raft::resources",
            message: "no handle".to_owned(),
        })
    }
}

impl Drop for Resources {
    fn drop(&mut self) {
        // SAFETY: the handle came from knn_cagra_gpu_create and is destroyed
        // only here.
        unsafe { ffi::knn_cagra_gpu_destroy(self.0.as_ptr()) };
    }
}

// SAFETY: raft's resources are not tied to the thread that created them,
// and every use goes through `&mut Gpu`, so one thread at a time; the
// current-device caveat is on `Gpu`.
unsafe impl Send for Resources {}

#[cfg(test)]
mod tests {
    use super::{Graph, graph_buffer};
    use crate::Error;

    #[test]
    fn refuses_a_graph_too_large_for_memory() {
        // The first overflows usize; the second needs more than isize::MAX
        // bytes. Both are refused without an allocation being attempted.
        assert_eq!(
            graph_buffer(usize::MAX, 2),
            Err(Error::HostMemory {
                rows: usize::MAX,
                degree: 2
            })
        );
        assert_eq!(
            graph_buffer(usize::MAX / 4, 1),
            Err(Error::HostMemory {
                rows: usize::MAX / 4,
                degree: 1
            })
        );
        let (buffer, len) = graph_buffer(3, 4).unwrap();
        assert_eq!((buffer.len(), len), (0, 12));
        assert!(buffer.capacity() >= 12);
    }

    #[test]
    fn counts_ids_that_name_no_vector() {
        // Two vectors of degree 3: ids 2, 7 and u32::MAX name none.
        let graph = Graph {
            neighbors: vec![1, 2, 0, 0, 7, u32::MAX],
            degree: 3,
        };
        assert_eq!(graph.out_of_range(), 3);
        assert_eq!(format!("{graph:?}"), "Graph { rows: Some(2), degree: 3 }");
    }
}
