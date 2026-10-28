//! Device memory after many builds, some failing inside cuVS: every build,
//! finished or not, must give back what it took. Needs the `gpu` feature,
//! a CUDA GPU and the conda environment in `environment.yml`.

// The clippy.toml test allowances cover `#[test]` functions, not the
// helpers beside them; in a test crate panicking is the failure mechanism
// everywhere.
#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    reason = "test crate"
)]
#![expect(
    unsafe_code,
    reason = "reads the device's free memory through the CUDA runtime's C API"
)]

use std::ffi::c_int;

use knn_gpu::{BuildParams, Error, Gpu, HostVectors, Metric};

#[link(name = "cudart")]
unsafe extern "C" {
    fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> c_int;
}

fn free_device_memory() -> usize {
    let (mut free, mut total) = (0, 0);
    // SAFETY: two valid out-pointers.
    let status = unsafe { cudaMemGetInfo(&raw mut free, &raw mut total) };
    assert_eq!(status, 0, "cudaMemGetInfo failed");
    free
}

fn random_vectors(count: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            #[expect(clippy::cast_precision_loss, reason = "a 24-bit value is exact in f32")]
            let value = (state >> 40) as f32;
            value / 16_777_216.0 - 0.5
        })
        .collect()
}

const ROWS: usize = 20_000;
const DIM: usize = 64;
const MIB: usize = 1 << 20;

/// Parameters cuVS refuses only after the dataset is on the device: more
/// IVF-PQ lists than vectors. The test checks the refusal comes from the
/// build step, so the failure path it measures frees an uploaded dataset.
fn failing(params: &BuildParams) -> BuildParams {
    BuildParams {
        n_lists: u32::try_from(ROWS).unwrap() + 1,
        ..*params
    }
}

#[test]
fn builds_return_their_device_memory() {
    let values = random_vectors(ROWS * DIM, 7);
    let params = BuildParams::new(20_000, 64, 16, Metric::L2).unwrap();
    let mut gpu = Gpu::new().unwrap();

    // The first build loads kernels and creates the library handles cuVS
    // keeps; measure after it, and after one failure.
    gpu.build(HostVectors::F32(&values), DIM, &params).unwrap();
    let failed = gpu.build(HostVectors::F32(&values), DIM, &failing(&params));
    assert!(
        matches!(
            failed,
            Err(Error::Cuvs {
                call: "cagra::build",
                ..
            })
        ),
        "{failed:?}"
    );
    let baseline = free_device_memory();

    let (mut built, mut refused) = (0, 0);
    for job in 0..100 {
        if job % 4 == 3 {
            let result = gpu.build(HostVectors::F32(&values), DIM, &failing(&params));
            assert!(
                matches!(
                    result,
                    Err(Error::Cuvs {
                        call: "cagra::build",
                        ..
                    })
                ),
                "job {job}: {result:?}"
            );
            refused += 1;
        } else {
            gpu.build(HostVectors::F32(&values), DIM, &params).unwrap();
            built += 1;
        }
    }
    let after = free_device_memory();
    println!(
        "{built} built, {refused} refused; free device memory {} MiB before, {} MiB after",
        baseline / MIB,
        after / MIB
    );
    // The dataset alone is 5 MiB, so a leak per build would lose
    // hundreds of MiB over 100 builds; the margin absorbs other
    // processes sharing the GPU (a laptop's display among them).
    assert!(
        after + 64 * MIB >= baseline,
        "{} MiB fewer free after 100 builds",
        (baseline - after) / MIB
    );
}
