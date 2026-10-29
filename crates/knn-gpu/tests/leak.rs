//! Device memory after many builds, some failing inside cuVS: every build,
//! finished or not, must give back what it took; and a reservation holds
//! what it took until dropped. Needs the `gpu` feature, a CUDA GPU and the
//! conda environment in `environment.yml`.

// The clippy.toml test allowances cover `#[test]` functions, not the
// helpers beside them; in a test crate panicking is the failure mechanism
// everywhere.
#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    reason = "test crate"
)]

use knn_gpu::{BuildParams, Error, Gpu, HostVectors, Metric, Reservation, device_memory};

fn free_device_memory() -> usize {
    device_memory().unwrap().free
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

#[test]
fn a_reservation_holds_its_memory_until_dropped() {
    let before = free_device_memory();
    let reservation = Reservation::new(256 * MIB).unwrap();
    assert_eq!(reservation.bytes(), 256 * MIB);
    let held = free_device_memory();
    drop(reservation);
    let after = free_device_memory();
    println!(
        "free device memory {} MiB before, {} MiB held, {} MiB after",
        before / MIB,
        held / MIB,
        after / MIB
    );
    // The first allocation may also create the context's own state; the
    // margin covers that and other processes sharing the GPU.
    assert!(
        held + 256 * MIB <= before + 16 * MIB,
        "only {} MiB taken",
        before.saturating_sub(held) / MIB
    );
    assert!(
        after + 16 * MIB >= before,
        "{} MiB not returned",
        before.saturating_sub(after) / MIB
    );
    let too_much = Reservation::new(usize::MAX / 2);
    assert!(
        matches!(
            too_much,
            Err(Error::Cuvs {
                call: "cudaMalloc",
                ..
            })
        ),
        "{too_much:?}"
    );
    assert_eq!(Reservation::new(0).unwrap().bytes(), 0);
}
