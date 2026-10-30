//! Device memory after many builds, some failing inside cuVS, and after
//! many binary ones: every build, finished or not, must give back what it
//! took; a reservation holds what
//! it took until dropped; and the device's shared memory a block reads as a
//! GPU's can. Needs the `gpu` feature, a CUDA GPU and the conda environment
//! in `environment.yml`.

// The clippy.toml test allowances cover `#[test]` functions, not the
// helpers beside them; in a test crate panicking is the failure mechanism
// everywhere.
#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    reason = "test crate"
)]

use std::sync::Mutex;

use knn_gpu::{
    BinaryParams, BuildParams, Error, Gpu, HostVectors, Metric, Reservation, device_memory,
    shared_memory_per_block,
};

/// Held by each test that builds or measures free device memory: cargo
/// runs the tests on parallel threads, two builds at once can crash the
/// process (see `Gpu`), and one test's allocations would skew another's
/// measurements.
static GPU: Mutex<()> = Mutex::new(());

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
    let _gpu = GPU.lock().expect("a GPU test failed holding the lock");
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
    let _gpu = GPU.lock().expect("a GPU test failed holding the lock");
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

#[test]
fn reads_the_shared_memory_a_block_may_take() {
    // Every GPU cuVS 26.08 runs on (compute capability 7.0 and later)
    // lets a block opt in to more than the default 48 KiB, and none yet
    // to 256 KiB.
    let bytes = shared_memory_per_block().unwrap();
    println!("{bytes} bytes of shared memory a block");
    assert!((48 << 10..256 << 10).contains(&bytes), "{bytes}");
}

#[test]
fn binary_builds_return_their_device_memory() {
    let _gpu = GPU.lock().expect("a GPU test failed holding the lock");
    // 20,000 vectors of 256 bits: random bytes from the float draw's bits.
    let bits: Vec<u8> = random_vectors(ROWS * 8, 11)
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let params = BinaryParams::new(20_000, 256, 16).unwrap();
    let mut gpu = Gpu::new().unwrap();
    gpu.build_binary(&bits, 256, &params).unwrap();
    let baseline = free_device_memory();
    for _ in 0..40 {
        gpu.build_binary(&bits, 256, &params).unwrap();
    }
    let after = free_device_memory();
    println!(
        "40 binary builds; free device memory {} MiB before, {} MiB after",
        baseline / MIB,
        after / MIB
    );
    // The graph alone is 5 MiB on the device, so a leak per build would
    // lose some 200 MiB over 40.
    assert!(
        after + 64 * MIB >= baseline,
        "{} MiB fewer free after 40 builds",
        (baseline - after) / MIB
    );
}
