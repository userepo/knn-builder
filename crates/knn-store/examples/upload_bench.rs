//! Times uploads through [`Store::upload`] with a writer that does no
//! work, over a grid of part sizes and requests in flight.
//!
//! The writer copies a block it holds until the size is reached, so the
//! time is the transfer's: what the requests can carry, without an index
//! to serialize. S3 settings as the service reads them (`S3_ENDPOINT_URL`,
//! `AWS_DEFAULT_REGION`, credentials); the bucket must exist.
//!
//! ```text
//! upload_bench BUCKET KEY SIZE_MIB PART_MIB[,PART_MIB...] IN_FLIGHT[,IN_FLIGHT...] RUNS
//! ```
//!
//! Each upload prints one line of JSON: the part size and requests in
//! flight, the run, the seconds from the upload's start to the file
//! stored, the writer's waits within them, and the rate in MiB/s. Every
//! upload goes to `KEY`, which is deleted at the end. A part below S3's
//! 5 MiB is raised to it, and the line says what was used.

#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    reason = "a measuring tool: failing loudly is the point"
)]

use std::io::Write;
use std::sync::Arc;
use std::time::Instant;
use std::{env, process};

use knn_store::{Cancel, Settings, Store, Transfers};

const MIB: u64 = 1 << 20;
/// What the writer copies from, a few parts' worth at the smallest size.
const BLOCK: usize = 16 << 20;

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let [bucket, key, size, parts, in_flight, runs] = args.as_slice() else {
        eprintln!("usage: upload_bench BUCKET KEY SIZE_MIB PART_MIB[,...] IN_FLIGHT[,...] RUNS");
        process::exit(2);
    };
    let size = number(size) * MIB;
    let runs = number(runs);
    let settings = Settings::from_env();
    // A pattern rather than one byte, should anything on the way shorten
    // runs of a byte.
    let block: Arc<Vec<u8>> = Arc::new((0..=250).cycle().take(BLOCK).collect());

    let mut last = None;
    for part in numbers(parts) {
        for flights in numbers(in_flight) {
            let transfers = Transfers {
                part_size: usize::try_from(part * MIB).unwrap(),
                upload_concurrency: usize::try_from(flights).unwrap(),
                ..Transfers::default()
            };
            // A store of its own, so each setting starts on new connections.
            let store = Store::new(&settings, transfers).await;
            let used = store.transfers();
            for run in 1..=runs {
                let start = Instant::now();
                let mut upload = store.upload(bucket, key, size, None, &Cancel::new());
                let block = Arc::clone(&block);
                let (waits, took) = tokio::task::spawn_blocking(move || {
                    let mut left = size;
                    while left > 0 {
                        let take = usize::try_from(left).map_or(BLOCK, |left| left.min(BLOCK));
                        upload.write_all(&block[..take]).unwrap();
                        left -= take as u64;
                    }
                    let waits = upload.finish().unwrap();
                    (waits, start.elapsed())
                })
                .await
                .unwrap();
                let seconds = took.as_secs_f64();
                #[expect(clippy::cast_precision_loss, reason = "sizes far below 2^53")]
                let rate = (size / MIB) as f64 / seconds;
                println!(
                    "{{\"client\": \"knn-store\", \"part_mib\": {}, \"in_flight\": {}, \
                     \"run\": {run}, \"seconds\": {seconds:.3}, \"part_wait\": {:.3}, \
                     \"finish_wait\": {:.3}, \"mib_per_s\": {rate:.0}}}",
                    used.part_size as u64 / MIB,
                    used.upload_concurrency,
                    waits.part.as_secs_f64(),
                    waits.finish.as_secs_f64(),
                );
            }
            last = Some(store);
        }
    }
    if let Some(store) = last {
        store
            .sdk_client()
            .delete_object()
            .bucket(bucket)
            .key(key)
            .send()
            .await
            .unwrap();
    }
}

/// A count from the command line.
fn number(text: &str) -> u64 {
    text.parse()
        .unwrap_or_else(|_| panic!("not a number: {text}"))
}

/// A comma-separated list of counts from the command line.
fn numbers(text: &str) -> Vec<u64> {
    text.split(',').map(number).collect()
}
