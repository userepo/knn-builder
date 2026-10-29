//! End to end on the GPU against S3 (LocalStack): uploads inputs written
//! by `tools/make_inputs.py`, builds them through the pipeline with
//! knn-gpu, and fetches the index files for `tools/baseline/score.py`.
//! `build` reports each job's stage times, its index file's size, the
//! process's peak resident memory (`VmHWM`), and the device memory the job
//! took at its peak: the free memory before it less the least seen free
//! while it ran, read every 5 ms by default. `put` and `get` run as
//! processes of their own, so their buffers stay out of the resident peak.
//! Needs the `gpu` feature, and S3 settings as the service reads them
//! (`S3_ENDPOINT_URL`, credentials).
//!
//! ```text
//! e2e put   BUCKET INPUTS NAME...   upload NAME/vectors.knnvec, NAME/doc_ids.knndid
//! e2e build BUCKET INPUTS OUT [--m M] [--leave BYTES] [--every MICROSECONDS] NAME...
//!                                   build each and append to OUT/runs.json: with m = M in place
//!                                   of the input's, with device memory held so that only BYTES
//!                                   are free when the job starts, and reading the free memory
//!                                   every MICROSECONDS (0: as often as it can)
//! e2e get   BUCKET OUT              fetch each built file to OUT/NAME/run-1.faiss
//! e2e drop  BUCKET NAME...          delete every object under NAME/
//! ```
//!
//! Builds of one input with several m write one file name, each over the
//! last, so `get` refuses a `runs.json` that holds more than one completed
//! build of an input: build each m into an OUT of its own to fetch them.
//!
//! Under WSL2 the driver pages device memory out to the host when a build
//! needs more than is free, so memory held with `--leave` lowers what the
//! device reports free but does not make a build fail; on a native Linux
//! driver a build that needs more than is left fails out of memory. The
//! pipeline's own check of the free memory before each build is off here:
//! a measuring tool builds what it is asked, to see what the build takes.

#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    reason = "a measuring tool: failing loudly is the point"
)]

use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use knn_builder::{BuildError, Builder, DataType, Job, Pipeline, SpaceType, Vectors};
use knn_gpu::{Gpu, Reservation, device_memory};
use knn_store::{Cancel, Settings, Store, Transfers};
use serde_json::{Value, json};

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let store = Store::new(&Settings::from_env(), Transfers::default()).await;
    match args.first().map(String::as_str) {
        Some("put") => put(&store, &args[1], Path::new(&args[2]), &args[3..]).await,
        Some("build") => {
            build(
                &store,
                &args[1],
                Path::new(&args[2]),
                Path::new(&args[3]),
                &args[4..],
            )
            .await;
        }
        Some("get") => get(&store, &args[1], Path::new(&args[2])).await,
        Some("drop") => drop_inputs(&store, &args[1], &args[2..]).await,
        _ => panic!("usage: e2e put|build|get ... (see the file's docs)"),
    }
}

/// Creates the bucket if need be, and uploads each input's two blobs.
async fn put(store: &Store, bucket: &str, inputs: &Path, names: &[String]) {
    let client = store.sdk_client();
    if client.head_bucket().bucket(bucket).send().await.is_err() {
        client.create_bucket().bucket(bucket).send().await.unwrap();
    }
    for name in names {
        for blob in ["vectors.knnvec", "doc_ids.knndid"] {
            let path = inputs.join(name).join(blob);
            let size = fs::metadata(&path).unwrap().len();
            let mut upload = store.upload(
                bucket,
                &format!("{name}/{blob}"),
                size,
                None,
                &Cancel::new(),
            );
            tokio::task::spawn_blocking(move || {
                let mut file = io::BufReader::with_capacity(1 << 23, File::open(path).unwrap());
                io::copy(&mut file, &mut upload).unwrap();
                upload.finish().unwrap();
            })
            .await
            .unwrap();
            println!("put {name}/{blob}: {size} bytes");
        }
    }
}

/// The job `tools/baseline/run_baseline.py` would request for `name`,
/// with `m` in place of the input's when given.
fn job(bucket: &str, name: &str, meta: &Value, m: Option<u32>) -> Job {
    let number = |key: &str| meta[key].as_u64().unwrap();
    Job {
        bucket: bucket.to_owned(),
        vector_path: format!("{name}/vectors.knnvec"),
        doc_id_path: format!("{name}/doc_ids.knndid"),
        dimension: u32::try_from(number("dimension")).unwrap(),
        doc_count: u32::try_from(number("doc_count")).unwrap(),
        data_type: match meta["data_type"].as_str().unwrap() {
            "float" => DataType::Float,
            "half_float" => DataType::HalfFloat,
            other => panic!("data_type {other}"),
        },
        space_type: match meta["space_type"].as_str().unwrap() {
            "l2" => SpaceType::L2,
            "innerproduct" => SpaceType::InnerProduct,
            other => panic!("space_type {other}"),
        },
        m: m.unwrap_or_else(|| u32::try_from(number("m")).unwrap()),
        ef_construction: i32::try_from(number("ef_construction")).unwrap(),
        ef_search: i32::try_from(number("ef_search")).unwrap(),
        skip_stored_vectors: meta["skip_stored_vectors"].as_bool().unwrap(),
    }
}

/// The process's peak resident memory since the last
/// [`reset_peak_rss`], or since it started, in bytes.
fn peak_rss() -> u64 {
    let status = fs::read_to_string("/proc/self/status").unwrap();
    let line = status.lines().find(|l| l.starts_with("VmHWM:")).unwrap();
    let kib: u64 = line.split_whitespace().nth(1).unwrap().parse().unwrap();
    kib * 1024
}

/// Restarts the peak from the current resident memory (Linux 4.0 and
/// later), so the next reading is one job's own.
fn reset_peak_rss() {
    fs::write("/proc/self/clear_refs", "5").unwrap();
}

/// The GPU, telling the pipeline nothing of its free memory, so no build
/// is failed before it starts.
struct Unchecked(Gpu);

impl Builder for Unchecked {
    fn build(
        &mut self,
        vectors: &Vectors,
        dimension: usize,
        params: &knn_gpu::BuildParams,
    ) -> Result<Vec<u32>, BuildError> {
        Builder::build(&mut self.0, vectors, dimension, params)
    }
}

/// Reads the device's free memory every `every` until `stop` is set, and
/// returns the least it saw.
fn least_free(stop: Arc<AtomicBool>, every: Duration) -> thread::JoinHandle<usize> {
    thread::spawn(move || {
        let mut least = usize::MAX;
        loop {
            least = least.min(device_memory().unwrap().free);
            if stop.load(Ordering::Relaxed) {
                return least;
            }
            if !every.is_zero() {
                thread::sleep(every);
            }
        }
    })
}

/// `build`'s options.
struct BuildFlags {
    /// `--m`: m in place of each input's.
    m: Option<u32>,
    /// `--leave`: the device memory to leave free for each job.
    leave: Option<usize>,
    /// `--every`: how often to read the free memory.
    every: Duration,
}

/// `build`'s options, in any order before the names, and the names.
fn build_flags(mut args: &[String]) -> (BuildFlags, &[String]) {
    let mut flags = BuildFlags {
        m: None,
        leave: None,
        every: Duration::from_millis(5),
    };
    loop {
        match args {
            [flag, value, ..] if flag == "--m" => flags.m = Some(value.parse().unwrap()),
            [flag, value, ..] if flag == "--leave" => flags.leave = Some(value.parse().unwrap()),
            [flag, value, ..] if flag == "--every" => {
                flags.every = Duration::from_micros(value.parse().unwrap());
            }
            _ => return (flags, args),
        }
        args = &args[2..];
    }
}

/// Builds each input, one job after another, and appends a record per job
/// to `out/runs.json` in the shape `score.py` reads. A record is kept per
/// input and m: a later run of both replaces it, as it replaces a record
/// written before records carried their m.
async fn build(store: &Store, bucket: &str, inputs: &Path, out: &Path, args: &[String]) {
    let (flags, names) = build_flags(args);
    let gpu = Unchecked(Gpu::new().unwrap());
    let after_init = peak_rss();
    // One job at a time, so no budget is needed.
    let pipeline = Pipeline::start(store.clone(), gpu, u64::MAX);
    let runs_path = out.join("runs.json");
    let mut runs: Vec<Value> = fs::read(&runs_path)
        .ok()
        .map(|bytes| serde_json::from_slice(&bytes).unwrap())
        .unwrap_or_default();
    for name in names {
        let meta: Value =
            serde_json::from_slice(&fs::read(inputs.join(name).join("input.json")).unwrap())
                .unwrap();
        let job = job(bucket, name, &meta, flags.m);
        let shape = json!({
            "doc_count": job.doc_count,
            "dimension": job.dimension,
            "data_type": meta["data_type"],
            "blob_dtype": meta["blob_dtype"],
            "m": job.m,
        });
        // What another process on the GPU would take, for this job.
        let held = flags.leave.map(|leave| {
            let free = device_memory().unwrap().free;
            Reservation::new(free.saturating_sub(leave)).unwrap()
        });
        let before = device_memory().unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let sampler = least_free(Arc::clone(&stop), flags.every);
        reset_peak_rss();
        let start = Instant::now();
        let outcome = pipeline.submit(job).outcome().await;
        let wall = start.elapsed().as_secs_f64();
        stop.store(true, Ordering::Relaxed);
        let least = sampler.join().unwrap();
        let mut device = json!({
            "total_bytes": before.total,
            "free_before_bytes": before.free,
            "least_free_bytes": least,
            "peak_bytes": before.free.saturating_sub(least),
            "read_every_us": flags.every.as_micros(),
        });
        if let Some(held) = held {
            device["held_bytes"] = json!(held.bytes());
        }
        let mut record = match outcome {
            Ok(built) => {
                let t = built.timings;
                let size = store
                    .sdk_client()
                    .head_object()
                    .bucket(bucket)
                    .key(format!("{name}/{}", built.file_name))
                    .send()
                    .await
                    .unwrap()
                    .content_length()
                    .unwrap();
                json!({
                    "input": name,
                    "run": 1,
                    "task_status": "COMPLETED_INDEX_BUILD",
                    "file_name": built.file_name,
                    "file_size": size,
                    "wall_seconds": round(wall),
                    "seconds": {
                        "download": round(t.download.duration().as_secs_f64()),
                        "build": round(t.build.duration().as_secs_f64()),
                        "upload": round(t.upload.duration().as_secs_f64()),
                    },
                    "peak_rss_bytes": peak_rss(),
                    "peak_rss_after_gpu_init_bytes": after_init,
                })
            }
            Err(error) => json!({
                "input": name,
                "run": 1,
                "task_status": "FAILED_INDEX_BUILD",
                "error_message": error.to_string(),
                "wall_seconds": round(wall),
            }),
        };
        record["shape"] = shape;
        record["device"] = device;
        println!("{record}");
        // A record without a shape predates m in the record: one per input.
        runs.retain(|r| {
            r["input"] != record["input"]
                || (r.get("shape").is_some() && r["shape"]["m"] != record["shape"]["m"])
        });
        runs.push(record);
    }
    pipeline.close().await;
    fs::create_dir_all(out).unwrap();
    let mut file = File::create(&runs_path).unwrap();
    file.write_all(serde_json::to_string_pretty(&runs).unwrap().as_bytes())
        .unwrap();
    file.write_all(b"\n").unwrap();
}

fn round(seconds: f64) -> f64 {
    (seconds * 1000.0).round() / 1000.0
}

/// Fetches each completed run's index file next to `runs.json`.
async fn get(store: &Store, bucket: &str, out: &Path) {
    let runs_path = out.join("runs.json");
    let mut runs: Vec<Value> = serde_json::from_slice(&fs::read(&runs_path).unwrap()).unwrap();
    let completed = |r: &&Value| r["task_status"] == "COMPLETED_INDEX_BUILD";
    let mut seen = HashSet::new();
    for record in runs.iter().filter(completed) {
        let name = record["input"].as_str().unwrap();
        assert!(
            seen.insert(name),
            "{name} was built more than once (with several m): the builds share one file; \
             build each m into an OUT of its own to fetch it"
        );
    }
    for record in &mut runs {
        if record["task_status"] != "COMPLETED_INDEX_BUILD" {
            continue;
        }
        let name = record["input"].as_str().unwrap().to_owned();
        let key = format!("{name}/{}", record["file_name"].as_str().unwrap());
        let local: PathBuf = out.join(&name).join("run-1.faiss");
        fs::create_dir_all(local.parent().unwrap()).unwrap();
        // Streamed to the file a frame at a time: an index file can be
        // larger than the memory there is to hold it.
        let mut body = store
            .sdk_client()
            .get_object()
            .bucket(bucket)
            .key(&key)
            .send()
            .await
            .unwrap()
            .body;
        let mut file = io::BufWriter::with_capacity(1 << 23, File::create(&local).unwrap());
        let mut size = 0_u64;
        while let Some(frame) = body.try_next().await.unwrap() {
            file.write_all(&frame).unwrap();
            size += frame.len() as u64;
        }
        file.flush().unwrap();
        record["size"] = json!(size);
        println!("got {name}: {size} bytes");
    }
    fs::write(
        &runs_path,
        serde_json::to_string_pretty(&runs).unwrap() + "\n",
    )
    .unwrap();
}

/// Deletes every object under each `NAME/`: an input's blobs and the
/// files built from it.
async fn drop_inputs(store: &Store, bucket: &str, names: &[String]) {
    let client = store.sdk_client();
    for name in names {
        let listed = client
            .list_objects_v2()
            .bucket(bucket)
            .prefix(format!("{name}/"))
            .send()
            .await
            .unwrap();
        for object in listed.contents() {
            let key = object.key().unwrap();
            client
                .delete_object()
                .bucket(bucket)
                .key(key)
                .send()
                .await
                .unwrap();
            println!("dropped {key}");
        }
    }
}
