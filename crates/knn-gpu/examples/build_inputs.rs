//! Builds every input under `INPUTS_DIR` `RUNS` times as the upstream
//! service would, with knn-gpu and knn-write, for the parity check against
//! the service's baseline (`tools/baseline/score.py`).
//!
//! Usage: `build_inputs INPUTS_DIR OUT_DIR RUNS`. Writes
//! `OUT_DIR/<input>/run-<k>.faiss` and `OUT_DIR/runs.json`, in the
//! baseline runner's format plus each build's own timings and its count of
//! out-of-range neighbor ids.

// An example: a bad input is reported by panicking.
#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    reason = "example"
)]

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use knn_gpu::{BuildParams, Gpu, HostVectors, Metric, fp16};
use knn_write::{IndexFile, IndexSpec, Vectors};
use serde_json::{Value, json};

struct Input {
    name: String,
    meta: Value,
    f32s: Vec<f32>,
    f16s: Vec<u16>,
    ids: Vec<i32>,
}

impl Input {
    fn read(dir: &Path) -> Self {
        let meta: Value =
            serde_json::from_slice(&fs::read(dir.join("input.json")).unwrap()).unwrap();
        let blob = fs::read(dir.join("vectors.knnvec")).unwrap();
        let (mut f32s, mut f16s) = (Vec::new(), Vec::new());
        match (
            meta["data_type"].as_str().unwrap(),
            meta["blob_dtype"].as_str().unwrap(),
        ) {
            ("float", "<f4") => f32s = le(&blob, f32::from_le_bytes),
            ("half_float", "<f2") => f16s = le(&blob, u16::from_le_bytes),
            ("half_float", "<f4") => {
                // As the service converts on download.
                let values = le(&blob, f32::from_le_bytes);
                f16s = vec![0; values.len()];
                fp16::convert(&values, &mut f16s);
            }
            other => panic!("unsupported input {other:?}"),
        }
        Self {
            name: dir.file_name().unwrap().to_string_lossy().into_owned(),
            ids: le(
                &fs::read(dir.join("doc_ids.knndid")).unwrap(),
                i32::from_le_bytes,
            ),
            meta,
            f32s,
            f16s,
        }
    }

    fn number(&self, key: &str) -> u32 {
        u32::try_from(self.meta[key].as_u64().unwrap()).unwrap()
    }
}

fn le<const N: usize, T>(bytes: &[u8], from: fn([u8; N]) -> T) -> Vec<T> {
    let (chunks, rest) = bytes.as_chunks::<N>();
    assert!(rest.is_empty(), "a blob of {} bytes", bytes.len());
    chunks.iter().copied().map(from).collect()
}

fn build(gpu: &mut Gpu, input: &Input, out: &Path, run: usize) -> Value {
    let dimension = input.number("dimension");
    let metric = match input.meta["space_type"].as_str().unwrap() {
        "l2" => Metric::L2,
        "innerproduct" => Metric::InnerProduct,
        other => panic!("space type {other}"),
    };
    let params = BuildParams::new(
        input.number("doc_count"),
        dimension,
        input.number("m"),
        metric,
    )
    .unwrap();
    let host = if input.f16s.is_empty() {
        HostVectors::F32(&input.f32s)
    } else {
        HostVectors::F16(&input.f16s)
    };

    let started = Instant::now();
    let graph = match gpu.build(host, dimension as usize, &params) {
        Ok(graph) => graph,
        Err(error) => {
            return json!({"input": input.name, "run": run, "task_status": "FAILED_INDEX_BUILD",
                          "error_message": error.to_string()});
        }
    };
    let build_seconds = started.elapsed().as_secs_f64();

    let started = Instant::now();
    let file = IndexFile::new(IndexSpec {
        dimension: dimension as usize,
        metric: match metric {
            Metric::L2 => knn_write::Metric::L2,
            Metric::InnerProduct => knn_write::Metric::InnerProduct,
        },
        graph: graph.neighbors(),
        graph_degree: graph.degree(),
        vectors: if input.f16s.is_empty() {
            Vectors::F32(&input.f32s)
        } else {
            Vectors::F16(&input.f16s)
        },
        ids: &input.ids,
        ef_construction: i32::try_from(input.number("ef_construction")).unwrap(),
        ef_search: i32::try_from(input.number("ef_search")).unwrap(),
        skip_stored_vectors: input.meta["skip_stored_vectors"].as_bool().unwrap(),
    })
    .unwrap();
    let path = out.join(&input.name).join(format!("run-{run}.faiss"));
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    let mut bytes = Vec::new();
    file.write_to(&mut bytes).unwrap();
    fs::write(&path, &bytes).unwrap();
    let write_seconds = started.elapsed().as_secs_f64();

    json!({
        "input": input.name,
        "run": run,
        "task_status": "COMPLETED_INDEX_BUILD",
        "size": bytes.len(),
        "seconds": {"build": round(build_seconds), "write": round(write_seconds)},
        "graph_degree": graph.degree(),
        "out_of_range": graph.out_of_range(),
    })
}

fn round(seconds: f64) -> f64 {
    (seconds * 1000.0).round() / 1000.0
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (inputs, out, runs): (PathBuf, PathBuf, usize) = (
        args[1].clone().into(),
        args[2].clone().into(),
        args[3].parse().unwrap(),
    );
    let mut dirs: Vec<PathBuf> = fs::read_dir(&inputs)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|dir| dir.join("input.json").exists())
        .collect();
    dirs.sort();
    let inputs: Vec<Input> = dirs.iter().map(|dir| Input::read(dir)).collect();

    let mut gpu = Gpu::new().unwrap();
    let mut records = Vec::new();
    // Runs in the baseline's order: every input once per run.
    for run in 1..=runs {
        for input in &inputs {
            let record = build(&mut gpu, input, &out, run);
            println!("{record}");
            records.push(record);
        }
    }
    fs::create_dir_all(&out).unwrap();
    fs::write(
        out.join("runs.json"),
        serde_json::to_string_pretty(&records).unwrap() + "\n",
    )
    .unwrap();
}
