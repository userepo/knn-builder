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

use knn_gpu::{BinaryParams, BuildParams, Gpu, Graph, HostVectors, Metric, fp16};
use knn_write::{IndexFile, IndexSpec, Vectors};
use serde_json::{Value, json};

struct Input {
    name: String,
    meta: Value,
    f32s: Vec<f32>,
    f16s: Vec<u16>,
    i8s: Vec<i8>,
    /// A binary input's packed bits.
    bits: Vec<u8>,
    ids: Vec<i32>,
}

impl Input {
    fn read(dir: &Path) -> Self {
        let meta: Value =
            serde_json::from_slice(&fs::read(dir.join("input.json")).unwrap()).unwrap();
        let blob = fs::read(dir.join("vectors.knnvec")).unwrap();
        let (mut f32s, mut f16s, mut i8s, mut bits) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
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
            ("byte", "<i1") => i8s = blob.iter().map(|&b| b.cast_signed()).collect(),
            ("binary", "<u1") => bits = blob,
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
            i8s,
            bits,
        }
    }

    /// A float, fp16 or byte input's vectors as the build takes them, and
    /// as the writer stores them.
    fn vectors(&self) -> (HostVectors<'_>, Vectors<'_>) {
        if !self.i8s.is_empty() {
            (HostVectors::I8(&self.i8s), Vectors::I8(&self.i8s))
        } else if !self.f16s.is_empty() {
            (HostVectors::F16(&self.f16s), Vectors::F16(&self.f16s))
        } else {
            (HostVectors::F32(&self.f32s), Vectors::F32(&self.f32s))
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

/// The input's graph, the metric its file records, and its vectors as the
/// file stores them.
fn build_graph<'a>(
    gpu: &mut Gpu,
    input: &'a Input,
) -> (
    Result<Graph, knn_gpu::Error>,
    knn_write::Metric,
    Vectors<'a>,
) {
    let (doc_count, dimension, m) = (
        input.number("doc_count"),
        input.number("dimension"),
        input.number("m"),
    );
    if !input.bits.is_empty() {
        let params = BinaryParams::new(doc_count, dimension, m).unwrap();
        let graph = gpu.build_binary(&input.bits, dimension as usize, &params);
        return (
            graph,
            knn_write::Metric::Hamming,
            Vectors::Bits(&input.bits),
        );
    }
    let (metric, file_metric) = match input.meta["space_type"].as_str().unwrap() {
        "l2" => (Metric::L2, knn_write::Metric::L2),
        "innerproduct" => (Metric::InnerProduct, knn_write::Metric::InnerProduct),
        other => panic!("space type {other}"),
    };
    let params = BuildParams::new(doc_count, dimension, m, metric).unwrap();
    let (host, vectors) = input.vectors();
    (
        gpu.build(host, dimension as usize, &params),
        file_metric,
        vectors,
    )
}

fn build(gpu: &mut Gpu, input: &Input, out: &Path, run: usize) -> Value {
    let dimension = input.number("dimension");
    let started = Instant::now();
    let (graph, metric, vectors) = build_graph(gpu, input);
    let graph = match graph {
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
        metric,
        graph: graph.neighbors(),
        graph_degree: graph.degree(),
        vectors,
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
