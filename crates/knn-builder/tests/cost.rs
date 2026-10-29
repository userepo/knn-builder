//! The cost model against what it was fitted from: every build of the grid
//! (`fixtures/grid/runs.json`) costs what `tools/grid/fit.py`'s models in
//! `fixtures/grid/fits.json` say, so the constants are the fit's, and a
//! refit that changes them fails here until the model follows.

#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    reason = "test crate"
)]

use std::fs;
use std::path::Path;

use knn_builder::cost::{CpuThreads, Fitted};
use knn_builder::{DataType, Job, SpaceType};
use serde_json::Value;

/// The RTX 4070 Laptop GPU's shared memory a block, which the grid ran on.
const RTX_4070: Fitted = Fitted {
    shared_memory_per_block: 101_376,
};

fn fixture(name: &str) -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/grid")
        .join(name);
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}

/// The grid's completed builds, the warm-up ones left out as fit.py
/// leaves them.
fn builds() -> Vec<Value> {
    let Value::Array(runs) = fixture("runs.json") else {
        panic!("runs.json is a list")
    };
    runs.into_iter()
        .filter(|r| r["task_status"] == "COMPLETED_INDEX_BUILD")
        .filter(|r| r["shape"]["doc_count"].as_u64().unwrap() >= 100_000)
        .collect()
}

fn job(record: &Value) -> Job {
    let shape = &record["shape"];
    let number = |key: &str| u32::try_from(shape[key].as_u64().unwrap()).unwrap();
    Job {
        bucket: "grid".into(),
        vector_path: "a/vectors.knnvec".into(),
        doc_id_path: "a/doc_ids.knndid".into(),
        dimension: number("dimension"),
        doc_count: number("doc_count"),
        data_type: match shape["data_type"].as_str().unwrap() {
            "float" => DataType::Float,
            "half_float" => DataType::HalfFloat,
            other => panic!("{other}"),
        },
        space_type: SpaceType::L2,
        m: number("m"),
        ef_construction: 100,
        ef_search: 100,
        skip_stored_vectors: false,
    }
}

/// A model of fits.json, evaluated on `terms` as fit.py names them.
fn evaluate(model: &Value, terms: &[(&str, f64)]) -> f64 {
    let model = model.as_object().unwrap();
    for name in model.keys() {
        assert!(
            terms.iter().any(|(t, _)| t == name),
            "fit.py's term {name} is not modeled"
        );
    }
    terms
        .iter()
        .map(|(name, value)| model.get(*name).map_or(0.0, |c| c.as_f64().unwrap()) * value)
        .sum()
}

fn close(rust: f64, fit: f64, what: &str, record: &Value) {
    assert!(
        (rust - fit).abs() <= 1e-9 * fit.abs().max(1.0),
        "{what} of {}: {rust} here, {fit} in fits.json",
        record["shape"]
    );
}

#[test]
fn costs_what_the_fit_says() {
    let fits = fixture("fits.json");
    let builds = builds();
    assert_eq!(builds.len(), 58);
    for record in &builds {
        let job = job(record);
        let shape = &record["shape"];
        let n = shape["doc_count"].as_f64().unwrap();
        let d = shape["dimension"].as_f64().unwrap();
        let fp32 = shape["blob_dtype"] == "<f4";
        let s = if fp32 { 4.0 } else { 2.0 };
        let g = 4.0 * shape["m"].as_f64().unwrap();
        // fit.py's gate: the measured case.
        let slow = if job.dimension == 768 && job.m == 32 {
            n.powf(1.5) * d
        } else {
            0.0
        };
        let stages = RTX_4070.stages(&job);
        let build = evaluate(
            &fits["build_seconds"]["terms"],
            &[
                ("1", 1.0),
                ("n d s", n * d * s),
                ("n^1.5 d", n.powf(1.5) * d),
                ("n g^2", n * g * g),
                ("n^1.5 d, slow search", slow),
            ],
        );
        close(stages.build, build, "build seconds", record);
        let upload = evaluate(
            &fits["upload_seconds"]["terms"],
            &[("1", 1.0), ("n d s", n * d * s), ("n g", n * g), ("n", n)],
        );
        close(stages.upload, upload, "upload seconds", record);
        // The model counts a half_float blob as fp32 until its size is
        // read; the grid's fp16 blobs were fp16.
        if fp32 {
            let download = evaluate(
                &fits["download_seconds"]["terms"],
                &[("1", 1.0), ("n d s", n * d * s), ("n", n)],
            );
            close(stages.download, download, "download seconds", record);
        }
        let file = evaluate(
            &fits["file_bytes"]["terms"],
            &[("1", 1.0), ("n d s", n * d * s), ("n g", n * g), ("n", n)],
        );
        close(Fitted::file_bytes(&job), file, "file bytes", record);
        let device = &fits["device_peak_bytes"];
        let rows = device["rows"].as_f64().unwrap();
        let ivf_pq = n * d / 4.0
            + evaluate(
                &device["ivf_pq_phase"],
                &[("1", 1.0), ("d", d), ("n d", n * d)],
            );
        let graph = 4.0 * n * g
            + evaluate(
                &device["graph_phase"],
                &[
                    ("1", 1.0),
                    ("n g", n * g),
                    ("min(n, rows) g", n.min(rows) * g),
                ],
            );
        close(
            Fitted::device_peak_bytes(&job),
            n * d * s + ivf_pq.max(graph),
            "device peak",
            record,
        );
    }
}

#[test]
fn finds_the_slow_search_where_the_grid_did() {
    for record in builds() {
        let job = job(&record);
        let measured = job.dimension == 768 && job.m == 32;
        assert_eq!(RTX_4070.slow_search(&job), measured, "{}", record["shape"]);
    }
    // A GPU whose blocks can take d 768's 196,608-byte lookup table keeps
    // the fast search.
    let mut job = job(&builds()[0]);
    (job.dimension, job.m) = (768, 32);
    let larger = Fitted {
        shared_memory_per_block: 232_448,
    };
    assert!(!larger.slow_search(&job));
}

#[test]
fn never_forecasts_less_device_memory_than_the_dataset_codes_and_graph() {
    for (doc_count, dimension, m) in [(5_000_000, 128, 32), (10_000_000, 768, 16)] {
        let job = Job {
            doc_count,
            dimension,
            m,
            ..job(&builds()[0])
        };
        let (n, d, g) = (f64::from(doc_count), f64::from(dimension), f64::from(4 * m));
        let floor = n * d * 4.0 + (n * d / 4.0).max(4.0 * n * g);
        assert!(Fitted::device_peak_bytes(&job) >= floor);
    }
}

#[test]
fn takes_the_cpu_build_the_fit_says() {
    let fits = fixture("fits.json");
    let Value::Array(builds) = fixture("cpu.json") else {
        panic!("cpu.json is a list")
    };
    let template = job(&self::builds()[0]);
    for record in builds {
        let shape = &record["shape"];
        let n = shape["doc_count"].as_f64().unwrap();
        let d = shape["dimension"].as_f64().unwrap();
        let job = Job {
            doc_count: u32::try_from(shape["doc_count"].as_u64().unwrap()).unwrap(),
            dimension: u32::try_from(shape["dimension"].as_u64().unwrap()).unwrap(),
            ..template.clone()
        };
        let (threads, key) = match record["threads"].as_u64().unwrap() {
            1 => (CpuThreads::One, "cpu_seconds_1_threads"),
            4 => (CpuThreads::Four, "cpu_seconds_4_threads"),
            other => panic!("{other} threads"),
        };
        let fit = evaluate(
            &fits[key]["terms"],
            &[
                ("1", 1.0),
                ("n log2(n)", n * n.log2()),
                ("n log2(n) d", n * n.log2() * d),
            ],
        );
        close(Fitted::cpu_seconds(&job, threads), fit, key, &record);
    }
}
