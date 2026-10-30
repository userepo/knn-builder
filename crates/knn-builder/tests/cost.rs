//! The cost model against what it was fitted from: every build of the grid
//! (`fixtures/grid/runs.json`) costs what `tools/grid/fit.py`'s models in
//! `fixtures/grid/fits.json` say, so the constants are the fit's, and a
//! refit that changes them fails here until the model follows. And against
//! what it was not fitted to, byte builds (`fixtures/grid/byte-runs.json`,
//! `byte-cpu.json`), predicted within the errors of the builds it was. The
//! binary models are held to `binary-fits.json` on every binary build alike
//! (`binary-runs.json`, `binary-cpu.json`).

#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    reason = "test crate"
)]

use std::fs;
use std::path::Path;

use knn_builder::cost::{CpuThreads, Fitted, Stages};
use knn_builder::{DataType, Job, SpaceType};
use serde_json::Value;

const MIB: f64 = 1_048_576.0;

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
    completed("runs.json")
}

/// The completed builds of a grid's runs, the warm-up ones left out.
fn completed(name: &str) -> Vec<Value> {
    let Value::Array(runs) = fixture(name) else {
        panic!("{name} is a list")
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
            "byte" => DataType::Byte,
            "binary" => DataType::Binary,
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

/// Relative errors, `predicted / observed - 1`, as fit.py summarizes them.
#[derive(Debug)]
struct Errors {
    median_abs: f64,
    max_abs: f64,
    most_under: f64,
}

impl Errors {
    fn of(pairs: impl IntoIterator<Item = (f64, f64)>) -> Self {
        let relative: Vec<f64> = pairs.into_iter().map(|(p, y)| p / y - 1.0).collect();
        let mut abs: Vec<f64> = relative.iter().map(|e| e.abs()).collect();
        abs.sort_by(f64::total_cmp);
        let mid = abs.len() / 2;
        let median_abs = if abs.len().is_multiple_of(2) {
            f64::midpoint(abs[mid - 1], abs[mid])
        } else {
            abs[mid]
        };
        Self {
            median_abs,
            max_abs: abs[abs.len() - 1],
            most_under: relative.iter().copied().fold(0.0, f64::min),
        }
    }

    /// fits.json's errors of `model` on the builds it was fitted to.
    fn trained(fits: &Value, model: &str) -> Self {
        let train = &fits[model]["train"];
        let number = |key: &str| train[key].as_f64().unwrap();
        Self {
            median_abs: number("median_abs"),
            max_abs: number("max_abs"),
            most_under: number("most_under"),
        }
    }

    /// No worse than `bound` on the median, the largest and the most under.
    fn within(&self, bound: &Self) -> bool {
        self.median_abs <= bound.median_abs && self.within_spread(bound)
    }

    /// No worse than `bound` on the largest and the most under.
    fn within_spread(&self, bound: &Self) -> bool {
        self.max_abs <= bound.max_abs && self.most_under >= bound.most_under
    }
}

/// One stage's seconds of a forecast.
type StageOf = fn(&Stages) -> f64;

#[test]
fn predicts_byte_builds_as_closely_as_the_float_ones() {
    let fits = fixture("fits.json");
    let bytes = completed("byte-runs.json");
    assert_eq!(bytes.len(), 24);
    let stages: [(&str, &str, StageOf); 3] = [
        ("download_seconds", "download", |s| s.download),
        ("build_seconds", "build", |s| s.build),
        ("upload_seconds", "upload", |s| s.upload),
    ];
    for (model, key, of) in stages {
        let errors = Errors::of(bytes.iter().map(|r| {
            let observed = r["seconds"][key].as_f64().unwrap();
            (of(&RTX_4070.stages(&job(r))), observed)
        }));
        let trained = Errors::trained(&fits, model);
        assert!(errors.within(&trained), "{model}: {errors:?}, {trained:?}");
    }

    // The device peak: off by what the float build of the same shape is
    // off by, a quarter of the dataset smaller, so by more of a smaller
    // peak; the margin still covers it.
    let float: Vec<Value> = builds()
        .into_iter()
        .filter(|r| r["shape"]["blob_dtype"] == "<f4")
        .collect();
    let under = |r: &Value| {
        r["device"]["peak_bytes"].as_f64().unwrap() - Fitted::device_peak_bytes(&job(r))
    };
    for record in &bytes {
        let shape = |r: &Value| {
            (
                r["shape"]["doc_count"].clone(),
                r["shape"]["dimension"].clone(),
                r["shape"]["m"].clone(),
            )
        };
        let twin = float.iter().find(|r| shape(r) == shape(record)).unwrap();
        assert!(
            under(record) <= under(twin).max(0.0) + MIB,
            "device peak of {}: under by {} bytes, the float build by {}",
            record["shape"],
            under(record),
            under(twin)
        );
        let needed = Fitted::device_bytes_needed(&job(record));
        assert!(
            needed >= record["device"]["peak_bytes"].as_u64().unwrap(),
            "{}",
            record["shape"]
        );
    }

    // The file: as close as for the builds fitted, which fits.json rounds
    // to none.
    for record in &bytes {
        let observed = record["file_size"].as_f64().unwrap();
        let predicted = Fitted::file_bytes(&job(record));
        assert!(
            (predicted / observed - 1.0).abs() < 1e-4,
            "file bytes of {}: {predicted} for {observed}",
            record["shape"]
        );
    }
}

#[test]
fn costs_what_the_binary_fit_says() {
    let fits = fixture("binary-fits.json");
    let builds = completed("binary-runs.json");
    assert_eq!(builds.len(), 30);
    let device = &fits["device_peak_bytes"];
    let rows = device["rows"].as_f64().unwrap();
    for record in &builds {
        let job = job(record);
        let shape = &record["shape"];
        let n = shape["doc_count"].as_f64().unwrap();
        let d = shape["dimension"].as_f64().unwrap();
        let g = 4.0 * shape["m"].as_f64().unwrap();
        let build = evaluate(
            &fits["build_seconds"]["terms"],
            &[
                ("1", 1.0),
                ("n d", n * d),
                ("n g", n * g),
                ("n g^2", n * g * g),
                ("n g^2 d", n * g * g * d),
            ],
        );
        close(RTX_4070.stages(&job).build, build, "build seconds", record);
        let file = evaluate(
            &fits["file_bytes"]["terms"],
            &[
                ("1", 1.0),
                ("n d / 8", n * d / 8.0),
                ("n g", n * g),
                ("n", n),
            ],
        );
        close(Fitted::file_bytes(&job), file, "file bytes", record);
        let nn_descent = n * d / 8.0
            + 4.0 * n * g
            + evaluate(
                &device["nn_descent_phase"],
                &[("1", 1.0), ("n", n), ("n g", n * g)],
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
            n * d / 8.0 + nn_descent.max(graph),
            "device peak",
            record,
        );
        // The margin covers every peak.
        assert!(
            Fitted::device_bytes_needed(&job) >= record["device"]["peak_bytes"].as_u64().unwrap(),
            "{shape}"
        );
    }
    // The transfers are the float models', within the float grid's spread:
    // a median over transfers of tenths of a second against LocalStack
    // moves with LocalStack (5.7% for the downloads here, 5.5% there).
    let float = fixture("fits.json");
    let stages: [(&str, &str, StageOf); 2] = [
        ("download_seconds", "download", |s| s.download),
        ("upload_seconds", "upload", |s| s.upload),
    ];
    for (model, key, of) in stages {
        let errors = Errors::of(builds.iter().map(|r| {
            let observed = r["seconds"][key].as_f64().unwrap();
            (of(&RTX_4070.stages(&job(r))), observed)
        }));
        let trained = Errors::trained(&float, model);
        assert!(
            errors.within_spread(&trained),
            "{model}: {errors:?}, {trained:?}"
        );
    }
}

#[test]
fn takes_the_binary_cpu_build_the_fit_says() {
    let fits = fixture("binary-fits.json");
    let template = job(&completed("binary-runs.json")[0]);
    let Value::Array(cpu) = fixture("binary-cpu.json") else {
        panic!("binary-cpu.json is a list")
    };
    assert_eq!(cpu.len(), 14);
    for record in &cpu {
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
        close(Fitted::cpu_seconds(&job, threads), fit, key, record);
    }
}

/// A data node builds a byte segment, over `SQ8_direct_signed`, in about
/// the time of a float one of its shape, so the CPU model is the float
/// one: each byte build within a fifth of the float build's seconds.
#[test]
fn takes_a_byte_cpu_build_for_a_float_one() {
    let float = fixture("cpu.json");
    let Value::Array(builds) = fixture("byte-cpu.json") else {
        panic!("byte-cpu.json is a list")
    };
    assert_eq!(builds.len(), 16);
    for record in &builds {
        assert_eq!(record["shape"]["data_type"], "byte");
        let key = |r: &Value| {
            (
                r["shape"]["doc_count"].clone(),
                r["shape"]["dimension"].clone(),
                r["threads"].clone(),
            )
        };
        let twin = float
            .as_array()
            .unwrap()
            .iter()
            .find(|r| key(r) == key(record))
            .unwrap();
        let ratio = record["seconds"].as_f64().unwrap() / twin["seconds"].as_f64().unwrap();
        assert!((0.8..=1.2).contains(&ratio), "{}: {ratio}", record["shape"]);
    }
}

#[test]
fn counts_a_byte_value_as_one_byte() {
    // Blob, file and upload see a byte vector of d values as a float one of
    // d / 4; the device holds 1 byte a value where float takes 4.
    let float = Job {
        doc_count: 500_000,
        dimension: 128,
        ..job(&builds()[0])
    };
    let byte = Job {
        data_type: DataType::Byte,
        ..float.clone()
    };
    let quarter = Job {
        dimension: 32,
        ..float.clone()
    };
    let (b, q) = (RTX_4070.stages(&byte), RTX_4070.stages(&quarter));
    assert_eq!((b.download, b.upload), (q.download, q.upload));
    assert_eq!(Fitted::file_bytes(&byte), Fitted::file_bytes(&quarter));
    let dataset = f64::from(float.doc_count) * f64::from(float.dimension);
    let saved = Fitted::device_peak_bytes(&float) - Fitted::device_peak_bytes(&byte);
    assert!((saved - 3.0 * dataset).abs() < 1.0, "{saved}");
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
