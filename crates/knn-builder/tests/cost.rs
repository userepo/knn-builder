//! The cost model against what it was fitted from: every build of the grid
//! (`fixtures/grid/runs.json`) costs what `tools/grid/fit.py`'s models in
//! `fixtures/grid/fits.json` say, so the constants are the fit's, and a
//! refit that changes them fails here until the model follows. And against
//! what it was not fitted to, byte builds (`fixtures/grid/byte-runs.json`,
//! `byte-cpu.json`), predicted within the errors of the builds it was. The
//! binary models are held to `binary-fits.json` on every binary build alike
//! (`binary-runs.json`, `binary-cpu.json`). A build's device peak is, as
//! fit.py takes it, the larger of its 5 ms sample and its reading in
//! `tight.json`, where it has one. `first.json` holds what the grids leave
//! out: builds of fewer vectors than the IVF-PQ build's batch, and each
//! build as its process's first.

#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    reason = "test crate"
)]

use std::fs;
use std::num::NonZeroUsize;
use std::path::Path;

use knn_builder::cost::{CpuThreads, Fitted, Stages};
use knn_builder::{DataType, Job, SpaceType};
use serde_json::Value;

/// The RTX 4070 Laptop GPU's shared memory a block, which the grid ran on.
const RTX_4070: Fitted = Fitted {
    shared_memory_per_block: 101_376,
    projection_threads: NonZeroUsize::MIN.saturating_add(7),
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

/// A build's device peak: the larger of its 5 ms sample and its reading
/// in `tight.json`, which the briefest builds need.
fn peak(record: &Value) -> f64 {
    let Value::Array(tight) = fixture("tight.json") else {
        panic!("tight.json is a list")
    };
    let sampled = record["device"]["peak_bytes"].as_f64().unwrap();
    tight
        .iter()
        .find(|t| t["input"] == record["input"] && t["m"] == record["shape"]["m"])
        .map_or(sampled, |t| sampled.max(t["peak_bytes"].as_f64().unwrap()))
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
        project_link_to_base_layer: false,
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
        // The codes and fit.py's BATCH_ROWS of vectors at most.
        let ivf_pq = n * d / 4.0
            + n.min(65_536.0) * d * s
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
            ivf_pq.max(graph),
            "device peak",
            record,
        );
    }
}

/// `first.json`'s input as a job: `l2-NxD`, fp16 where it says so.
fn first_job(record: &Value) -> Job {
    let name = record["input"].as_str().unwrap();
    let mut parts = name.split('-');
    let (n, d) = parts.nth(1).unwrap().split_once('x').unwrap();
    Job {
        doc_count: n.parse().unwrap(),
        dimension: d.parse().unwrap(),
        data_type: if parts.next() == Some("fp16") {
            DataType::HalfFloat
        } else {
            DataType::Float
        },
        m: u32::try_from(record["m"].as_u64().unwrap()).unwrap(),
        ..job(&builds()[0])
    }
}

#[test]
fn needs_what_a_process_s_first_build_takes() {
    let Value::Array(first) = fixture("first.json") else {
        panic!("first.json is a list")
    };
    assert_eq!(first.len(), 6);
    for record in &first {
        let job = first_job(record);
        let needed = Fitted::device_bytes_needed(&job);
        let first = record["first_peak_bytes"].as_u64().unwrap();
        let later = record["peak_bytes"].as_u64().unwrap();
        // The first build takes 40 to 42 MiB more at every size.
        let more = first - later;
        assert!((40 << 20..=42 << 20).contains(&more), "{}", record["input"]);
        assert!(needed >= first, "{}: {needed} for {first}", record["input"]);
        // Below the 100,000 vectors the model is fitted from, it
        // over-estimates: the peak alone covers a later build.
        assert!(
            job.doc_count >= 100_000
                || Fitted::device_peak_bytes(&job) >= record["peak_bytes"].as_f64().unwrap(),
            "{}",
            record["input"]
        );
    }
}

#[test]
fn holds_a_batch_of_no_more_vectors_than_the_job_has() {
    // 1,000 vectors of dimension 64 as float and as fp16 peak alike in
    // first.json, to the device's 2 MiB: their batches are 1,000 rows, 128
    // KB apart, where 65,536 rows would be 8 MiB apart.
    let float = Job {
        doc_count: 1_000,
        dimension: 64,
        ..job(&builds()[0])
    };
    let half = Job {
        data_type: DataType::HalfFloat,
        ..float.clone()
    };
    let apart = Fitted::device_peak_bytes(&float) - Fitted::device_peak_bytes(&half);
    assert!((apart - 1_000.0 * 64.0 * 2.0).abs() < 1.0, "{apart}");
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
        ..RTX_4070
    };
    assert!(!larger.slow_search(&job));
}

#[test]
fn never_forecasts_less_device_memory_than_the_codes_batch_and_graph() {
    for (doc_count, dimension, m) in [(5_000_000, 128, 32), (10_000_000, 768, 16)] {
        let job = Job {
            doc_count,
            dimension,
            m,
            ..job(&builds()[0])
        };
        let (n, d, g) = (f64::from(doc_count), f64::from(dimension), f64::from(4 * m));
        let floor = (n * d / 4.0 + 65_536.0 * d * 4.0).max(4.0 * n * g);
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

    /// Widened by the half unit of the fourth decimal fits.json rounds its
    /// errors to, so an error that equals a bound before rounding is within
    /// it.
    fn rounded_up(&self) -> Self {
        const HALF_UNIT: f64 = 0.000_05;
        Self {
            median_abs: self.median_abs + HALF_UNIT,
            max_abs: self.max_abs + HALF_UNIT,
            most_under: self.most_under - HALF_UNIT,
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

/// How far the float transfer models may miss a byte or binary build's
/// download or upload. Those blobs and files are small, and LocalStack's
/// own spread on transfers of a fraction of a second is wider than any one
/// grid's fit: in the session of the float grid, whose transfers fitted
/// within 20 and 24%, the binary grid's downloads missed by up to 45% and
/// its uploads by 34%. A blob counted at the wrong size would miss by a
/// factor of two or more. The forecast corrects transfer times from the
/// jobs that end.
const TRANSFER_SPREAD: f64 = 0.5;

/// The float transfer models on `builds`, each within [`TRANSFER_SPREAD`].
fn assert_transfers_within_spread(builds: &[Value]) {
    let stages: [(&str, StageOf); 2] = [("download", |s| s.download), ("upload", |s| s.upload)];
    for (key, of) in stages {
        let errors = Errors::of(builds.iter().map(|r| {
            let observed = r["seconds"][key].as_f64().unwrap();
            (of(&RTX_4070.stages(&job(r))), observed)
        }));
        assert!(errors.max_abs <= TRANSFER_SPREAD, "{key}: {errors:?}");
    }
}

#[test]
fn predicts_byte_builds_as_closely_as_the_float_ones() {
    let fits = fixture("fits.json");
    let bytes = completed("byte-runs.json");
    assert_eq!(bytes.len(), 24);
    // The build: as close as on the builds fitted. The transfers: within
    // LocalStack's spread.
    let errors = Errors::of(bytes.iter().map(|r| {
        let observed = r["seconds"]["build"].as_f64().unwrap();
        (RTX_4070.stages(&job(r)).build, observed)
    }));
    let trained = Errors::trained(&fits, "build_seconds");
    assert!(errors.within(&trained), "build: {errors:?}, {trained:?}");
    assert_transfers_within_spread(&bytes);

    // The device peak: as close as on the builds fitted, its batch a byte a
    // value; and within the margin.
    let errors = Errors::of(
        bytes
            .iter()
            .map(|r| (Fitted::device_peak_bytes(&job(r)), peak(r))),
    );
    let trained = Errors::trained(&fits, "device_peak_bytes");
    assert!(
        errors.within(&trained.rounded_up()),
        "device peak: {errors:?}, {trained:?}"
    );
    for record in &bytes {
        let needed = Fitted::device_bytes_needed(&job(record));
        #[expect(
            clippy::cast_precision_loss,
            reason = "bytes of a few GB, exact in f64"
        )]
        let needed = needed as f64;
        assert!(needed >= peak(record), "{}", record["shape"]);
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
    // The transfers are the float models', within LocalStack's spread.
    assert_transfers_within_spread(&builds);
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
    // d / 4; the IVF-PQ build's batch on the device holds 1 byte a value
    // where float takes 4, which shows where that phase is the peak, as for
    // 100,000 vectors of dimension 768.
    let float = Job {
        doc_count: 100_000,
        dimension: 768,
        ..job(&builds()[0])
    };
    let byte = Job {
        data_type: DataType::Byte,
        ..float.clone()
    };
    let quarter = Job {
        dimension: float.dimension / 4,
        ..float.clone()
    };
    let (b, q) = (RTX_4070.stages(&byte), RTX_4070.stages(&quarter));
    assert_eq!((b.download, b.upload), (q.download, q.upload));
    assert_eq!(Fitted::file_bytes(&byte), Fitted::file_bytes(&quarter));
    let batch = 65_536.0 * f64::from(float.dimension);
    let saved = Fitted::device_peak_bytes(&float) - Fitted::device_peak_bytes(&byte);
    assert!((saved - 3.0 * batch).abs() < 1.0, "{saved}");
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

/// The projection's model against what `knn-write`'s `projection_cost`
/// test measured it to take (`fixtures/grid/projection.json`) where it built
/// layer 1: within half and three times of each, the spread its fit left
/// between vectors in clusters apart and overlapping, and at m 2.
#[test]
fn models_the_projection_within_its_spread() {
    let Value::Array(records) = fixture("projection.json") else {
        panic!("projection.json is a list")
    };
    let mut checked = 0;
    for record in records.iter().filter(|r| r["outcome"] == "projected") {
        let number = |key: &str| record[key].as_u64().unwrap();
        let job = Job {
            doc_count: u32::try_from(number("n")).unwrap(),
            dimension: u32::try_from(number("dimension")).unwrap(),
            m: u32::try_from(number("m")).unwrap(),
            ef_construction: i32::try_from(number("ef_construction")).unwrap(),
            project_link_to_base_layer: true,
            ..job(&builds()[0])
        };
        // The service lowers no m of these counts.
        let degree = job.graph_params().unwrap().degree_for(job.doc_count);
        assert_eq!(degree, u32::try_from(number("degree")).unwrap());
        let model = Fitted {
            projection_threads: NonZeroUsize::new(usize::try_from(number("threads")).unwrap())
                .unwrap(),
            ..RTX_4070
        };
        let ratio = model.projection_seconds(&job) / record["seconds"].as_f64().unwrap();
        assert!((0.5..=3.0).contains(&ratio), "{ratio}: {record}");
        checked += 1;
    }
    assert!(checked >= 50, "{checked} records");
}
