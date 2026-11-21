//! The metrics page against `tests/data/metrics.txt`: a page with every
//! metric the service has, the optional ones included, every value of
//! every label, and a count in most series. CI puts the same file through
//! `promtool check metrics`, Prometheus's own reading of the format and
//! its naming rules, so the renderer is held to a page Prometheus takes.
//!
//! After a change meant to change the page, run this test with
//! `RECORD_METRICS_PAGE` set to write the file anew, and read the diff.

#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    reason = "test crate"
)]

use std::fs;
use std::iter;
use std::path::PathBuf;

use knn_builder::metrics::{
    Answered, BuildAnswer, FlightStage, GpuMemory, Label, Outcome, Route, Snapshot, TimedStage,
    VectorType,
};

fn recorded() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/metrics.txt")
}

/// `count` values spread over a histogram's buckets and past them: four
/// alike, then four `step` times those, and so on. By multiplication
/// alone, which rounds the same way everywhere, so the page's sums are the
/// same on every platform.
fn spread(count: usize, first: f64, step: f64) -> impl Iterator<Item = f64> {
    iter::successors(Some(first), move |value| Some(value * step))
        .flat_map(|value| [value; 4])
        .take(count)
}

/// Every metric with values in it, the optional ones included; a few
/// series are left at 0, as a running service leaves them.
fn sample() -> Snapshot {
    let mut snapshot = Snapshot {
        // Not the crate's, so a release leaves the page as it is.
        version: "0.0.0",
        http_requests: [
            (Route::Build, 200, 61),
            (Route::Build, 422, 2),
            (Route::Build, 507, 9),
            (Route::Status, 200, 1403),
            (Route::Status, 404, 3),
            (Route::Jobs, 401, 1),
            (Route::HeartBeat, 200, 240),
            (Route::Metrics, 200, 239),
            (Route::Other, 404, 4),
        ]
        .into_iter()
        .map(|(route, code, requests)| Answered {
            route,
            code,
            requests,
        })
        .collect(),
        jobs_held: 57,
        jobs_held_limit: 10_000,
        drain_seconds: 148.25,
        host_budget: 30_517 << 20,
        host_budget_used: 7_412 << 20,
        resident_memory: Some(4_104_531_968),
        start_time_seconds: 1_795_219_200.5,
        gpu_memory: Some(GpuMemory {
            total: 8_585_740_288,
            free: 7_301_234_688,
            last_build_peak: 2_164_260_864,
        }),
        ..Snapshot::default()
    };
    let counted = &mut snapshot.counted;
    for (answer, count) in BuildAnswer::ALL.iter().zip([55_u64, 6, 2, 1, 0, 3, 4, 2]) {
        counted.build_requests.add(*answer, count);
    }
    for (outcome, count) in Outcome::ALL
        .iter()
        .zip([44_u64, 2, 0, 1, 1, 0, 0, 1, 2, 3, 1, 0, 0])
    {
        counted.jobs_ended.add(*outcome, count);
    }
    for (data_type, vectors) in
        VectorType::ALL
            .iter()
            .zip([18_000_000_u64, 6_500_000, 9_250_000, 4_000_000])
    {
        counted.vectors_built.add(*data_type, vectors);
    }
    for (stage, jobs) in FlightStage::ALL.iter().zip([5_u64, 1, 0, 1, 0, 1]) {
        snapshot.jobs_in_flight.set(*stage, jobs);
    }
    for seconds in spread(44, 0.4, 2.5) {
        counted.job_seconds.observe(seconds);
    }
    for seconds in spread(44, 0.06, 2.75) {
        counted.queue_seconds.observe(seconds);
    }
    let stages = [
        (TimedStage::Download, 44, 0.08, 2.0, 1.25),
        (TimedStage::Build, 44, 0.2, 2.25, 0.5),
        (TimedStage::Projection, 9, 1.5, 3.0, 1.75),
        (TimedStage::RecallCheck, 30, 0.3, 1.5, 0.875),
        (TimedStage::Upload, 44, 0.15, 2.125, 2.5),
    ];
    for (stage, count, first, step, ratio) in stages {
        for seconds in spread(count, first, step) {
            counted.stage_seconds.observe(stage, seconds);
        }
        snapshot.stage_model_ratio.set(stage, ratio);
    }
    for ratio in spread(44, 0.375, 1.25) {
        counted.forecast_ratio.observe(ratio);
    }
    // Thirty jobs checked, two of them failed for it.
    for recall in (0..30).map(|i| 1.0 - f64::from(i * i) / 1600.0) {
        counted.recall.observe(recall);
    }
    snapshot
}

#[test]
fn renders_the_recorded_page() {
    let page = sample().render();
    if std::env::var_os("RECORD_METRICS_PAGE").is_some() {
        fs::write(recorded(), &page).unwrap();
    }
    let recorded = fs::read_to_string(recorded()).unwrap();
    for (number, (rendered, recorded)) in page.lines().zip(recorded.lines()).enumerate() {
        assert_eq!(rendered, recorded, "line {}", number + 1);
    }
    assert_eq!(page.lines().count(), recorded.lines().count());
    assert_eq!(page, recorded, "the line endings differ");
}

/// The sample adds to an idle page the metrics that one leaves out, and
/// leaves few of the page's series at 0.
#[test]
fn the_sample_fills_the_page() {
    let page = sample().render();
    let idle = Snapshot::default().render();
    let series = |page: &str| page.lines().filter(|line| !line.starts_with('#')).count();
    // The requests and the memory an idle page does not list.
    assert_eq!(series(&page), series(&idle) + 9 + 4);
    let zeros = page.lines().filter(|line| line.ends_with(" 0")).count();
    // Those left: a few outcomes, answers and stages, and the buckets
    // below a histogram's least value.
    assert!(zeros < series(&page) / 8, "{zeros} series at 0");
}
