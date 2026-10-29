//! When each job in the pipeline will end, and when a job submitted now
//! would: the pipeline's own rules, played forward from where each job is.
//! A job downloads once the builder has taken the job before it and the
//! budget has room for its buffers, which it holds until its upload ends;
//! it builds once it is downloaded and the upload slot is free; it uploads
//! once it is built and the upload before it has ended. Each stage takes
//! the jobs in the order they were submitted.
//!
//! How long each stage takes comes from a [`StageModel`], corrected by what
//! completed jobs took: a ratio of observed to modeled time per stage, and
//! one per build configuration besides, since the kernels cuVS picks for a
//! configuration can make it much slower than its size alone says.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use knn_gpu::BuildParams;

use crate::cost::{Fitted, Stages};
use crate::pipeline::Budget;
use crate::{DataType, InFlight, Job, SpaceType, Stage, Timings};

/// How long a job's stages take, before what completed jobs teach.
pub trait StageModel {
    /// `job`'s stages, in seconds.
    fn stages(&self, job: &Job) -> Stages;
}

impl StageModel for Fitted {
    fn stages(&self, job: &Job) -> Stages {
        Fitted::stages(self, job)
    }
}

impl<F: Fn(&Job) -> Stages> StageModel for F {
    fn stages(&self, job: &Job) -> Stages {
        self(job)
    }
}

/// A stage that has run past its forecast is expected to need this share of
/// the time it has run so far again: it says nothing of when it will end,
/// but a stage that has run long tends to run longer.
const OVERRUN: f64 = 0.25;

/// The weight of the latest job in a ratio, once the earlier ones together
/// weigh more than it.
const LATEST: f64 = 0.25;

/// When jobs will end, from now.
#[derive(Debug, Clone, PartialEq)]
pub struct Forecast {
    /// Until each job in flight ends, in the order given.
    pub ends: Vec<Duration>,
    /// Until a job submitted now would end, when one was given.
    pub next: Option<Duration>,
}

/// Forecasts jobs through one pipeline, and learns from those that end.
#[derive(Debug)]
pub struct Forecaster<M> {
    model: M,
    /// The pipeline's budget, in MiB.
    budget: u32,
    download: Ratio,
    build: Ratio,
    upload: Ratio,
    /// Build ratios by configuration, where one is known.
    builds: HashMap<Configuration, Ratio>,
}

/// What decides which kernels a build runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Configuration {
    pq_dim: u32,
    degree: u32,
    space_type: SpaceType,
    data_type: DataType,
}

impl Configuration {
    fn of(job: &Job) -> Self {
        let params = BuildParams::new(
            job.doc_count,
            job.dimension,
            job.m,
            job.space_type.build_metric(),
        )
        .ok();
        Self {
            pq_dim: params.map_or(0, |p| p.pq_dim),
            degree: params.map_or(0, |p| p.graph_degree),
            space_type: job.space_type,
            data_type: job.data_type,
        }
    }
}

/// Observed over modeled time, as a running mean of its logarithm.
#[derive(Debug, Clone, Copy, Default)]
struct Ratio {
    log: f64,
    seen: u32,
}

impl Ratio {
    fn factor(self) -> f64 {
        self.log.exp()
    }

    /// Folds in one job's `observed` and `modeled` seconds. The ratio
    /// starts from `prior` (a logarithm), which weighs as one job; each
    /// job then weighs as much as those before it together, until its
    /// weight falls to [`LATEST`], from where older jobs fade.
    fn learn(&mut self, prior: f64, observed: f64, modeled: f64) {
        if !(observed > 0.0 && modeled > 0.0) {
            return;
        }
        if self.seen == 0 {
            self.log = prior;
        }
        let weight = (1.0 / f64::from(self.seen.saturating_add(2))).max(LATEST);
        self.log += weight * ((observed / modeled).ln() - self.log);
        self.seen = self.seen.saturating_add(1);
    }
}

impl<M: StageModel> Forecaster<M> {
    /// A forecaster for a pipeline started with `host_memory` bytes of
    /// budget, its stages as `model` has them until jobs end.
    pub fn new(model: M, host_memory: u64) -> Self {
        Self {
            model,
            budget: Budget::total(host_memory),
            download: Ratio::default(),
            build: Ratio::default(),
            upload: Ratio::default(),
            builds: HashMap::new(),
        }
    }

    /// Learns from a job that ended with its index stored, by how long
    /// each of its stages took.
    pub fn learn(&mut self, job: &Job, timings: &Timings) {
        let modeled = self.model.stages(job);
        let seconds = |span: crate::Span| span.duration().as_secs_f64();
        self.download
            .learn(0.0, seconds(timings.download), modeled.download);
        self.upload
            .learn(0.0, seconds(timings.upload), modeled.upload);
        // A configuration seen for the first time starts from what all
        // builds have shown so far.
        let prior = self.build.log;
        self.builds
            .entry(Configuration::of(job))
            .or_default()
            .learn(prior, seconds(timings.build), modeled.build);
        self.build.learn(0.0, seconds(timings.build), modeled.build);
    }

    /// `job`'s stages as the forecast takes them: modeled, then corrected
    /// by what completed jobs took.
    #[must_use]
    pub fn stages(&self, job: &Job) -> Stages {
        let modeled = self.model.stages(job);
        let build = self
            .builds
            .get(&Configuration::of(job))
            .copied()
            .unwrap_or(self.build);
        Stages {
            download: modeled.download * self.download.factor(),
            build: modeled.build * build.factor(),
            upload: modeled.upload * self.upload.factor(),
        }
    }

    /// When each job `in_flight` (as [`Pipeline::in_flight`] gives them,
    /// in order) will end, and `next`, submitted behind them now, would.
    ///
    /// [`Pipeline::in_flight`]: crate::Pipeline::in_flight
    #[must_use]
    pub fn forecast(&self, now: Instant, in_flight: &[InFlight], next: Option<&Job>) -> Forecast {
        let mut lanes = Lanes::default();
        let ends = in_flight
            .iter()
            .map(|job| {
                let elapsed = now.saturating_duration_since(job.since).as_secs_f64();
                duration(lanes.run(self, &job.job, job.stage, elapsed, job.canceled))
            })
            .collect();
        let next = next.map(|job| duration(lanes.run(self, job, Stage::Queued, 0.0, false)));
        Forecast { ends, next }
    }
}

/// `seconds` as a duration; a model's time that is not one (negative, or
/// not a number) is never.
fn duration(seconds: f64) -> Duration {
    Duration::try_from_secs_f64(seconds).unwrap_or(Duration::MAX)
}

/// The time left in a stage modeled to take `modeled` seconds, `elapsed`
/// into it.
fn remaining(modeled: f64, elapsed: f64) -> f64 {
    if elapsed < modeled {
        modeled - elapsed
    } else {
        elapsed * OVERRUN
    }
}

/// When each stage, its slot and the budget are next free, in seconds from
/// now, as the jobs are played through in order.
#[derive(Debug, Default)]
struct Lanes {
    /// The download stage is done with the jobs so far.
    download: f64,
    /// The builder has taken the last job sent to the build slot.
    build_slot: f64,
    /// The builder is done with the jobs so far.
    build: f64,
    /// The upload stage has taken the last job sent to the upload slot.
    upload_slot: f64,
    /// The upload stage is done with the jobs so far.
    upload: f64,
    /// The budget's shares taken, in MiB, and until when.
    held: Vec<(f64, u32)>,
}

impl Lanes {
    /// Plays `job`, in `stage` for `elapsed` seconds, through what is left
    /// of the pipeline; returns when it ends.
    fn run<M: StageModel>(
        &mut self,
        forecaster: &Forecaster<M>,
        job: &Job,
        stage: Stage,
        elapsed: f64,
        canceled: bool,
    ) -> f64 {
        let stages = forecaster.stages(job);
        let share = Budget::share(job, forecaster.budget);
        // Withdrawn before it started: it never takes anything.
        if canceled && stage == Stage::Queued {
            return 0.0;
        }
        // Downloaded by: a canceled download stops at once, and the job
        // ends there.
        let downloaded = match stage {
            Stage::Queued => {
                let reserved = self.download.max(self.build_slot);
                let start = self.room(reserved, share, forecaster.budget);
                start + stages.download
            }
            Stage::Downloading if canceled => return 0.0,
            Stage::Downloading => remaining(stages.download, elapsed),
            Stage::Loaded | Stage::Building | Stage::Graphed | Stage::Uploading => 0.0,
        };
        self.download = self.download.max(downloaded);
        // Built by. The builder reserves the upload slot before it takes
        // the next job, so a build starts once that slot is free too. A
        // canceled job in the build slot ends as the builder takes it; a
        // canceled build runs to its end.
        let built = match stage {
            Stage::Queued | Stage::Downloading | Stage::Loaded => {
                let start = downloaded.max(self.build).max(self.upload_slot);
                self.build_slot = start;
                if canceled {
                    self.build = start;
                    self.held.push((start, share));
                    return start;
                }
                start + stages.build
            }
            Stage::Building => remaining(stages.build, elapsed),
            Stage::Graphed | Stage::Uploading => 0.0,
        };
        self.build = self.build.max(built);
        // Uploaded by: a canceled upload stops at once.
        let uploaded = match stage {
            Stage::Uploading if canceled => 0.0,
            Stage::Uploading => remaining(stages.upload, elapsed),
            _ => {
                let start = built.max(self.upload);
                self.upload_slot = start;
                if canceled {
                    start
                } else {
                    start + stages.upload
                }
            }
        };
        self.upload = self.upload.max(uploaded);
        self.held.push((uploaded, share));
        uploaded
    }

    /// The earliest time from `at` on when `share` MiB fit in a budget of
    /// `total` beside the shares held then.
    ///
    /// Shares given back by `at` are dropped for good: each queued job
    /// asks from a time no earlier than the one before it did (the
    /// download stage and the build slot never come free sooner), so they
    /// would never count again, and the forecast stays linear in the jobs
    /// played however long the queue.
    fn room(&mut self, at: f64, share: u32, total: u32) -> f64 {
        self.held.retain(|&(until, _)| until > at);
        let mut releases = self.held.clone();
        releases.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut used: u64 = releases.iter().map(|&(_, mib)| u64::from(mib)).sum();
        let mut time = at;
        for (until, mib) in releases {
            if used + u64::from(share) <= u64::from(total) {
                return time;
            }
            used -= u64::from(mib);
            time = until;
        }
        time
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::{Forecaster, StageModel, Stages};
    use crate::job::tests::job;
    use crate::pipeline::Budget;
    use crate::{InFlight, Job, Span, Stage, Timings};

    /// One second to download, two to build, one to upload, whatever the
    /// job.
    fn fixed(_: &Job) -> Stages {
        Stages {
            download: 1.0,
            build: 2.0,
            upload: 1.0,
        }
    }

    fn at(now: Instant, job: Job, stage: Stage, elapsed: f64, canceled: bool) -> InFlight {
        InFlight {
            id: 0,
            job,
            stage,
            since: now.checked_sub(Duration::from_secs_f64(elapsed)).unwrap(),
            canceled,
        }
    }

    fn seconds(durations: &[Duration]) -> Vec<f64> {
        durations
            .iter()
            .map(|d| (d.as_secs_f64() * 1e6).round() / 1e6)
            .collect()
    }

    fn queued(now: Instant, count: usize) -> Vec<InFlight> {
        (0..count)
            .map(|_| at(now, job("bucket", 1_000, 64), Stage::Queued, 0.0, false))
            .collect()
    }

    #[test]
    fn plays_queued_jobs_through_the_stages() {
        let now = Instant::now();
        let forecaster = Forecaster::new(fixed, u64::MAX);
        let forecast = forecaster.forecast(now, &queued(now, 3), Some(&job("bucket", 1_000, 64)));
        // Each download waits for the builder to take the job before; each
        // build for the upload slot; each upload for the one before.
        assert_eq!(seconds(&forecast.ends), [4.0, 6.0, 8.0]);
        assert_eq!(forecast.next.map(|d| d.as_secs_f64()), Some(10.0));
    }

    #[test]
    fn waits_for_room_in_the_budget() {
        let now = Instant::now();
        // Two of these do not fit at once: each downloads once the one
        // before has ended.
        let large = job("bucket", 100_000, 128);
        let share = Budget::share(&large, u32::MAX);
        let budget = u64::from(share * 3 / 2) << 20;
        let forecaster = Forecaster::new(fixed, budget);
        let jobs: Vec<_> = (0..3)
            .map(|_| at(now, large.clone(), Stage::Queued, 0.0, false))
            .collect();
        let forecast = forecaster.forecast(now, &jobs, None);
        assert_eq!(seconds(&forecast.ends), [4.0, 8.0, 12.0]);
    }

    #[test]
    fn starts_each_job_where_it_is() {
        let now = Instant::now();
        let small = || job("bucket", 1_000, 64);
        let jobs = [
            at(now, small(), Stage::Uploading, 0.5, false),
            at(now, small(), Stage::Building, 0.5, false),
            at(now, small(), Stage::Downloading, 0.25, false),
            at(now, small(), Stage::Queued, 0.0, false),
        ];
        let forecast = Forecaster::new(fixed, u64::MAX).forecast(now, &jobs, None);
        assert_eq!(seconds(&forecast.ends), [0.5, 2.5, 4.5, 6.5]);
    }

    #[test]
    fn expects_a_stage_past_its_forecast_to_run_on() {
        let now = Instant::now();
        let jobs = [at(
            now,
            job("bucket", 1_000, 64),
            Stage::Uploading,
            2.0,
            false,
        )];
        let forecast = Forecaster::new(fixed, u64::MAX).forecast(now, &jobs, None);
        assert_eq!(seconds(&forecast.ends), [0.5]);
    }

    #[test]
    fn plays_canceled_jobs_as_the_stages_end_them() {
        let now = Instant::now();
        let small = || job("bucket", 1_000, 64);
        // A canceled build runs to its end, then its upload stops at once;
        // a canceled queued job never starts.
        let jobs = [
            at(now, small(), Stage::Building, 0.5, true),
            at(now, small(), Stage::Queued, 0.0, true),
            at(now, small(), Stage::Queued, 0.0, false),
        ];
        let forecast = Forecaster::new(fixed, u64::MAX).forecast(now, &jobs, None);
        assert_eq!(seconds(&forecast.ends), [1.5, 0.0, 4.5]);
        // A canceled download stops at once, and the job behind it takes
        // its place.
        let jobs = [
            at(now, small(), Stage::Downloading, 0.5, true),
            at(now, small(), Stage::Queued, 0.0, false),
        ];
        let forecast = Forecaster::new(fixed, u64::MAX).forecast(now, &jobs, None);
        assert_eq!(seconds(&forecast.ends), [0.0, 4.0]);
    }

    fn timings(download: f64, build: f64, upload: f64) -> Timings {
        let start = Instant::now();
        let span = |from: f64, to: f64| Span {
            start: start + Duration::from_secs_f64(from),
            end: start + Duration::from_secs_f64(to),
        };
        Timings {
            download: span(0.0, download),
            build: span(download, download + build),
            upload: span(download + build, download + build + upload),
        }
    }

    /// The build `forecaster` expects of `job`, ready to build now.
    fn build<M: StageModel>(forecaster: &Forecaster<M>, job: &Job) -> f64 {
        let now = Instant::now();
        let loaded = at(now, job.clone(), Stage::Loaded, 0.0, false);
        // The upload, modeled at a second, never learns here.
        forecaster.forecast(now, &[loaded], None).ends[0].as_secs_f64() - 1.0
    }

    #[test]
    fn learns_from_jobs_that_end() {
        let mut forecaster = Forecaster::new(fixed, u64::MAX);
        let built = job("bucket", 1_000, 64);
        // Builds twice as long as modeled: the first job moves the ratio
        // halfway from the model, the second a third of the rest.
        forecaster.learn(&built, &timings(1.0, 4.0, 1.0));
        assert!((build(&forecaster, &built) - 2.0 * 2f64.sqrt()).abs() < 1e-9);
        forecaster.learn(&built, &timings(1.0, 4.0, 1.0));
        assert!((build(&forecaster, &built) - 2.0 * 2f64.powf(2.0 / 3.0)).abs() < 1e-9);
    }

    #[test]
    fn keeps_a_build_ratio_per_configuration() {
        let mut forecaster = Forecaster::new(fixed, u64::MAX);
        let mut slow = job("bucket", 1_000, 64);
        slow.m = 32;
        let other = job("bucket", 1_000, 128);
        for _ in 0..60 {
            forecaster.learn(&slow, &timings(1.0, 8.0, 1.0));
        }
        // The slow configuration has learned its own; another, never seen,
        // takes the ratio of all builds so far, the same jobs here.
        assert!((build(&forecaster, &slow) - 8.0).abs() < 1e-3);
        assert!((build(&forecaster, &other) - 8.0).abs() < 1e-3);
        // Once the other is seen to run as modeled, it keeps its own, and
        // the slow one keeps its.
        for _ in 0..60 {
            forecaster.learn(&other, &timings(1.0, 2.0, 1.0));
        }
        assert!((build(&forecaster, &other) - 2.0).abs() < 1e-3);
        assert!((build(&forecaster, &slow) - 8.0).abs() < 1e-3);
    }
}
