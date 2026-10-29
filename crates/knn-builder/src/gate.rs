//! What lets a new job in, once the upstream service's own checks have:
//! whether the GPU has the memory for its build, whether it would end
//! before the data node stops waiting for it, and, when asked, whether the
//! data node would build it sooner itself. A job refused here gets 507 at
//! once, which k-NN does not retry: the data node builds the segment on its
//! CPU now, rather than after a timeout. As jobs end, and the forecast
//! learns from them, the jobs let in and not yet building are forecast
//! again, and those that would now miss are named, to be dropped.
//!
//! The data node's wait starts when its request returns. It asks for the
//! job's status first after three poll intervals and then once an interval,
//! each with up to 20% of jitter, and polls no more once its timeout has
//! passed: a job is seen ended at the first poll after it ends, which comes
//! up to 1.2 intervals later, or at the first poll at all, up to 3.6
//! intervals in, and that poll must come before the timeout.

use std::time::{Duration, Instant};

use crate::cost::{CpuThreads, Fitted};
use crate::forecast::Forecaster;
use crate::request::JobId;
use crate::service::{Refusal, Settings};
use crate::{InFlight, Job, Stage, Timings};

/// The most a poll interval stretches with k-NN's jitter.
const JITTER: f64 = 1.2;

/// Polls k-NN waits before its first.
const FIRST_POLL: f64 = 3.0;

/// The checks, and the forecast they rest on.
#[derive(Debug)]
pub(crate) struct Gate {
    forecaster: Forecaster<Fitted>,
    timeout: f64,
    poll: f64,
    cpu: Option<CpuThreads>,
    /// The memory a build may take, when there is a GPU to read.
    device_memory: Option<u64>,
}

/// A job let in: when it is forecast to end.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Admitted {
    /// From now until it ends.
    pub(crate) ends_in: Duration,
}

impl Gate {
    pub(crate) fn new(settings: &Settings) -> Self {
        let model = settings
            .device
            .map_or_else(Fitted::default, |device| Fitted {
                shared_memory_per_block: device.shared_memory_per_block,
            });
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "bytes, as the pipeline's budget takes them"
        )]
        let host_memory = settings.cpu_memory_limit as u64;
        Self {
            forecaster: Forecaster::new(model, host_memory),
            timeout: settings.client_timeout.as_secs_f64(),
            poll: settings.poll_interval.as_secs_f64(),
            cpu: settings.cpu_build,
            device_memory: settings.device.map(|device| device.free_memory),
        }
    }

    /// Learns from a job that ended with its index stored.
    pub(crate) fn learn(&mut self, job: &Job, timings: &Timings) {
        self.forecaster.learn(job, timings);
    }

    /// Lets `job` in behind the jobs `in_flight`, or says why not.
    pub(crate) fn admit(
        &self,
        job_id: &JobId,
        vector_path: &str,
        job: &Job,
        in_flight: &[InFlight],
        now: Instant,
    ) -> Result<Admitted, Refusal> {
        if let Some(free) = self.device_memory {
            let needed = Fitted::device_bytes_needed(job);
            if needed > free {
                tracing::info!(
                    %job_id,
                    vector_path,
                    needed_mib = needed.div_ceil(1 << 20),
                    free_mib = free >> 20,
                    "too little device memory for the build"
                );
                return Err(Refusal::TooLarge {
                    job_id: job_id.clone(),
                    vector_path: vector_path.to_owned(),
                });
            }
        }
        let forecast = self.forecaster.forecast(now, in_flight, Some(job));
        let ends = forecast.next.map_or(f64::INFINITY, |d| d.as_secs_f64());
        let seen = self.latest_seen(ends);
        if seen >= self.timeout {
            tracing::info!(%job_id, vector_path, forecast_s = ends, "forecast to end too late");
            return Err(Refusal::Late {
                job_id: job_id.clone(),
                vector_path: vector_path.to_owned(),
                seen: whole_seconds(seen),
                timeout: whole_seconds(self.timeout),
            });
        }
        if let Some(threads) = self.cpu {
            // The data node sees the job ended at a poll, then reads the
            // index back about as fast as the service wrote it.
            let remote = self.seen(ends) + self.forecaster.stages(job).upload;
            let cpu = Fitted::cpu_seconds(job, threads);
            if remote > cpu {
                return Err(Refusal::CpuSooner {
                    job_id: job_id.clone(),
                    vector_path: vector_path.to_owned(),
                    remote: whole_seconds(remote),
                    cpu: whole_seconds(cpu),
                });
            }
        }
        Ok(Admitted {
            ends_in: Duration::from_secs_f64(ends),
        })
    }

    /// The latest the data node, its polls stretched by all their jitter,
    /// sees a job that ends at `ends` seconds.
    fn latest_seen(&self, ends: f64) -> f64 {
        let first = FIRST_POLL * JITTER * self.poll;
        if ends <= first {
            first
        } else {
            ends + JITTER * self.poll
        }
    }

    /// The jobs `in_flight` not yet building that the forecast now has
    /// seen ended only past the data node's timeout: their ids.
    /// `admitted` gives when a job was let in, for the jobs it knows.
    pub(crate) fn overdue(
        &self,
        now: Instant,
        in_flight: &[InFlight],
        admitted: impl Fn(u64) -> Option<Instant>,
    ) -> Vec<u64> {
        let forecast = self.forecaster.forecast(now, in_flight, None);
        in_flight
            .iter()
            .zip(&forecast.ends)
            .filter(|(flight, _)| {
                !flight.canceled
                    && matches!(
                        flight.stage,
                        Stage::Queued | Stage::Downloading | Stage::Loaded
                    )
            })
            .filter_map(|(flight, ends)| {
                let since = now.saturating_duration_since(admitted(flight.id)?);
                let ends = since.saturating_add(*ends).as_secs_f64();
                (self.latest_seen(ends) >= self.timeout).then_some(flight.id)
            })
            .collect()
    }

    /// The data node's timeout, in whole seconds.
    pub(crate) fn timeout_seconds(&self) -> u64 {
        whole_seconds(self.timeout)
    }

    /// When the data node, polling without jitter, sees a job that ends
    /// at `ends` seconds.
    fn seen(&self, ends: f64) -> f64 {
        let first = FIRST_POLL * self.poll;
        if ends <= first || self.poll <= 0.0 {
            return ends.max(first);
        }
        first + ((ends - first) / self.poll).ceil() * self.poll
    }
}

/// `seconds`, rounded up to a whole number; past `u64::MAX`, that.
fn whole_seconds(seconds: f64) -> u64 {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "non-negative, saturating"
    )]
    let whole = seconds.ceil() as u64;
    whole
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::Gate;
    use crate::cost::{CpuThreads, Fitted};
    use crate::job::tests::job;
    use crate::request::JobId;
    use crate::service::{Device, Refusal, Settings};
    use crate::{InFlight, Job, Stage};

    fn admit(gate: &Gate, job: &Job, in_flight: &[InFlight]) -> Result<f64, Refusal> {
        gate.admit(
            &JobId::from("j"),
            "a/b.knnvec",
            job,
            in_flight,
            Instant::now(),
        )
        .map(|admitted| admitted.ends_in.as_secs_f64())
    }

    #[test]
    fn admits_a_job_forecast_to_end_in_time() {
        let gate = Gate::new(&Settings::default());
        let small = job("bucket", 100_000, 128);
        let stages = Fitted::default().stages(&small);
        let ends = admit(&gate, &small, &[]).unwrap();
        assert!((ends - (stages.download + stages.build + stages.upload)).abs() < 1e-6);
    }

    #[test]
    fn refuses_a_build_the_devices_memory_cannot_hold() {
        let gate = Gate::new(&Settings {
            device: Some(Device {
                free_memory: 1 << 30,
                shared_memory_per_block: 101_376,
            }),
            ..Settings::default()
        });
        // About 3.5 GiB at its peak; about 120 MiB.
        assert!(matches!(
            admit(&gate, &job("bucket", 1_000_000, 768), &[]),
            Err(Refusal::TooLarge { .. })
        ));
        assert!(admit(&gate, &job("bucket", 100_000, 128), &[]).is_ok());
    }

    #[test]
    fn refuses_a_job_forecast_to_end_past_the_timeout() {
        // A 20 s timeout and 5 s polls: the first poll comes by 18 s, and
        // a job ending after 14 s may be seen only after 20 s.
        let gate = Gate::new(&Settings {
            client_timeout: Duration::from_secs(20),
            ..Settings::default()
        });
        let large = job("bucket", 1_000_000, 768);
        let Err(Refusal::Late { seen, timeout, .. }) = admit(&gate, &large, &[]) else {
            panic!("a 1M x 768 job ends in about half a minute")
        };
        assert!((30..70).contains(&seen), "{seen}");
        assert_eq!(timeout, 20);
        let small = job("bucket", 10_000, 128);
        assert!(admit(&gate, &small, &[]).unwrap() < 14.0);
        // Behind jobs that take longer than that, it is late too.
        let now = Instant::now();
        let ahead: Vec<_> = (0..4)
            .map(|id| InFlight {
                id,
                job: job("bucket", 500_000, 384),
                stage: Stage::Queued,
                since: now,
                canceled: false,
            })
            .collect();
        assert!(matches!(
            admit(&gate, &small, &ahead),
            Err(Refusal::Late { .. })
        ));
    }

    #[test]
    fn refuses_a_job_the_data_node_builds_sooner_when_asked() {
        let settings = Settings {
            cpu_build: Some(CpuThreads::Four),
            ..Settings::default()
        };
        let gate = Gate::new(&settings);
        // A small job: built here in under a second, but the data node
        // first asks after 15 s; on its CPU, a fraction of a second.
        let small = job("bucket", 10_000, 128);
        let Err(Refusal::CpuSooner { remote, cpu, .. }) = admit(&gate, &small, &[]) else {
            panic!("the data node builds 10,000 vectors sooner itself")
        };
        assert_eq!((remote, cpu), (16, 1));
        // A large one: back in under a minute, against more than one on
        // the CPU.
        assert!(admit(&gate, &job("bucket", 1_000_000, 768), &[]).is_ok());
        // Unasked, the small one goes in.
        assert!(admit(&Gate::new(&Settings::default()), &small, &[]).is_ok());
    }

    #[test]
    fn refuses_every_job_when_the_first_poll_may_come_too_late() {
        // 14 s, against a first poll from 12 to 18 s in: the data node
        // may give up without asking once, however soon the job ends.
        let gate = Gate::new(&Settings {
            client_timeout: Duration::from_secs(14),
            ..Settings::default()
        });
        assert!(matches!(
            admit(&gate, &job("bucket", 10_000, 128), &[]),
            Err(Refusal::Late { .. })
        ));
        assert_eq!(gate.latest_seen(1.0), 18.0);
        assert_eq!(gate.latest_seen(20.0), 26.0);
    }

    #[test]
    fn sees_a_job_ended_at_the_data_nodes_next_poll() {
        let gate = Gate::new(&Settings::default());
        // Polls at 15 s, then every 5 s.
        assert_eq!(gate.seen(2.0), 15.0);
        assert_eq!(gate.seen(15.0), 15.0);
        assert_eq!(gate.seen(15.5), 20.0);
        assert_eq!(gate.seen(31.0), 35.0);
        let unpolled = Gate::new(&Settings {
            poll_interval: Duration::ZERO,
            ..Settings::default()
        });
        assert_eq!(unpolled.seen(2.5), 2.5);
    }
}
