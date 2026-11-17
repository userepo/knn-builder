//! The projection a job asks for with `project_link_to_base_layer`, run on
//! its graph between the build and the write, as the service runs it in its
//! conversion after the build: `knn_write::projection::project_upper_level`
//! with the service's settings, on a quarter of the processors. A failure
//! leaves the graph as built, with a warning, as the service's does. A
//! cancel stops it between two of its steps or two of layer 1's batches.

use std::num::NonZeroUsize;
use std::sync::OnceLock;
use std::time::Instant;

use knn_store::Cancel;
use knn_write::projection::{Outcome, ProjectionError, project_upper_level};

use crate::{Job, Span, Vectors};

/// The threads layer 1 is built on: a quarter of the processors, at least
/// one, as the service gives Faiss (`get_omp_num_threads`). The service
/// counts the machine's processors; this counts those the process may use,
/// its CPU affinity and cgroup quota applied, so a container limited to
/// fewer processors takes fewer. The rest stay with the downloads, the
/// next build and the uploads.
pub(crate) fn threads() -> NonZeroUsize {
    static THREADS: OnceLock<NonZeroUsize> = OnceLock::new();
    *THREADS.get_or_init(|| {
        let processors = std::thread::available_parallelism().map_or(1, NonZeroUsize::get);
        NonZeroUsize::new(processors / 4).unwrap_or(NonZeroUsize::MIN)
    })
}

/// Projects layer-1 links into `graph`, `degree` ids a vector, when `job`
/// asks, and logs what it did at `DEBUG`, or its failure as a warning.
/// Returns its time when it built layer 1 and wrote its links: not when
/// the gate found the graph connected, too few nodes were promoted, it
/// failed, or `cancel` stopped it, which leaves the graph as built too.
pub(crate) fn project(
    job: &Job,
    vectors: &Vectors,
    graph: &mut [u32],
    degree: usize,
    cancel: &Cancel,
) -> Option<Span> {
    let settings = job.projection(threads())?;
    let start = Instant::now();
    let projected = project_upper_level(
        graph,
        degree,
        vectors.as_write(),
        job.dimension as usize,
        &settings,
        &|| cancel.is_canceled(),
    );
    let ms = |duration: std::time::Duration| duration.as_secs_f64() * 1000.0;
    let vector_path = job.vector_path.as_str();
    let nodes = job.doc_count;
    match projected {
        Ok(Outcome::Connected { reachable, gate }) => {
            tracing::debug!(
                vector_path,
                nodes,
                reachable,
                gate_ms = ms(gate),
                "upper-level projection skipped"
            );
            None
        }
        Ok(Outcome::TooFewPromoted {
            reachable,
            promoted,
            gate,
        }) => {
            tracing::debug!(
                vector_path,
                nodes,
                reachable,
                promoted,
                gate_ms = ms(gate),
                "upper-level projection: nothing to do"
            );
            None
        }
        Ok(Outcome::Projected {
            reachable,
            promoted,
            links,
            gate,
            layer1,
            projection,
        }) => {
            tracing::debug!(
                vector_path,
                nodes,
                reachable,
                promoted,
                links_added = links,
                gate_ms = ms(gate),
                layer1_build_ms = ms(layer1),
                projection_ms = ms(projection),
                "upper-level projection"
            );
            Some(Span::since(start))
        }
        // An outcome a later version reports: its time is not the model's.
        Ok(_) => None,
        Err(ProjectionError::Stopped) => {
            tracing::debug!(
                vector_path,
                "upper-level projection stopped: the job was canceled"
            );
            None
        }
        Err(error) => {
            tracing::warn!(
                vector_path,
                %error,
                "upper-level projection failed, the graph is unchanged"
            );
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use knn_store::Cancel;

    use super::project;
    use crate::job::tests::job;
    use crate::{FakeBuilder, Vectors};

    #[test]
    fn stops_for_a_canceled_job_with_the_graph_as_built() {
        let mut job = job("bucket", 600, 16);
        job.project_link_to_base_layer = true;
        let degree = job.graph_degree().unwrap() as usize;
        // Four clusters of 150 vectors, far apart, as the graph's.
        let built = FakeBuilder::clustered_graph(600, degree, 4);
        let values = (0..600_u16 * 16)
            .map(|i| f32::from(i / 2_400) * 100.0 + f32::from(i % 97) / 97.0)
            .collect();
        let vectors = Vectors::F32(values);

        let canceled = Cancel::new();
        canceled.cancel();
        let mut graph = built.clone();
        assert_eq!(project(&job, &vectors, &mut graph, degree, &canceled), None);
        assert_eq!(graph, built);

        // Not canceled, it projects and reports its time.
        let mut graph = built.clone();
        assert!(project(&job, &vectors, &mut graph, degree, &Cancel::new()).is_some());
        assert_ne!(graph, built);

        // A job that does not ask is left as built.
        job.project_link_to_base_layer = false;
        let mut graph = built.clone();
        assert_eq!(
            project(&job, &vectors, &mut graph, degree, &Cancel::new()),
            None
        );
        assert_eq!(graph, built);
    }
}
