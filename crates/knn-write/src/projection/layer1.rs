//! The HNSW level the projection builds over the promoted nodes, as the
//! service's `build_layer1` builds it: a Faiss `IndexHNSWFlat` of M links a
//! node with every node on one level, filled by `IndexHNSW::add` over
//! float32 copies of the promoted vectors. [`Insertion::Sequential`] is that
//! insertion as Faiss runs it on one thread, where its result is the same
//! every run: a level of 100 nodes or fewer always, and any level where
//! the service gives Faiss one OpenMP thread. On more threads the
//! service's larger levels vary run to run. [`Insertion::Batched`] runs on
//! several threads and still gives one result, Faiss's own for a level of
//! 100 nodes or fewer.
//!
//! Faiss's heaps are `std::priority_queue`s, whose order among equal
//! distances is libstdc++'s heap algorithm; [`Heap`] runs that algorithm,
//! so ties (common for byte vectors) resolve as they do there.

// Every id here is one the build gave a node, below the node count, and
// every row lies within the neighbor table of count x M.
#![expect(
    clippy::indexing_slicing,
    reason = "node ids below the node count, rows within the neighbor table"
)]

use std::any::Any;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, RwLock};
use std::thread::{Scope, ScopedJoinHandle};

use super::{ProjectionError, Stop, kernels};
use crate::Metric;
use crate::mt19937::Mt19937;

/// The seed of the generator that shuffles the insertion order
/// (`hnsw_add_vertices`'s `rng2`).
const INSERTION_SEED: u32 = 789;

/// The share of a full row a prune leaves free (`HNSW::prune_headroom`).
const PRUNE_HEADROOM: f32 = 0.2;

/// An empty slot.
const EMPTY: u32 = u32::MAX;

/// The most nodes Faiss adds on one thread whatever its thread count
/// (`hnsw_add_vertices`'s `omp parallel if (i1 > i0 + 100)`): its level is
/// then the same every run, and the batched one is built as it is.
const ONE_THREAD_NODES: u32 = 100;

/// A batch is at most this share of the nodes already in, so most of what
/// a batch's searches could find is there.
const BATCH_SHARE: usize = 16;

/// The largest batch: enough nodes for many threads, few enough that the
/// nodes of one batch, which do not see each other, stay a small part of a
/// large layer.
const MAX_BATCH: usize = 128;

/// The fewest searches a thread is woken for; for fewer, waking it costs
/// about what it saves.
const SEARCHES_A_THREAD: usize = 4;

/// The fewest rows taking links back a thread is woken for: most take
/// their link into a free slot, at next to no cost.
const ROWS_A_THREAD: usize = 256;

/// What a thread says when it finds the level's lock poisoned: a panic
/// left the level half written, so the build cannot go on.
const POISONED: &str = "the level's lock is poisoned: a thread panicked while changing it";

/// How the level's nodes go in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Insertion {
    /// One at a time in Faiss's order, each searching the level as the
    /// nodes before it left it: Faiss's own level on one thread, entry for
    /// entry.
    Sequential,
    /// In batches, one node at a time while the level is small, then up to
    /// 128 at once, each batch searching the level as the batches before
    /// it left it, on up to `threads` threads. The level is the same
    /// whatever the thread count, and close to the sequential one: the
    /// nodes of a batch do not see each other, as Faiss's own threads see
    /// only part of each other's work. A level of 100 nodes or fewer goes
    /// in as [`Insertion::Sequential`]'s does: Faiss adds one that small
    /// on one thread whatever its thread count, so the level is its own.
    Batched {
        /// The most threads to search and link on, the calling one among
        /// them; the others are started once and kept for every batch. A
        /// batch searches on a thread for every 4 of its nodes, so on 32
        /// at most, and links back on a thread for every 256 rows the
        /// links go into, on no more than search. Each thread holds a
        /// byte for every node.
        threads: NonZeroUsize,
    },
}

/// Builds the level: `vectors`, rows of `dimension` float32 values in
/// promoted order, linked with up to `m` neighbors each at `ef_construction`,
/// as Faiss's `IndexHNSWFlat(dimension, m, metric)` with every node on one
/// level, `efConstruction` set, adds them, in the way `insertion` says.
/// Returns its graph: a row of `m` positions into `vectors` a node, an
/// empty slot as `u32::MAX` (Faiss's -1), as `project_links` takes it.
///
/// # Errors
///
/// Returns a [`ProjectionError`] when the dimension or m is 0, the vectors
/// are not whole rows, there are more of them than Faiss's ids can name,
/// or the metric is Hamming, which the service does not project.
pub fn build_layer1(
    vectors: &[f32],
    dimension: usize,
    metric: Metric,
    m: usize,
    ef_construction: usize,
    insertion: Insertion,
) -> Result<Vec<u32>, ProjectionError> {
    build_layer1_until(
        vectors,
        dimension,
        metric,
        m,
        ef_construction,
        insertion,
        &|| false,
    )
}

/// [`build_layer1`], given up with [`ProjectionError::Stopped`] once `stop`
/// says so: asked before each node in order, and before each batch.
pub(super) fn build_layer1_until(
    vectors: &[f32],
    dimension: usize,
    metric: Metric,
    m: usize,
    ef_construction: usize,
    insertion: Insertion,
    stop: Stop<'_>,
) -> Result<Vec<u32>, ProjectionError> {
    if dimension == 0 || m == 0 || !vectors.len().is_multiple_of(dimension) {
        return Err(ProjectionError::Layer1Input {
            len: vectors.len(),
            dimension,
            m,
        });
    }
    let count = vectors.len() / dimension;
    let n = i32::try_from(count)
        .map(i32::unsigned_abs)
        .map_err(|_| ProjectionError::TooManyNodes { count })?;
    let build = Build {
        storage: Storage {
            vectors,
            dimension,
            metric,
        },
        m,
        ef: ef_construction,
    };
    let neighbors = match metric {
        Metric::L2 => build.run::<Distance>(n, insertion, stop),
        Metric::InnerProduct => build.run::<Similarity>(n, insertion, stop),
        Metric::Hamming => return Err(ProjectionError::Hamming),
    };
    neighbors.ok_or(ProjectionError::Stopped)
}

/// The threads that search at most, each with its table of the nodes seen:
/// one in order, and in batches a thread for every [`SEARCHES_A_THREAD`]
/// of a batch's nodes.
fn searchers(insertion: Insertion) -> usize {
    match insertion {
        Insertion::Sequential => 1,
        Insertion::Batched { threads } => threads.get().min(MAX_BATCH.div_ceil(SEARCHES_A_THREAD)),
    }
}

/// The most host memory [`build_layer1`] takes besides its vectors for
/// `count` nodes, in bytes: its rows of `m` ids, the insertion order, a
/// byte a node for each thread that searches, the search heaps (counted
/// as four times `ef_construction` entries each, itself held to the node
/// count, since a heap holds no node twice), and in batches a batch's
/// targets and the rows it changes, with their copies.
pub(super) fn host_bytes(count: u64, m: u64, ef_construction: u64, insertion: Insertion) -> u64 {
    let searchers = searchers(insertion) as u64;
    let rows = count.saturating_mul(m).saturating_mul(4);
    let order = count.saturating_mul(4);
    let marks = count.saturating_mul(searchers);
    // Two heaps of 8-byte entries, each counted at four times efConstruction.
    let heaps = searchers.saturating_mul(
        ef_construction
            .min(count)
            .saturating_add(m)
            .saturating_add(1)
            .saturating_mul(2 * 4 * 8),
    );
    let batch = match insertion {
        Insertion::Sequential => m.saturating_mul(4),
        Insertion::Batched { .. } => {
            let batch = MAX_BATCH as u64;
            // The targets, a vector of m ids each, and the links back by
            // row: pairs, groups, and each row's copy as a thread returns it.
            batch
                .saturating_mul(m.saturating_mul(4).saturating_add(64))
                .saturating_add(
                    batch
                        .saturating_mul(m)
                        .saturating_mul(m.saturating_mul(4).saturating_add(128)),
                )
        }
    };
    rows.saturating_add(order)
        .saturating_add(marks)
        .saturating_add(heaps)
        .saturating_add(batch)
}

/// Faiss's two comparators (`CMax` for distances, `CMin` for
/// similarities), as the one question HNSW asks of them: whether a score
/// is worse than another (`C::cmp`).
trait Order {
    fn worse(a: f32, b: f32) -> bool;
}

/// Smaller is better: L2.
struct Distance;

impl Order for Distance {
    fn worse(a: f32, b: f32) -> bool {
        a > b
    }
}

/// Larger is better: inner product.
struct Similarity;

impl Order for Similarity {
    fn worse(a: f32, b: f32) -> bool {
        a < b
    }
}

/// A node and its score (`NodeDistCloser`, `NodeDistFarther`).
#[derive(Debug, Clone, Copy)]
struct Node {
    d: f32,
    id: u32,
}

/// `NodeDistCloserT::operator<`: the worst node on top.
fn closer<O: Order>(a: Node, b: Node) -> bool {
    O::worse(b.d, a.d)
}

/// `NodeDistFartherT::operator<`: the best node on top.
fn farther<O: Order>(a: Node, b: Node) -> bool {
    O::worse(a.d, b.d)
}

/// `std::priority_queue` as libstdc++ runs it: `push` is `push_heap`,
/// `pop` is `pop_heap` (`__adjust_heap` down to a leaf, then
/// `__push_heap` back up), so equal elements leave in the same order.
struct Heap {
    items: Vec<Node>,
    less: fn(Node, Node) -> bool,
}

impl Heap {
    fn new(less: fn(Node, Node) -> bool) -> Self {
        Self {
            items: Vec::new(),
            less,
        }
    }

    fn len(&self) -> usize {
        self.items.len()
    }

    fn top(&self) -> Option<Node> {
        self.items.first().copied()
    }

    fn push(&mut self, node: Node) {
        self.items.push(node);
        let hole = self.items.len() - 1;
        self.sift_up(hole, 0, node);
    }

    fn pop(&mut self) {
        let len = self.items.len();
        if len > 1 {
            // __pop_heap: the last element goes into the hole the top
            // leaves, over the first len - 1.
            let value = self.items[len - 1];
            self.items[len - 1] = self.items[0];
            self.adjust(0, len - 1, value);
        }
        self.items.pop();
    }

    /// `__push_heap(first, hole, top, value)`.
    fn sift_up(&mut self, mut hole: usize, top: usize, value: Node) {
        let less = self.less;
        while hole > top {
            let parent = (hole - 1) / 2;
            if !less(self.items[parent], value) {
                break;
            }
            self.items[hole] = self.items[parent];
            hole = parent;
        }
        self.items[hole] = value;
    }

    /// `__adjust_heap(first, hole, len, value)`.
    fn adjust(&mut self, mut hole: usize, len: usize, value: Node) {
        let less = self.less;
        let top = hole;
        let mut child = hole;
        while child < len.saturating_sub(1) / 2 {
            child = 2 * (child + 1);
            if less(self.items[child], self.items[child - 1]) {
                child -= 1;
            }
            self.items[hole] = self.items[child];
            hole = child;
        }
        if len & 1 == 0 && len >= 2 && child == (len - 2) / 2 {
            child = 2 * (child + 1);
            self.items[hole] = self.items[child - 1];
            hole = child - 1;
        }
        self.sift_up(hole, top, value);
    }
}

/// Which nodes a search has seen (`VisitedTableVector`): a byte a node,
/// holding the number of the search that last saw it, all cleared when the
/// numbers run out.
struct Visited {
    marks: Vec<u8>,
    epoch: u8,
}

impl Visited {
    fn new(n: u32) -> Self {
        Self {
            marks: vec![0; n as usize],
            epoch: 1,
        }
    }

    /// Marks `id`; whether it was unmarked.
    fn set(&mut self, id: u32) -> bool {
        let mark = &mut self.marks[id as usize];
        if *mark == self.epoch {
            return false;
        }
        *mark = self.epoch;
        true
    }

    /// Unmarks every node.
    fn advance(&mut self) {
        if self.epoch < u8::MAX {
            self.epoch += 1;
        } else {
            self.marks.fill(0);
            self.epoch = 1;
        }
    }
}

/// The flat storage's distance computer (`FlatL2Dis`, `FlatIPDis`).
struct Storage<'a> {
    vectors: &'a [f32],
    dimension: usize,
    metric: Metric,
}

impl Storage<'_> {
    fn row(&self, id: u32) -> &[f32] {
        let start = id as usize * self.dimension;
        &self.vectors[start..start + self.dimension]
    }

    /// The query's score against `id` (`operator()`).
    fn score(&self, query: &[f32], id: u32) -> f32 {
        match self.metric {
            Metric::InnerProduct => kernels::inner_product(query, self.row(id)),
            Metric::L2 | Metric::Hamming => kernels::l2(query, self.row(id)),
        }
    }

    /// Four scores at once (`distances_batch_4`).
    fn scores_4(&self, query: &[f32], ids: [u32; 4]) -> [f32; 4] {
        // Written out, as the kernels' four sums are: `map` is not always
        // inlined here.
        let [id0, id1, id2, id3] = ids;
        let rows = [self.row(id0), self.row(id1), self.row(id2), self.row(id3)];
        match self.metric {
            Metric::InnerProduct => kernels::inner_product_batch_4(query, rows),
            Metric::L2 | Metric::Hamming => kernels::l2_batch_4(query, rows),
        }
    }

    /// Two stored vectors' score (`symmetric_dis(i, j)`, which passes `j`
    /// first).
    fn symmetric(&self, i: u32, j: u32) -> f32 {
        match self.metric {
            Metric::InnerProduct => kernels::inner_product(self.row(j), self.row(i)),
            Metric::L2 | Metric::Hamming => kernels::l2(self.row(j), self.row(i)),
        }
    }
}

/// The build's fixed parts.
struct Build<'a> {
    storage: Storage<'a>,
    m: usize,
    ef: usize,
}

impl Build<'_> {
    /// `hnsw_add_vertices` for nodes 0..n on one level; `None` once `stop`
    /// says to give up.
    fn run<O: Order>(&self, n: u32, insertion: Insertion, stop: Stop<'_>) -> Option<Vec<u32>> {
        let mut neighbors = vec![EMPTY; n as usize * self.m];
        let order = insertion_order(n);
        // The first node becomes the entry point and links to nothing; the
        // level's only, it stays the entry point.
        let Some((&entry, rest)) = order.split_first() else {
            return Some(neighbors);
        };
        match insertion {
            Insertion::Batched { threads } if n > ONE_THREAD_NODES => {
                self.batched::<O>(&mut neighbors, n, entry, rest, threads, stop)?;
            }
            Insertion::Sequential | Insertion::Batched { .. } => {
                let mut visited = Visited::new(n);
                for &pt in rest {
                    if stop() {
                        return None;
                    }
                    let targets = self.targets::<O>(&neighbors, &mut visited, pt, entry);
                    self.link::<O>(&mut neighbors, pt, &targets);
                }
            }
        }
        Some(neighbors)
    }

    /// The nodes `pt` links to, in the order `add_links_starting_from`
    /// adds them: the search's results pruned to M, worst first.
    fn targets<O: Order>(
        &self,
        neighbors: &[u32],
        visited: &mut Visited,
        pt: u32,
        entry: u32,
    ) -> Vec<u32> {
        let query = self.storage.row(pt);
        let d_entry = self.storage.score(query, entry);
        let mut results = Heap::new(closer::<O>);
        self.search::<O>(neighbors, visited, query, entry, d_entry, &mut results);
        self.shrink_heap::<O>(&mut results, self.m);
        let mut targets = Vec::with_capacity(results.len());
        while let Some(top) = results.top() {
            targets.push(top.id);
            results.pop();
        }
        targets
    }

    /// Links `pt` to `targets` and each of them back to `pt`, as
    /// `add_links_starting_from` does.
    fn link<O: Order>(&self, neighbors: &mut [u32], pt: u32, targets: &[u32]) {
        for &other in targets {
            self.add_link::<O>(&mut neighbors[self.row(pt)], pt, other);
        }
        for &other in targets {
            self.add_link::<O>(&mut neighbors[self.row(other)], other, pt);
        }
    }

    fn row(&self, id: u32) -> std::ops::Range<usize> {
        let start = id as usize * self.m;
        start..start + self.m
    }

    /// The nodes after the entry point in batches: each batch's nodes find
    /// their targets in the graph as it stood before the batch, on up to
    /// `threads` threads, then link in batch order. A node's targets depend
    /// on that graph alone, and every row a batch changes is a batch
    /// node's own or a target's, never both (no row links to a batch node
    /// before its batch links), so the target rows can change in parallel
    /// too, each taking its links in batch order: the result is the same
    /// whatever the thread count. The threads are started once and kept
    /// for every batch, which wakes those it has work for. `None` once
    /// `stop`, asked before each batch, says to give up.
    fn batched<O: Order>(
        &self,
        neighbors: &mut [u32],
        n: u32,
        entry: u32,
        rest: &[u32],
        threads: NonZeroUsize,
        stop: Stop<'_>,
    ) -> Option<()> {
        // No batch is larger than the size the whole count gives, so none
        // searches on more threads than that one would. Its links back
        // could use more only at an m past 64 with hardly a link pruned,
        // so they get the same threads and none of their own.
        let most = batch_size(rest.len())
            .div_ceil(SEARCHES_A_THREAD)
            .min(searchers(Insertion::Batched { threads }));
        let batches = Batches {
            build: self,
            neighbors: RwLock::new(neighbors),
            rest,
            entry,
        };
        let work = |step: &Step, visited: &mut Visited, next: &AtomicUsize| {
            batches.share::<O>(step, visited, next)
        };
        std::thread::scope(|scope| {
            // Each thread has a table of the nodes its search has seen, the
            // calling one too.
            let mut visited = Visited::new(n);
            let mut crew = Crew::hire(scope, &work, (1..most).map(|_| Visited::new(n)));
            let mut inserted = 1;
            let mut start = 0;
            while start < rest.len() {
                if stop() {
                    return None;
                }
                let size = batch_size(inserted).min(rest.len() - start);
                let places = start..start + size;
                let batch = &rest[places.clone()];
                let targets = crew.run(
                    &Step::Search(places),
                    size.div_ceil(SEARCHES_A_THREAD),
                    &mut visited,
                );
                {
                    // Each batch node's own row, empty until now.
                    let mut neighbors = batches.neighbors.write().expect(POISONED);
                    for (&pt, targets) in batch.iter().zip(&targets) {
                        for &other in targets {
                            self.add_link::<O>(&mut neighbors[self.row(pt)], pt, other);
                        }
                    }
                }
                let back = Arc::new(Back::new(batch, &targets));
                let rows = crew.run(
                    &Step::Link(Arc::clone(&back)),
                    back.rows.len().div_ceil(ROWS_A_THREAD),
                    &mut visited,
                );
                {
                    let mut neighbors = batches.neighbors.write().expect(POISONED);
                    for ((other, _), row) in back.rows.iter().zip(&rows) {
                        neighbors[self.row(*other)].copy_from_slice(row);
                    }
                }
                inserted += size;
                start += size;
            }
            Some(())
        })
    }

    /// `search_neighbors_to_add`, its optimized form: neighbors' scores
    /// four at a time, the rest one by one.
    fn search<O: Order>(
        &self,
        neighbors: &[u32],
        visited: &mut Visited,
        query: &[f32],
        entry: u32,
        d_entry: f32,
        results: &mut Heap,
    ) {
        let mut candidates = Heap::new(farther::<O>);
        let start = Node {
            d: d_entry,
            id: entry,
        };
        candidates.push(start);
        results.push(start);
        visited.set(entry);

        let ef = self.ef;
        let update = |results: &mut Heap, candidates: &mut Heap, id: u32, d: f32| {
            if results.len() < ef || results.top().is_some_and(|top| O::worse(top.d, d)) {
                results.push(Node { d, id });
                candidates.push(Node { d, id });
                if results.len() > ef {
                    results.pop();
                }
            }
        };

        while let Some(current) = candidates.top() {
            if results.top().is_some_and(|top| O::worse(current.d, top.d)) {
                break;
            }
            candidates.pop();
            let mut buffered = [0_u32; 4];
            let mut count = 0;
            for &id in &neighbors[self.row(current.id)] {
                if id == EMPTY {
                    break;
                }
                if !visited.set(id) {
                    continue;
                }
                buffered[count] = id;
                count += 1;
                if count == 4 {
                    let scores = self.storage.scores_4(query, buffered);
                    for (&id, d) in buffered.iter().zip(scores) {
                        update(results, &mut candidates, id, d);
                    }
                    count = 0;
                }
            }
            for &id in &buffered[..count] {
                let d = self.storage.score(query, id);
                update(results, &mut candidates, id, d);
            }
        }
        visited.advance();
    }

    /// `shrink_neighbor_list_inner`: keeps at most `max_size` of `heap`
    /// by Faiss's heuristic, unless it holds fewer.
    fn shrink_heap<O: Order>(&self, heap: &mut Heap, max_size: usize) {
        if heap.len() < max_size {
            return;
        }
        let mut input = Heap::new(farther::<O>);
        while let Some(top) = heap.top() {
            input.push(top);
            heap.pop();
        }
        let mut output: Vec<Node> = Vec::new();
        // shrink_neighbor_list: best first, a node kept unless a node
        // already kept scores better with it than the query does.
        while let Some(v1) = input.top() {
            input.pop();
            let good = output
                .iter()
                .all(|v2| !O::worse(v1.d, self.storage.symmetric(v2.id, v1.id)));
            if good {
                output.push(v1);
                if output.len() >= max_size {
                    break;
                }
            }
        }
        for node in output {
            heap.push(node);
        }
    }

    /// `add_link(src, dest)` on `src`'s row: into the first empty slot
    /// after the last link, or, the row full, the row and `dest` pruned to
    /// the row less its headroom, written back worst first.
    fn add_link<O: Order>(&self, row: &mut [u32], src: u32, dest: u32) {
        if row.last() == Some(&EMPTY) {
            let mut i = row.len();
            while i > 0 && row[i - 1] == EMPTY {
                i -= 1;
            }
            row[i] = dest;
            return;
        }
        let mut kept = Heap::new(closer::<O>);
        kept.push(Node {
            d: self.storage.symmetric(src, dest),
            id: dest,
        });
        for &neigh in &*row {
            kept.push(Node {
                d: self.storage.symmetric(src, neigh),
                id: neigh,
            });
        }
        self.shrink_heap::<O>(&mut kept, pruned_size(row.len()));
        let mut i = 0;
        while let Some(top) = kept.top() {
            row[i] = top.id;
            i += 1;
            kept.pop();
        }
        row[i..].fill(EMPTY);
    }
}

/// The insertion order: nodes 0..n shuffled by Faiss's permutation "to get
/// rid of dataset order bias".
fn insertion_order(n: u32) -> Vec<u32> {
    let mut order: Vec<u32> = (0..n).collect();
    let mut rng = Mt19937::new(INSERTION_SEED);
    for j in 0..order.len() {
        let span = (order.len() - j) as u64;
        let k = j + usize::try_from(u64::from(rng.next_u32()) % span).unwrap_or(0);
        order.swap(j, k);
    }
    order
}

/// A batch's size once `inserted` nodes are in: one at a time until twice
/// [`BATCH_SHARE`] are, then a [`BATCH_SHARE`]th of them, at most
/// [`MAX_BATCH`]. It depends on the count alone, so the level does too.
fn batch_size(inserted: usize) -> usize {
    (inserted / BATCH_SHARE).clamp(1, MAX_BATCH)
}

/// What a batch asks of the threads it wakes.
#[derive(Clone)]
enum Step {
    /// The targets of the nodes at these places of the insertion order
    /// after the entry point.
    Search(Range<usize>),
    /// The rows a batch's links go back into, each with its links added.
    Link(Arc<Back>),
}

/// A batch's links back, by the row they go into.
struct Back {
    /// Each link as the row it goes into and the batch node it adds,
    /// sorted by row, a row's in batch order.
    links: Vec<(u32, u32)>,
    /// Each row that takes links, and where they lie in `links`.
    rows: Vec<(u32, Range<usize>)>,
}

impl Back {
    /// The links back of `batch`, each node of it linked to its `targets`.
    fn new(batch: &[u32], targets: &[Vec<u32>]) -> Self {
        let mut links: Vec<(u32, u32)> = batch
            .iter()
            .zip(targets)
            .flat_map(|(&pt, targets)| targets.iter().map(move |&other| (other, pt)))
            .collect();
        // The stable sort keeps each row's links in batch order.
        links.sort_by_key(|&(other, _)| other);
        let mut rows = Vec::new();
        let mut start = 0;
        for group in links.chunk_by(|a, b| a.0 == b.0) {
            // chunk_by yields no empty group.
            if let Some(&(other, _)) = group.first() {
                rows.push((other, start..start + group.len()));
            }
            start += group.len();
        }
        Self { links, rows }
    }
}

/// What the threads of a batched build share: the level as the batches so
/// far left it, which a step's threads read and the calling thread changes
/// between steps.
struct Batches<'a> {
    build: &'a Build<'a>,
    neighbors: RwLock<&'a mut [u32]>,
    /// The insertion order after the entry point.
    rest: &'a [u32],
    entry: u32,
}

impl Batches<'_> {
    /// One thread's share of `step`: each item it took from `next`, with
    /// the targets it found for the node or the row as its links leave it.
    fn share<O: Order>(
        &self,
        step: &Step,
        visited: &mut Visited,
        next: &AtomicUsize,
    ) -> Share<Vec<u32>> {
        let neighbors = self.neighbors.read().expect(POISONED);
        let neighbors: &[u32] = &neighbors;
        let mut share = Share::new();
        match step {
            Step::Search(places) => {
                let batch = &self.rest[places.clone()];
                while let Some((i, &pt)) = take(batch, next) {
                    let targets = self.build.targets::<O>(neighbors, visited, pt, self.entry);
                    share.push((i, targets));
                }
            }
            Step::Link(back) => {
                while let Some((i, (other, links))) = take(&back.rows, next) {
                    let mut row = neighbors[self.build.row(*other)].to_vec();
                    for &(_, pt) in &back.links[links.clone()] {
                        self.build.add_link::<O>(&mut row, *other, pt);
                    }
                    share.push((i, row));
                }
            }
        }
        share
    }
}

/// The next of `items` no thread has taken from `next`, with its place:
/// each thread takes one at a time, so a slow item holds up only its own
/// thread.
fn take<'a, T>(items: &'a [T], next: &AtomicUsize) -> Option<(usize, &'a T)> {
    let i = next.fetch_add(1, Ordering::Relaxed);
    items.get(i).map(|item| (i, item))
}

/// One thread's part of a step: what it made of each item it took, with
/// the item's place among the step's.
type Share<R> = Vec<(usize, R)>;

/// The threads kept for a build's steps besides the calling one, so that a
/// step wakes the threads it would otherwise have to start. Each has a
/// worker of its own (in a build, its table of the nodes seen) and takes
/// its share of every step sent to it, as the calling thread takes its
/// own, until the crew is dropped.
struct Crew<'scope, S, R, F> {
    /// A thread's share of a step, taken from the step's counter.
    work: &'scope F,
    hands: Vec<Hand<'scope, S, R>>,
}

/// One of a crew's threads: the steps go to it and its shares come back.
struct Hand<'scope, S, R> {
    steps: Sender<(S, Arc<AtomicUsize>)>,
    shares: Receiver<Share<R>>,
    thread: ScopedJoinHandle<'scope, ()>,
}

impl<'scope, S: Clone + Send + 'scope, R: Send + 'scope, F> Crew<'scope, S, R, F> {
    /// Starts a thread for each of `workers`, which it keeps; `work` is a
    /// thread's share of a step, taken from the step's counter.
    fn hire<'env, W>(
        scope: &'scope Scope<'scope, 'env>,
        work: &'env F,
        workers: impl IntoIterator<Item = W>,
    ) -> Self
    where
        W: Send + 'scope,
        F: Fn(&S, &mut W, &AtomicUsize) -> Share<R> + Sync,
    {
        let hands = workers
            .into_iter()
            .map(|mut worker| {
                let (steps, asked) = channel::<(S, Arc<AtomicUsize>)>();
                let (answers, shares) = channel();
                let thread = scope.spawn(move || {
                    for (step, next) in asked {
                        // The calling thread waits for each share it asked
                        // for, until it gives the build up.
                        if answers.send(work(&step, &mut worker, &next)).is_err() {
                            return;
                        }
                    }
                });
                Hand {
                    steps,
                    shares,
                    thread,
                }
            })
            .collect();
        Self { work, hands }
    }

    /// `step` on up to `threads` threads, the calling one with `own` among
    /// them and the rest the crew's first: every item's result, in the
    /// items' order, whichever thread took it.
    fn run<W>(&mut self, step: &S, threads: usize, own: &mut W) -> Vec<R>
    where
        F: Fn(&S, &mut W, &AtomicUsize) -> Share<R>,
    {
        let next = Arc::new(AtomicUsize::new(0));
        let asked = threads.saturating_sub(1).min(self.hands.len());
        for hand in self.hands.iter().take(asked) {
            // A thread that is gone has panicked: seen below, where its
            // share is missing.
            let _gone = hand.steps.send((step.clone(), Arc::clone(&next)));
        }
        let mut done = (self.work)(step, own, &next);
        for i in 0..asked {
            let Some(hand) = self.hands.get(i) else {
                break;
            };
            match hand.shares.recv() {
                Ok(share) => done.extend(share),
                Err(_) => carry_panic(self.hands.swap_remove(i).thread),
            }
        }
        // Each item was taken once.
        done.sort_unstable_by_key(|&(i, _)| i);
        done.into_iter().map(|(_, result)| result).collect()
    }
}

/// Ends the build as the panic of a crew's thread, which left without its
/// share, would have ended it without threads: a thread only computes, so
/// its panic is a bug, carried to the caller's thread.
fn carry_panic(thread: ScopedJoinHandle<'_, ()>) -> ! {
    let payload: Box<dyn Any + Send> = match thread.join() {
        Err(payload) => payload,
        Ok(()) => Box::new("a thread of the level's build left without its share"),
    };
    std::panic::resume_unwind(payload)
}

/// `max_size -= max_size * prune_headroom`: the row's size less the
/// headroom, in float32 as C++ converts it, truncated back.
fn pruned_size(row: usize) -> usize {
    #[expect(clippy::cast_precision_loss, reason = "a row's size, far below 2^24")]
    let size = row as f32;
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "C++'s float to size_t conversion of a value from 0 to the row's size"
    )]
    let pruned = (size - size * PRUNE_HEADROOM) as usize;
    pruned
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::num::NonZeroUsize;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex, RwLock};
    use std::time::{Duration, Instant};

    use super::{
        Back, Batches, Build, Crew, Distance, EMPTY, Heap, Insertion, Node, Share, Step, Storage,
        Visited, build_layer1, closer, pruned_size, take,
    };
    use crate::Metric;
    use crate::projection::ProjectionError;

    #[test]
    fn prunes_to_the_row_less_its_headroom() {
        // `max_size -= max_size * 0.2f` as g++ 13 computes it on size_t.
        for (row, pruned) in [
            (1, 0),
            (2, 1),
            (3, 2),
            (4, 3),
            (5, 4),
            (8, 6),
            (10, 8),
            (12, 9),
            (15, 12),
            (16, 12),
            (24, 19),
            (31, 24),
            (32, 25),
            (48, 38),
            (64, 51),
            (100, 80),
            (128, 102),
            (512, 409),
            (2048, 1638),
        ] {
            assert_eq!(pruned_size(row), pruned, "{row}");
        }
    }

    /// `ops` pushes and pops on a max-heap of scores, as the test program
    /// that recorded libstdc++'s order makes them: a 64-bit LCG (Knuth's
    /// MMIX constants, seed 42), its draw's top 31 bits r; a pop where r is
    /// a multiple of 3 and the heap is not empty, else a push of score r % 4
    /// with the operation's number as its id; then pops to the end.
    fn popped(ops: u32) -> Vec<u32> {
        let mut heap = Heap::new(closer::<Distance>);
        let mut state: u64 = 42;
        let mut out = Vec::new();
        for i in 0..ops {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let r = state >> 33;
            if !r.is_multiple_of(3) || heap.len() == 0 {
                #[expect(clippy::cast_precision_loss, reason = "0 to 3")]
                heap.push(Node {
                    d: (r % 4) as f32,
                    id: i,
                });
            } else {
                out.push(heap.top().unwrap().id);
                heap.pop();
            }
        }
        while let Some(top) = heap.top() {
            out.push(top.id);
            heap.pop();
        }
        out
    }

    #[test]
    fn scores_with_the_kernels_faiss_s_storage_uses() {
        // Inputs where the one-at-a-time and the four-at-a-time sums
        // differ, and what the service's Faiss returns for them from an
        // IndexHNSWFlat's storage: its one-at-a-time score and
        // symmetric_dis sum as the single kernels do, its
        // distances_batch_4 as the four-at-a-time ones.
        let l2 = [
            4096.0_f32, 1.0, 0.0, 0.0, 1.0, 1.0, //
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let storage = Storage {
            vectors: &l2,
            dimension: 6,
            metric: Metric::L2,
        };
        let query = storage.row(0);
        assert_eq!(storage.score(query, 1), 16_777_218.0);
        assert_eq!(storage.scores_4(query, [1; 4]), [16_777_216.0; 4]);
        assert_eq!(storage.symmetric(0, 1), 16_777_218.0);

        let ip = [
            1e8_f32, -1e8, 0.0, 0.0, 1.0, 1.0, 1.0, //
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ];
        let storage = Storage {
            vectors: &ip,
            dimension: 7,
            metric: Metric::InnerProduct,
        };
        let query = storage.row(0);
        assert_eq!(storage.score(query, 1), 3.0);
        assert_eq!(storage.scores_4(query, [1; 4]), [1.0; 4]);
        assert_eq!(storage.symmetric(0, 1), 3.0);
    }

    /// A step for the crew's tests: the squares of `items`, each thread's
    /// first of them waiting until `together` threads have one.
    #[derive(Clone)]
    struct Squares {
        items: Arc<Vec<u32>>,
        started: Arc<AtomicUsize>,
        together: usize,
    }

    impl Squares {
        fn of(items: &[u32], together: usize) -> Self {
            Self {
                items: Arc::new(items.to_vec()),
                started: Arc::new(AtomicUsize::new(0)),
                together,
            }
        }
    }

    #[test]
    fn shares_each_step_among_the_threads_it_keeps() {
        let items: Vec<u32> = (0..1000).collect();
        let squares: Vec<u32> = items.iter().map(|x| x * x).collect();
        // What each worker took, the calling thread's first, and the
        // threads that worked.
        let runs: Vec<AtomicUsize> = (0..7).map(|_| AtomicUsize::new(0)).collect();
        let threads = Mutex::new(HashSet::new());
        let taken = || -> Vec<usize> { runs.iter().map(|r| r.load(Ordering::SeqCst)).collect() };
        // A thread's first item waits until the step's threads all have
        // one, so they take turns; after ten seconds it goes on, so too few
        // threads fail below instead of hanging.
        let work = |step: &Squares, worker: &mut usize, next: &AtomicUsize| {
            threads.lock().unwrap().insert(std::thread::current().id());
            let mut share = Share::new();
            while let Some((i, &x)) = take(&step.items, next) {
                if share.is_empty() {
                    step.started.fetch_add(1, Ordering::SeqCst);
                    let deadline = Instant::now() + Duration::from_secs(10);
                    while step.started.load(Ordering::SeqCst) < step.together
                        && Instant::now() < deadline
                    {
                        std::thread::yield_now();
                    }
                }
                runs[*worker].fetch_add(1, Ordering::SeqCst);
                share.push((i, x * x));
            }
            share
        };
        std::thread::scope(|scope| {
            let mut crew = Crew::hire(scope, &work, 1..7_usize);
            let mut own = 0;
            // Two steps on all seven: the same seven threads both times,
            // asked for exactly or for more than there are.
            for asked in [7, 100] {
                let got = crew.run(&Squares::of(&items, 7), asked, &mut own);
                assert_eq!(got, squares);
            }
            assert_eq!(taken().iter().sum::<usize>(), 2000);
            assert!(taken().iter().all(|&r| r > 1), "{:?}", taken());
            assert_eq!(threads.lock().unwrap().len(), 7);
            // On three: the calling thread and the crew's first two.
            let before = taken();
            let got = crew.run(&Squares::of(&items, 3), 3, &mut own);
            assert_eq!(got, squares);
            let took: Vec<usize> = taken().iter().zip(&before).map(|(a, b)| a - b).collect();
            assert!(took[..3].iter().all(|&r| r > 0), "{took:?}");
            assert_eq!(took[3..], [0; 4]);
            // On one, or on none: the calling thread alone.
            for asked in [1, 0] {
                let before = taken();
                let got = crew.run(&Squares::of(&items[..4], 1), asked, &mut own);
                assert_eq!(got, [0, 1, 4, 9]);
                let took: Vec<usize> = taken().iter().zip(&before).map(|(a, b)| a - b).collect();
                assert_eq!(took, [4, 0, 0, 0, 0, 0, 0]);
            }
        });
        // A crew of none leaves every step to the calling thread.
        std::thread::scope(|scope| {
            let mut crew = Crew::hire(scope, &work, std::iter::empty());
            let got = crew.run(&Squares::of(&items, 1), 7, &mut 0);
            assert_eq!(got, squares);
        });
        assert_eq!(threads.lock().unwrap().len(), 7);
    }

    #[test]
    #[should_panic(expected = "the kept thread's own words")]
    fn carries_a_kept_thread_s_panic_to_the_caller() {
        let work = |(): &(), worker: &mut usize, _: &AtomicUsize| -> Share<u32> {
            assert_eq!(*worker, 0, "the kept thread's own words");
            Share::new()
        };
        std::thread::scope(|scope| {
            let mut crew = Crew::hire(scope, &work, [1_usize]);
            crew.run(&(), 2, &mut 0);
        });
    }

    #[test]
    fn takes_its_share_of_a_search_and_of_the_links_back() {
        // Four nodes on a line, none linked yet: each of the last three
        // finds the entry point alone.
        let vectors = [0.0_f32, 1.0, 2.0, 3.0];
        let build = Build {
            storage: Storage {
                vectors: &vectors,
                dimension: 1,
                metric: Metric::L2,
            },
            m: 2,
            ef: 4,
        };
        let mut neighbors = [EMPTY; 8];
        let batches = Batches {
            build: &build,
            neighbors: RwLock::new(&mut neighbors),
            rest: &[1, 2, 3],
            entry: 0,
        };
        let step = Step::Search(0..3);
        let next = AtomicUsize::new(0);
        let mut visited = Visited::new(4);
        let targets = batches.share::<Distance>(&step, &mut visited, &next);
        assert_eq!(targets, [(0, vec![0]), (1, vec![0]), (2, vec![0])]);
        // Nothing is left of the step for a thread that comes after.
        assert_eq!(batches.share::<Distance>(&step, &mut visited, &next), []);
        // The three links back all go into the entry point's row, which
        // holds two: it keeps the nearest.
        let targets: Vec<Vec<u32>> = targets.into_iter().map(|(_, found)| found).collect();
        let step = Step::Link(Arc::new(Back::new(batches.rest, &targets)));
        let linked = batches.share::<Distance>(&step, &mut visited, &AtomicUsize::new(0));
        assert_eq!(linked, [(0, vec![1, EMPTY])]);
    }

    /// `count` vectors of `dimension` values from 0 to 1, from a 64-bit LCG
    /// (Knuth's MMIX constants).
    fn uniform(count: usize, dimension: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..count * dimension)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                #[expect(clippy::cast_precision_loss, reason = "24 bits")]
                let value = (state >> 40) as f32 / 16_777_216.0;
                value
            })
            .collect()
    }

    #[test]
    fn builds_one_level_on_more_threads_than_it_keeps() {
        // A batch of 128 searches on 32 threads at most: given more, the
        // build keeps 31 beside its own, and the level is the one a single
        // thread builds. At m 6 a third of these rows end full, so many a
        // link back prunes its row.
        let (count, dimension, m) = (2_400, 4, 6);
        let vectors = uniform(count, dimension, 7);
        let on = |threads: usize| {
            let insertion = Insertion::Batched {
                threads: NonZeroUsize::new(threads).unwrap(),
            };
            build_layer1(&vectors, dimension, Metric::L2, m, 80, insertion).unwrap()
        };
        let one = on(1);
        let full = one.chunks(m).filter(|row| !row.contains(&EMPTY)).count();
        assert!(full > count / 4, "{full} full rows");
        for threads in [33, 40, 64] {
            assert_eq!(on(threads), one, "{threads} threads");
        }
    }

    #[test]
    fn builds_a_level_faiss_adds_on_one_thread_as_faiss_does() {
        // Faiss adds 100 nodes or fewer on one thread, in order; from 101
        // its threads share them, and the batches stand in for those.
        let (dimension, m) = (4, 6);
        let batched = Insertion::Batched {
            threads: NonZeroUsize::new(4).unwrap(),
        };
        let level = |count: usize, insertion| {
            let vectors = uniform(count, dimension, 7);
            build_layer1(&vectors, dimension, Metric::L2, m, 80, insertion).unwrap()
        };
        for count in [2, 33, 64, 100] {
            assert_eq!(
                level(count, batched),
                level(count, Insertion::Sequential),
                "{count} nodes"
            );
        }
        assert_ne!(level(101, batched), level(101, Insertion::Sequential));
    }

    #[test]
    fn forgets_a_search_s_marks_however_many_searches_follow() {
        for searches in 1..600 {
            let mut visited = Visited::new(2);
            assert!(visited.set(0));
            assert!(!visited.set(0));
            for _ in 0..searches {
                visited.advance();
            }
            assert!(visited.set(0), "{searches}");
        }
    }

    #[test]
    fn breaks_ties_as_libstdcxx_does() {
        // std::priority_queue under g++ 13 (libstdc++): six equal scores
        // pushed in order leave as 0 2 5 4 1 3.
        let mut heap = Heap::new(closer::<Distance>);
        for id in 0..6 {
            heap.push(Node { d: 1.0, id });
        }
        let mut order = Vec::new();
        while let Some(top) = heap.top() {
            order.push(top.id);
            heap.pop();
        }
        assert_eq!(order, [0, 2, 5, 4, 1, 3]);
        // And 400 operations over four scores, popped as libstdc++ pops
        // them.
        let expected: Vec<u32> = "
        0 1 4 17 23 2 16 8 11 13 26 20 37 43 44 41 51 55 60 53 48 66 70 38 9 78 82 83
        92 93 80 79 81 88 95 89 109 90 126 128 132 142 111 125 137 150 154 155 143 164
        166 171 181 185 186 204 206 178 129 136 212 218 196 115 179 228 232 234 182 236
        226 231 251 120 146 173 256 261 263 265 266 267 271 273 259 118 114 169 165 296
        294 297 168 133 298 308 310 320 316 302 305 326 325 329 311 315 318 282 339 343
        344 347 348 355 349 350 361 370 362 364 368 377 379 389 337 6 393 395 394 397
        334 386 269 140 279 292 284 385 247 277 290 249 275 378 253 366 365 363 101 42
        46 85 215 151 235 233 246 141 207 145 131 62 227 123 56 223 198 191 381 396 18
        188 183 184 45 112 52 175 372 367 174 333 278 99 335 202 199 176 87 69 281 286
        158 107 39 15 177 119 54 319 354 387 152 341 76 103 163 138 110 342 229 117 357
        102 5 58 68 300 10 383 97 241 276 205 262 71 74 170 122 291 353 156 195 194 59
        127 244 309 180 172 14 384 360 116 192 230 32 121 312 380 19 258 242 293 219 24
        33 208 399 224 193 314 72 139 239 374 190 161 21 35 157 203 200
        "
        .split_whitespace()
        .map(|id| id.parse().unwrap())
        .collect();
        assert_eq!(popped(400), expected);
    }

    #[test]
    fn refuses_what_faiss_would_not_build() {
        let bad = ProjectionError::Layer1Input {
            len: 6,
            dimension: 4,
            m: 2,
        };
        assert_eq!(
            build_layer1(&[0.0; 6], 4, Metric::L2, 2, 10, Insertion::Sequential),
            Err(bad)
        );
        assert!(matches!(
            build_layer1(&[0.0; 8], 0, Metric::L2, 2, 10, Insertion::Sequential),
            Err(ProjectionError::Layer1Input { .. })
        ));
        assert!(matches!(
            build_layer1(&[0.0; 8], 4, Metric::L2, 0, 10, Insertion::Sequential),
            Err(ProjectionError::Layer1Input { .. })
        ));
        assert_eq!(
            build_layer1(&[0.0; 8], 4, Metric::Hamming, 2, 10, Insertion::Sequential),
            Err(ProjectionError::Hamming)
        );
        // No vectors, no graph; one, a row of empty slots.
        assert_eq!(
            build_layer1(&[], 4, Metric::L2, 2, 10, Insertion::Sequential),
            Ok(vec![])
        );
        assert_eq!(
            build_layer1(
                &[1.0; 4],
                4,
                Metric::InnerProduct,
                2,
                10,
                Insertion::Sequential
            ),
            Ok(vec![u32::MAX; 2])
        );
    }
}
