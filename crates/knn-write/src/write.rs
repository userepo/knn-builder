//! The file: an `IxMp` ID map over an `IHc2` graph and its storage, or
//! for binary vectors an `IBMp` ID map over an `IBHc` graph and an `IBxF`
//! storage, written in file order. See `docs/faiss-layout.md` for every
//! field.

use std::fmt;
use std::io::{self, BufWriter, Write};
use std::iter;

use crate::Error;
use crate::levels::{LevelTable, NodeLevel};

/// `write_index_header`'s two placeholders (`impl/index_write.cpp:100`).
const HEADER_DUMMY: i64 = 1 << 20;
/// A header's bytes after its fourcc: `d`, `ntotal`, two dummies,
/// `is_trained`, `metric_type`.
const HEADER_LEN: u64 = 4 + 8 + 8 + 8 + 1 + 4;
/// A binary index's header after its fourcc (`write_index_binary_header`):
/// `d`, `code_size`, `ntotal`, `is_trained`, `metric_type`.
const BINARY_HEADER_LEN: u64 = 4 + 4 + 8 + 1 + 4;
/// `IndexHNSWCagra::num_base_level_search_entrypoints`, at its default.
const BASE_LEVEL_ENTRY_POINTS: i32 = 32;
/// `ScalarQuantizer::QT_fp16`.
const QT_FP16: i32 = 4;
/// `ScalarQuantizer::QT_8bit_direct_signed`.
const QT_8BIT_DIRECT_SIGNED: i32 = 8;
/// Faiss's `NumericType`.
const NUMERIC_FLOAT32: i32 = 0;
const NUMERIC_FLOAT16: i32 = 1;
const NUMERIC_INT8: i32 = 3;
/// Bytes gathered before each write to the sink.
const CHUNK: usize = 1 << 16;

/// The field's space type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    /// `l2`.
    L2,
    /// `innerproduct`.
    InnerProduct,
    /// `hamming`, binary vectors' space, and theirs alone.
    Hamming,
}

impl Metric {
    /// Faiss's `MetricType`: a binary index records `METRIC_L2`, the
    /// default it never changes, whatever its distance.
    fn faiss(self) -> i32 {
        match self {
            Self::InnerProduct => 0,
            Self::L2 | Self::Hamming => 1,
        }
    }

    /// The flat storage's fourcc of float, fp16 or byte vectors. Binary
    /// vectors, the only ones measured in Hamming distance, write `IBxF`
    /// without it, so the Hamming arm never runs: it keeps the match whole.
    fn flat_fourcc(self) -> [u8; 4] {
        match self {
            Self::InnerProduct => *b"IxFI",
            Self::L2 | Self::Hamming => *b"IxF2",
        }
    }
}

/// The vectors, row-major, in node order.
#[derive(Clone, Copy)]
pub enum Vectors<'a> {
    /// Float32 values: an `IxF2` or `IxFI` flat storage.
    F32(&'a [f32]),
    /// IEEE half-precision bit patterns: an `IxSQ` storage of `QT_fp16`
    /// codes.
    F16(&'a [u16]),
    /// Int8 values: an `IxSQ` storage of `QT_8bit_direct_signed` codes,
    /// each value plus 128 as a byte.
    I8(&'a [i8]),
    /// Binary vectors, packed bits, dimension / 8 bytes each: an `IBxF`
    /// storage, under an `IBHc` graph. Only with [`Metric::Hamming`].
    Bits(&'a [u8]),
}

impl Vectors<'_> {
    fn len(&self) -> usize {
        match self {
            Self::F32(v) => v.len(),
            Self::F16(v) => v.len(),
            Self::I8(v) => v.len(),
            Self::Bits(v) => v.len(),
        }
    }

    fn numeric_type(&self) -> i32 {
        match self {
            Self::F32(_) => NUMERIC_FLOAT32,
            Self::F16(_) => NUMERIC_FLOAT16,
            Self::I8(_) => NUMERIC_INT8,
            // A binary graph records none.
            Self::Bits(_) => 0,
        }
    }

    fn is_binary(&self) -> bool {
        matches!(self, Self::Bits(_))
    }
}

/// The type and count only: a build's vectors run to millions of values.
impl fmt::Debug for Vectors<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::F32(v) => write!(f, "F32({} values)", v.len()),
            Self::F16(v) => write!(f, "F16({} values)", v.len()),
            Self::I8(v) => write!(f, "I8({} values)", v.len()),
            Self::Bits(v) => write!(f, "Bits({} bytes)", v.len()),
        }
    }
}

/// What a remote build hands the writer.
#[derive(Clone, Copy)]
pub struct IndexSpec<'a> {
    /// The vectors' dimension.
    pub dimension: usize,
    /// The field's space type.
    pub metric: Metric,
    /// The CAGRA graph, row-major: `graph_degree` neighbor ids per node,
    /// node i being the i-th vector.
    pub graph: &'a [u32],
    /// Neighbors per node; even, as the service's m x 4 always is, but
    /// for binary vectors, whose degree cuVS lowers to one less than the
    /// vector count when there are too few: an odd degree keeps the first
    /// degree - 1 neighbors of each node, as `GpuIndexBinaryCagra::copyTo`
    /// does.
    pub graph_degree: usize,
    /// The vectors: values, or for [`Vectors::Bits`] bytes. Their type is
    /// recorded even when they are not written; with
    /// `skip_stored_vectors`, only the type counts, and they
    /// may be empty, so a caller can free them before the write.
    pub vectors: Vectors<'a>,
    /// The doc ids, one per vector, in node order.
    pub ids: &'a [i32],
    /// The request's `ef_construction`, stored in the file.
    pub ef_construction: i32,
    /// The request's `ef_search`, stored in the file.
    pub ef_search: i32,
    /// Write `null` in place of the storage: the graph and ID map only.
    pub skip_stored_vectors: bool,
}

/// Counts in place of the graph and the ids.
impl fmt::Debug for IndexSpec<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IndexSpec")
            .field("dimension", &self.dimension)
            .field("metric", &self.metric)
            .field("graph", &format_args!("{} entries", self.graph.len()))
            .field("graph_degree", &self.graph_degree)
            .field("vectors", &self.vectors)
            .field("ids", &format_args!("{} ids", self.ids.len()))
            .field("ef_construction", &self.ef_construction)
            .field("ef_search", &self.ef_search)
            .field("skip_stored_vectors", &self.skip_stored_vectors)
            .finish()
    }
}

/// A validated [`IndexSpec`], its length known before the first byte is
/// written.
#[derive(Debug, Clone)]
pub struct IndexFile<'a> {
    spec: IndexSpec<'a>,
    count: i32,
    dimension: i32,
    degree: u32,
    table: LevelTable,
    neighbor_slots: u64,
    size: u64,
}

impl<'a> IndexFile<'a> {
    /// Checks `spec` and computes the file's length.
    ///
    /// Draws every node's level once to count the neighbor slots; nothing
    /// proportional to the vector count is kept.
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] when the counts disagree, the degree or
    /// dimension is out of range, binary vectors and the Hamming metric do
    /// not come together, a neighbor id names no vector (which the upstream
    /// service would write anyway; see [`Error::NeighborOutOfRange`]), or
    /// the file would not fit a `u64` length.
    pub fn new(spec: IndexSpec<'a>) -> Result<Self, Error> {
        let len = spec.ids.len();
        if len == 0 {
            return Err(Error::Empty);
        }
        let count = i32::try_from(len).map_err(|_| Error::TooManyVectors { count: len })?;
        let dimension = i32::try_from(spec.dimension)
            .ok()
            .filter(|&d| d > 0)
            .ok_or(Error::Dimension {
                dimension: spec.dimension,
            })?;
        let binary = spec.vectors.is_binary();
        if binary != (spec.metric == Metric::Hamming) {
            return Err(Error::Hamming { binary });
        }
        if binary && !spec.dimension.is_multiple_of(8) {
            return Err(Error::Dimension {
                dimension: spec.dimension,
            });
        }
        let degree_error = Error::GraphDegree {
            degree: spec.graph_degree,
        };
        if !binary && !spec.graph_degree.is_multiple_of(2) {
            return Err(degree_error);
        }
        let m = u32::try_from(spec.graph_degree / 2).map_err(|_| degree_error.clone())?;
        let table = LevelTable::new(m).ok_or(degree_error)?;
        // m and its doubled count fit i32, so the degree fits u32. It is the
        // slots of level 0: the graph's degree, less one where that is odd.
        let degree = m * 2;

        let expected = len.checked_mul(spec.graph_degree).ok_or(Error::TooLarge)?;
        if spec.graph.len() != expected {
            return Err(Error::GraphLength {
                actual: spec.graph.len(),
                expected,
                count: len,
                degree: spec.graph_degree,
            });
        }
        let row = if binary {
            spec.dimension / 8
        } else {
            spec.dimension
        };
        let expected = len.checked_mul(row).ok_or(Error::TooLarge)?;
        if !spec.skip_stored_vectors && spec.vectors.len() != expected {
            return Err(Error::VectorsLength {
                actual: spec.vectors.len(),
                expected,
                count: len,
                dimension: spec.dimension,
            });
        }
        if let Some((index, &neighbor)) = spec
            .graph
            .iter()
            .enumerate()
            .find(|&(_, &id)| u64::from(id) >= u64::from(count.unsigned_abs()))
        {
            return Err(Error::NeighborOutOfRange {
                node: index / spec.graph_degree,
                neighbor,
                count: len,
            });
        }

        let neighbor_slots = table
            .levels(len)
            .try_fold(0_u64, |total, node| {
                total.checked_add(u64::from(node.slots))
            })
            .ok_or(Error::TooLarge)?;
        let size = file_size(&spec, &table, neighbor_slots).ok_or(Error::TooLarge)?;
        Ok(Self {
            spec,
            count,
            dimension,
            degree,
            table,
            neighbor_slots,
            size,
        })
    }

    /// The file's length in bytes: exactly what [`IndexFile::write_to`]
    /// writes.
    #[must_use]
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Writes the file to `out`, in order, through a buffer of its own.
    ///
    /// # Errors
    ///
    /// Returns any error `out` returns, or an error of kind `Other` should
    /// the bytes written differ from [`IndexFile::size`], which would be a
    /// bug here.
    pub fn write_to<W: Write>(&self, out: W) -> io::Result<()> {
        let mut w = Sink::new(out);
        let spec = &self.spec;
        let n = i64::from(self.count);
        let len = self.count.unsigned_abs() as usize;
        let d = self.dimension;
        let metric = spec.metric.faiss();
        let binary = spec.vectors.is_binary();
        let code_size = d / 8;

        if binary {
            // IndexBinaryIDMap, wrapped before training, over
            // IndexBinaryHNSWCagra and its three CAGRA fields; copyTo
            // leaves keep_max_size_level0 false.
            w.binary_header(*b"IBMp", d, code_size, n, false, metric)?;
            w.binary_header(*b"IBHc", d, code_size, n, true, metric)?;
            w.bytes(&[0, 1])?; // keep_max_size_level0, base_level_only
            w.i32(BASE_LEVEL_ENTRY_POINTS)?;
        } else {
            // IndexIDMap: it wrapped the GPU index before training.
            w.header(*b"IxMp", d, n, false, metric)?;
            // IndexHNSWCagra and its four CAGRA fields.
            w.header(*b"IHc2", d, n, true, metric)?;
            w.bytes(&[1, 1])?; // keep_max_size_level0, base_level_only
            w.i32(BASE_LEVEL_ENTRY_POINTS)?;
            w.i32(spec.vectors.numeric_type())?;
        }

        // The HNSW block.
        let probas = self.table.assign_probas();
        let cum = self.table.cum_nneighbor_per_level();
        w.vector(probas.len(), probas.iter().map(|p| p.to_le_bytes()))?;
        w.vector(cum.len(), cum.iter().map(|c| c.to_le_bytes()))?;
        w.vector(
            len,
            self.table.levels(len).map(|node| node.stored.to_le_bytes()),
        )?;
        let offsets = self.table.levels(len).scan(0_u64, |end, node| {
            *end += u64::from(node.slots);
            Some(*end)
        });
        w.vector(
            len + 1,
            iter::once(0_u64).chain(offsets).map(u64::to_le_bytes),
        )?;
        let degree = self.degree;
        let neighbors = spec
            .graph
            .chunks_exact(spec.graph_degree)
            .zip(self.table.levels(len))
            .flat_map(|(row, node): (&[u32], NodeLevel)| {
                // Every id was checked to be below the count, at most
                // i32::MAX, so reinterpreting it is exact. Level 0 holds
                // the first `degree` of the row.
                let level0 = row
                    .iter()
                    .take(degree as usize)
                    .map(|&id| id.cast_signed().to_le_bytes());
                let upper = node.slots.saturating_sub(degree) as usize;
                level0.chain(iter::repeat_n((-1_i32).to_le_bytes(), upper))
            });
        w.vector_u64(self.neighbor_slots, neighbors)?;
        w.i32(-1)?; // entry_point
        w.i32(-1)?; // max_level
        w.i32(spec.ef_construction)?;
        w.i32(spec.ef_search)?;
        w.i32(1)?; // upper_beam, a deprecated constant

        // The storage.
        if spec.skip_stored_vectors {
            w.bytes(b"null")?;
        } else {
            match spec.vectors {
                Vectors::F32(values) => {
                    w.header(spec.metric.flat_fourcc(), d, n, true, metric)?;
                    w.vector(values.len(), values.iter().map(|x| x.to_le_bytes()))?;
                }
                Vectors::F16(values) => {
                    w.scalar_quantizer(d, n, metric, QT_FP16, 2)?;
                    // The codes are a vector of bytes, two per value.
                    w.vector(values.len() * 2, values.iter().map(|x| x.to_le_bytes()))?;
                }
                Vectors::I8(values) => {
                    w.scalar_quantizer(d, n, metric, QT_8BIT_DIRECT_SIGNED, 1)?;
                    // One byte per value, shifted from -128..127 to 0..255.
                    w.vector(
                        values.len(),
                        values.iter().map(|&x| [x.cast_unsigned() ^ 0x80]),
                    )?;
                }
                Vectors::Bits(bytes) => {
                    w.binary_header(*b"IBxF", d, code_size, n, true, metric)?;
                    w.vector(bytes.len(), bytes.iter().map(|&b| [b]))?;
                }
            }
        }

        // The ID map's ids, widened to Faiss's 64-bit idx_t.
        w.vector(len, spec.ids.iter().map(|&id| i64::from(id).to_le_bytes()))?;
        w.finish(self.size)
    }
}

/// The file's length, or `None` past `u64`; see `docs/faiss-layout.md`.
fn file_size(spec: &IndexSpec<'_>, table: &LevelTable, neighbor_slots: u64) -> Option<u64> {
    let n = u64::try_from(spec.ids.len()).ok()?;
    let values = u64::try_from(spec.vectors.len()).ok()?;
    let binary = spec.vectors.is_binary();
    let header = 4 + if binary {
        BINARY_HEADER_LEN
    } else {
        HEADER_LEN
    };
    let vector = |count: u64, width: u64| count.checked_mul(width)?.checked_add(8);
    let probas = u64::try_from(table.assign_probas().len()).ok()?;
    let cum = u64::try_from(table.cum_nneighbor_per_level().len()).ok()?;

    let storage = if spec.skip_stored_vectors {
        4
    } else {
        match spec.vectors {
            Vectors::F32(_) => header.checked_add(vector(values, 4)?)?,
            // qtype, rangestat, rangestat_arg, d, code_size, the empty
            // trained vector, then the codes.
            Vectors::F16(_) => (header + 4 + 4 + 4 + 8 + 8 + 8).checked_add(vector(values, 2)?)?,
            Vectors::I8(_) => (header + 4 + 4 + 4 + 8 + 8 + 8).checked_add(vector(values, 1)?)?,
            Vectors::Bits(_) => header.checked_add(vector(values, 1)?)?,
        }
    };
    // The CAGRA fields: two flags, the entry points, and but for binary
    // graphs the numeric type.
    let cagra = if binary { 1 + 1 + 4 } else { 1 + 1 + 4 + 4 };
    [
        header,                        // IxMp or IBMp
        header + cagra,                // IHc2 or IBHc and its CAGRA fields
        vector(probas, 8)?,            // assign_probas
        vector(cum, 4)?,               // cum_nneighbor_per_level
        vector(n, 4)?,                 // levels
        vector(n.checked_add(1)?, 8)?, // offsets
        vector(neighbor_slots, 4)?,    // neighbors
        5 * 4,                         // entry_point .. upper_beam
        storage,
        vector(n, 8)?, // id_map
    ]
    .into_iter()
    .try_fold(0_u64, u64::checked_add)
}

/// Writes through a buffer, counting bytes, and gathers arrays into
/// chunks so the sink sees few, large writes.
struct Sink<W: Write> {
    out: BufWriter<W>,
    written: u64,
    chunk: Vec<u8>,
}

impl<W: Write> Sink<W> {
    fn new(out: W) -> Self {
        Self {
            out: BufWriter::with_capacity(CHUNK, out),
            written: 0,
            chunk: Vec::with_capacity(CHUNK),
        }
    }

    fn bytes(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.out.write_all(bytes)?;
        self.written += bytes.len() as u64;
        Ok(())
    }

    fn i32(&mut self, value: i32) -> io::Result<()> {
        self.bytes(&value.to_le_bytes())
    }

    /// `write_index_header`, after the structure's fourcc.
    fn header(
        &mut self,
        fourcc: [u8; 4],
        d: i32,
        ntotal: i64,
        is_trained: bool,
        metric: i32,
    ) -> io::Result<()> {
        self.bytes(&fourcc)?;
        self.i32(d)?;
        self.bytes(&ntotal.to_le_bytes())?;
        self.bytes(&HEADER_DUMMY.to_le_bytes())?;
        self.bytes(&HEADER_DUMMY.to_le_bytes())?;
        self.bytes(&[u8::from(is_trained)])?;
        // A metric above 1 would add metric_arg; L2 and inner product do not.
        self.i32(metric)
    }

    /// An `IndexScalarQuantizer`'s header and quantizer, up to its codes:
    /// `qtype`, codes of `width` bytes per component, and no trained values.
    fn scalar_quantizer(
        &mut self,
        d: i32,
        ntotal: i64,
        metric: i32,
        qtype: i32,
        width: u64,
    ) -> io::Result<()> {
        self.header(*b"IxSQ", d, ntotal, true, metric)?;
        self.i32(qtype)?;
        self.i32(0)?; // rangestat
        self.bytes(&0.0_f32.to_le_bytes())?; // rangestat_arg
        let dim = u64::from(d.unsigned_abs());
        self.bytes(&dim.to_le_bytes())?;
        self.bytes(&(dim * width).to_le_bytes())?; // code_size
        self.bytes(&0_u64.to_le_bytes()) // trained: empty
    }

    /// `write_index_binary_header`, after the structure's fourcc.
    fn binary_header(
        &mut self,
        fourcc: [u8; 4],
        d: i32,
        code_size: i32,
        ntotal: i64,
        is_trained: bool,
        metric: i32,
    ) -> io::Result<()> {
        self.bytes(&fourcc)?;
        self.i32(d)?;
        self.i32(code_size)?;
        self.bytes(&ntotal.to_le_bytes())?;
        self.bytes(&[u8::from(is_trained)])?;
        self.i32(metric)
    }

    /// `WRITEVECTOR`: the element count as a u64, then the elements.
    fn vector<const N: usize>(
        &mut self,
        count: usize,
        items: impl IntoIterator<Item = [u8; N]>,
    ) -> io::Result<()> {
        self.vector_u64(count as u64, items)
    }

    fn vector_u64<const N: usize>(
        &mut self,
        count: u64,
        items: impl IntoIterator<Item = [u8; N]>,
    ) -> io::Result<()> {
        self.bytes(&count.to_le_bytes())?;
        self.items(items)
    }

    fn items<const N: usize>(
        &mut self,
        items: impl IntoIterator<Item = [u8; N]>,
    ) -> io::Result<()> {
        for item in items {
            self.chunk.extend_from_slice(&item);
            if self.chunk.len() >= CHUNK {
                self.flush_chunk()?;
            }
        }
        self.flush_chunk()
    }

    fn flush_chunk(&mut self) -> io::Result<()> {
        self.out.write_all(&self.chunk)?;
        self.written += self.chunk.len() as u64;
        self.chunk.clear();
        Ok(())
    }

    fn finish(mut self, expected: u64) -> io::Result<()> {
        self.out.flush()?;
        if self.written == expected {
            Ok(())
        } else {
            Err(io::Error::other(format!(
                "wrote {} bytes where the file is {expected}",
                self.written
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{IndexFile, IndexSpec, Metric, Vectors};
    use crate::Error;

    const GRAPH: [u32; 20] = [1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3, 4, 0, 1, 2, 4, 0, 1, 2, 3];
    const IDS: [i32; 5] = [10, 20, 30, 40, 50];
    const VECTORS: [f32; 10] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

    fn spec() -> IndexSpec<'static> {
        IndexSpec {
            dimension: 2,
            metric: Metric::L2,
            graph: &GRAPH,
            graph_degree: 4,
            vectors: Vectors::F32(&VECTORS),
            ids: &IDS,
            ef_construction: 100,
            ef_search: 100,
            skip_stored_vectors: false,
        }
    }

    fn error(spec: IndexSpec<'_>) -> Error {
        IndexFile::new(spec).unwrap_err()
    }

    #[test]
    fn writes_exactly_its_size() {
        for skip in [false, true] {
            let file = IndexFile::new(IndexSpec {
                skip_stored_vectors: skip,
                ..spec()
            })
            .unwrap();
            let mut out = Vec::new();
            file.write_to(&mut out).unwrap();
            assert_eq!(out.len() as u64, file.size());
        }
    }

    #[test]
    fn needs_only_the_type_of_skipped_vectors() {
        let halves = [0_u16; 10];
        let int8s = [0_i8; 10];
        for (full, empty) in [
            (Vectors::F32(&VECTORS), Vectors::F32(&[])),
            (Vectors::F16(&halves), Vectors::F16(&[])),
            (Vectors::I8(&int8s), Vectors::I8(&[])),
        ] {
            let bytes = |vectors| {
                let file = IndexFile::new(IndexSpec {
                    vectors,
                    skip_stored_vectors: true,
                    ..spec()
                })
                .unwrap();
                let mut out = Vec::new();
                file.write_to(&mut out).unwrap();
                out
            };
            assert_eq!(bytes(empty), bytes(full));
        }
        // Written, they must all be there.
        assert!(matches!(
            error(IndexSpec {
                vectors: Vectors::F32(&[]),
                ..spec()
            }),
            Error::VectorsLength { .. }
        ));
    }

    #[test]
    fn writes_int8_values_as_codes_plus_128() {
        let values = [-128, -1, 0, 1, 127, 5, 6, 7, 8, 9];
        let file = IndexFile::new(IndexSpec {
            vectors: Vectors::I8(&values),
            ..spec()
        })
        .unwrap();
        let mut out = Vec::new();
        file.write_to(&mut out).unwrap();
        // The codes' vector ends where the ID map's ids, a count and five
        // 8-byte ids, begin.
        let ids = out.len() - 8 - IDS.len() * 8;
        let codes = &out[ids - values.len()..ids];
        assert_eq!(codes, [0, 127, 128, 129, 255, 133, 134, 135, 136, 137]);
        let count = &out[ids - values.len() - 8..ids - values.len()];
        assert_eq!(count, 10_u64.to_le_bytes());
    }

    /// Five binary vectors of 16 bits and a graph of `degree`.
    fn binary_spec(graph: &[u32], degree: usize) -> IndexSpec<'_> {
        IndexSpec {
            dimension: 16,
            metric: Metric::Hamming,
            graph,
            graph_degree: degree,
            vectors: Vectors::Bits(&BITS),
            ..spec()
        }
    }

    const BITS: [u8; 10] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

    fn bytes(spec: IndexSpec<'_>) -> Vec<u8> {
        let file = IndexFile::new(spec).unwrap();
        let mut out = Vec::new();
        file.write_to(&mut out).unwrap();
        assert_eq!(out.len() as u64, file.size());
        out
    }

    #[test]
    fn writes_an_odd_binary_degree_as_its_even_part() {
        // Degree 4 plus one more neighbor a row: level 0 keeps the first
        // four, so the file is the degree-4 graph's.
        let odd: Vec<u32> = GRAPH
            .as_chunks::<4>()
            .0
            .iter()
            .enumerate()
            .flat_map(|(node, row)| {
                let extra = u32::try_from((node + 1) % 5).unwrap();
                row.iter().copied().chain([extra])
            })
            .collect();
        assert_eq!(bytes(binary_spec(&odd, 5)), bytes(binary_spec(&GRAPH, 4)));
        let out = bytes(binary_spec(&GRAPH, 4));
        assert_eq!(&out[..4], b"IBMp");
        assert_eq!(&out[25..29], b"IBHc");
        // An odd degree stays refused for other vectors.
        assert_eq!(
            error(IndexSpec {
                graph: &odd,
                graph_degree: 5,
                ..spec()
            }),
            Error::GraphDegree { degree: 5 }
        );
    }

    #[test]
    fn pairs_binary_vectors_with_hamming_alone() {
        assert_eq!(
            error(IndexSpec {
                metric: Metric::L2,
                ..binary_spec(&GRAPH, 4)
            }),
            Error::Hamming { binary: true }
        );
        assert_eq!(
            error(IndexSpec {
                metric: Metric::Hamming,
                ..spec()
            }),
            Error::Hamming { binary: false }
        );
        assert_eq!(
            error(IndexSpec {
                dimension: 12,
                ..binary_spec(&GRAPH, 4)
            }),
            Error::Dimension { dimension: 12 }
        );
        // Binary vectors count in bytes: 5 vectors of 16 bits are 10.
        assert!(matches!(
            error(IndexSpec {
                vectors: Vectors::Bits(&BITS[..9]),
                ..binary_spec(&GRAPH, 4)
            }),
            Error::VectorsLength {
                actual: 9,
                expected: 10,
                ..
            }
        ));
        let skipped = bytes(IndexSpec {
            skip_stored_vectors: true,
            vectors: Vectors::Bits(&[]),
            ..binary_spec(&GRAPH, 4)
        });
        assert_eq!(
            skipped,
            bytes(IndexSpec {
                skip_stored_vectors: true,
                ..binary_spec(&GRAPH, 4)
            })
        );
    }

    #[test]
    fn debug_shows_counts_not_data() {
        let shown = format!("{:?}", IndexFile::new(spec()).unwrap());
        assert!(shown.contains("graph: 20 entries"), "{shown}");
        assert!(shown.contains("vectors: F32(10 values)"), "{shown}");
        assert!(shown.contains("ids: 5 ids"), "{shown}");
        assert!(!shown.contains("[1, 2, 3"), "{shown}");
    }

    #[test]
    fn refuses_inconsistent_specs() {
        assert_eq!(error(IndexSpec { ids: &[], ..spec() }), Error::Empty);
        assert_eq!(
            error(IndexSpec {
                dimension: 0,
                ..spec()
            }),
            Error::Dimension { dimension: 0 }
        );
        for degree in [0, 2, 3, 5] {
            assert_eq!(
                error(IndexSpec {
                    graph_degree: degree,
                    ..spec()
                }),
                Error::GraphDegree { degree }
            );
        }
        assert!(matches!(
            error(IndexSpec {
                graph: &GRAPH[..16],
                ..spec()
            }),
            Error::GraphLength {
                actual: 16,
                expected: 20,
                ..
            }
        ));
        assert!(matches!(
            error(IndexSpec {
                vectors: Vectors::F32(&VECTORS[..9]),
                ..spec()
            }),
            Error::VectorsLength {
                actual: 9,
                expected: 10,
                ..
            }
        ));
        let mut graph = GRAPH;
        graph[13] = 5;
        assert_eq!(
            error(IndexSpec {
                graph: &graph,
                ..spec()
            }),
            Error::NeighborOutOfRange {
                node: 3,
                neighbor: 5,
                count: 5
            }
        );
    }
}
