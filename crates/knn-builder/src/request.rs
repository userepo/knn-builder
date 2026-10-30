//! The body of `POST /_build`, read as the upstream service's model reads
//! it (`IndexBuildParameters`, pydantic 2.10 in lax mode): the same
//! defaults, the same coercions, and on a bad request the same 422 body,
//! one error per field in the model's field order, unknown fields last.
//! `tests/data/requests.json` holds the service's own answers, recorded by
//! `tools/oracle/requests.py`.
//!
//! Where the service accepts a request knn-builder cannot build yet,
//! binary vectors and the hamming space, the error is an `unsupported` one
//! on that field, so k-NN builds the segment itself.
//!
//! The body is read as JSON (RFC 8259) in UTF-8, after a UTF-8 BOM if
//! there is one. Python's `json.loads` reads more, none of which k-NN's
//! client sends: forms outside the RFC, which are UTF-16 and UTF-32,
//! `NaN` and `Infinity`, and lone surrogates (escaped, or encoded in the
//! UTF-8); and JSON past the limits `serde_json` sets, as the RFC lets a
//! parser do, which are numbers past f64's range (`1e400`, infinite to
//! Python) and nesting past 128 levels. Such a body is refused here, with
//! 422 `json_invalid`, or with 400 where it is not UTF-8, as the service
//! answers a body it cannot decode.

use std::fmt::{self, Write as _};

use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use crate::{DataType, Job, JobError, SpaceType};

/// A build request that passed the service's checks.
#[derive(Debug, Clone, PartialEq)]
pub struct BuildRequest {
    /// `container_name`: the bucket.
    pub container_name: String,
    /// `vector_path`, ending in `.knnvec`.
    pub vector_path: String,
    /// `doc_id_path`.
    pub doc_id_path: String,
    /// `tenant_id`, empty when absent.
    pub tenant_id: String,
    /// Above 0. The service takes any integer; this takes one up to
    /// `u64::MAX`, leaving sizes no build could hold to the memory check.
    pub dimension: u64,
    /// Above 4.
    pub doc_count: u64,
    /// `data_type`.
    pub data_type: DataType,
    /// `index_parameters.space_type`.
    pub space_type: SpaceType,
    /// `index_parameters.algorithm_parameters.ef_construction`.
    pub ef_construction: i64,
    /// `index_parameters.algorithm_parameters.ef_search`.
    pub ef_search: i64,
    /// `index_parameters.algorithm_parameters.m`.
    pub m: i64,
    /// Any other `algorithm_parameters`, kept as the service keeps them.
    pub other_algorithm_parameters: Map<String, Value>,
    /// `skip_stored_vectors`.
    pub skip_stored_vectors: bool,
}

/// A refused request, as the service answers it.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Rejection {
    /// 422: the body is not JSON, or not a valid request. One error per
    /// field, in the service's order.
    Invalid(Vec<FieldError>),
    /// 400: the body is not UTF-8 text.
    Unreadable,
}

/// One field's error, as the service reports it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldError {
    /// The field's path: `body.index_parameters.space_type`.
    pub field: String,
    /// The service's message.
    pub message: String,
    /// The service's error type: `missing`, `greater_than`, `enum`, ...;
    /// `unsupported` for what only knn-builder refuses.
    pub kind: &'static str,
}

impl Rejection {
    /// The HTTP status: 422 or 400.
    #[must_use]
    pub fn status(&self) -> u16 {
        match self {
            Self::Invalid(_) => 422,
            Self::Unreadable => 400,
        }
    }

    /// The field errors; none for an unreadable body.
    #[must_use]
    pub fn errors(&self) -> &[FieldError] {
        match self {
            Self::Invalid(errors) => errors,
            Self::Unreadable => &[],
        }
    }

    /// The response body: for 422, `detail` and `errors` of `field`,
    /// `message`, `type`; for 400, the service's `detail`.
    #[must_use]
    pub fn body(&self) -> Value {
        match self {
            Self::Invalid(errors) => {
                let errors: Vec<Value> = errors
                    .iter()
                    .map(|e| json!({"field": e.field, "message": e.message, "type": e.kind}))
                    .collect();
                json!({"detail": "Validation Error", "errors": errors})
            }
            Self::Unreadable => json!({"detail": "There was an error parsing the body"}),
        }
    }
}

/// A job's id: the 64 hex digits [`BuildRequest::job_id`] derives, or
/// whatever id a status request names, unchecked.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct JobId(String);

impl JobId {
    /// The id as the service writes it.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// The id a status request names, which may be of no job.
impl From<&str> for JobId {
    fn from(id: &str) -> Self {
        Self(id.to_owned())
    }
}

impl fmt::Display for JobId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl BuildRequest {
    /// The job id: the SHA-256 of the vector path and tenant id joined by
    /// a dash, in hex, as the service derives it. A request repeated, as
    /// k-NN retries one, finds its job.
    #[must_use]
    pub fn job_id(&self) -> JobId {
        let digest = Sha256::digest(format!("{}-{}", self.vector_path, self.tenant_id));
        JobId(digest.iter().fold(String::new(), |mut hex, b| {
            let _ = write!(hex, "{b:02x}");
            hex
        }))
    }

    /// The parameters with defaults filled in, as the service's model
    /// dumps them.
    #[must_use]
    pub fn to_json(&self) -> Value {
        let mut algorithm_parameters = Map::new();
        algorithm_parameters.insert("ef_construction".into(), self.ef_construction.into());
        algorithm_parameters.insert("ef_search".into(), self.ef_search.into());
        algorithm_parameters.insert("m".into(), self.m.into());
        algorithm_parameters.extend(self.other_algorithm_parameters.clone());
        json!({
            "repository_type": "s3",
            "container_name": self.container_name,
            "vector_path": self.vector_path,
            "doc_id_path": self.doc_id_path,
            "tenant_id": self.tenant_id,
            "dimension": self.dimension,
            "doc_count": self.doc_count,
            "data_type": match self.data_type {
                DataType::Float => "float",
                DataType::HalfFloat => "half_float",
                DataType::Byte => "byte",
            },
            "engine": "faiss",
            "index_parameters": {
                "algorithm": "hnsw",
                "space_type": match self.space_type {
                    SpaceType::L2 => "l2",
                    SpaceType::InnerProduct => "innerproduct",
                },
                "algorithm_parameters": algorithm_parameters,
            },
            "skip_stored_vectors": self.skip_stored_vectors,
        })
    }

    /// The pipeline's job.
    ///
    /// # Errors
    ///
    /// [`JobError::OutOfRange`] for a parameter the service accepts but no
    /// build takes: a dimension, count or m past 32 bits or below 0, or an
    /// ef past 32 bits. The service takes such a job and fails it in its
    /// build; so does knn-builder, before downloading anything.
    pub fn job(&self) -> Result<Job, JobError> {
        fn within<T: TryFrom<i128> + Into<i128>>(
            name: &'static str,
            value: i128,
            min: T,
            max: T,
        ) -> Result<T, JobError> {
            T::try_from(value).map_err(|_| JobError::OutOfRange {
                name,
                value,
                min: min.into(),
                max: max.into(),
            })
        }
        let unsigned = |name, value: i128| within(name, value, u32::MIN, u32::MAX);
        let signed = |name, value: i64| within(name, value.into(), i32::MIN, i32::MAX);
        Ok(Job {
            bucket: self.container_name.clone(),
            vector_path: self.vector_path.clone(),
            doc_id_path: self.doc_id_path.clone(),
            dimension: unsigned("dimension", self.dimension.into())?,
            doc_count: unsigned("doc_count", self.doc_count.into())?,
            data_type: self.data_type,
            space_type: self.space_type,
            m: unsigned("m", self.m.into())?,
            ef_construction: signed("ef_construction", self.ef_construction)?,
            ef_search: signed("ef_search", self.ef_search)?,
            skip_stored_vectors: self.skip_stored_vectors,
        })
    }
}

/// The top-level fields, in the model's order.
const FIELDS: [&str; 11] = [
    "repository_type",
    "container_name",
    "vector_path",
    "doc_id_path",
    "tenant_id",
    "dimension",
    "doc_count",
    "data_type",
    "engine",
    "index_parameters",
    "skip_stored_vectors",
];

/// The vector path's pattern, as the service's message quotes it.
const VECTOR_PATH_PATTERN: &str = r".+\.knnvec$";

const MODEL_TYPE: &str = "Input should be a valid dictionary or object to extract fields from";

/// Reads a `POST /_build` body sent with the `Content-Type` header
/// `content_type`, as FastAPI reads one: as JSON, by [`parse`], when the
/// header is missing or empty or its media type is `application/json` or
/// `application/*+json`, in any case; otherwise as raw bytes, which the
/// model refuses, unless there are none.
///
/// # Errors
///
/// The [`Rejection`] the service would answer with.
pub fn parse_as(content_type: Option<&[u8]>, body: &[u8]) -> Result<BuildRequest, Rejection> {
    if is_json(content_type) {
        parse(body)
    } else if body.is_empty() {
        Err(single("body", "Field required", "missing"))
    } else {
        Err(single("body", MODEL_TYPE, "model_attributes_type"))
    }
}

/// Whether FastAPI reads a body of `content_type` as JSON. It takes the
/// media type as Python's `email.message` does: before any `;`, stripped,
/// in lower case, and `text/plain` unless it has exactly one `/`.
fn is_json(content_type: Option<&[u8]>) -> bool {
    let Some(value) = content_type.filter(|value| !value.is_empty()) else {
        return true;
    };
    let media = value
        .split(|&b| b == b';')
        .next()
        .unwrap_or_default()
        .trim_ascii()
        .to_ascii_lowercase();
    let mut parts = media.split(|&b| b == b'/');
    match (parts.next(), parts.next(), parts.next()) {
        (Some(b"application"), Some(subtype), None) => {
            subtype == b"json" || subtype.ends_with(b"+json")
        }
        _ => false,
    }
}

/// Reads a `POST /_build` body as JSON.
///
/// # Errors
///
/// The [`Rejection`] the service would answer with.
pub fn parse(body: &[u8]) -> Result<BuildRequest, Rejection> {
    // FastAPI reads only a body with bytes in it; an empty one is none.
    if body.is_empty() {
        return Err(single("body", "Field required", "missing"));
    }
    // json.loads drops a UTF-8 BOM, and decodes the rest before parsing
    // it: bytes that are not UTF-8 are a 400, whatever JSON they hold.
    let body = body.strip_prefix(b"\xef\xbb\xbf").unwrap_or(body);
    let text = std::str::from_utf8(body).map_err(|_| Rejection::Unreadable)?;
    let value: Value = match serde_json::from_str(text) {
        Ok(value) => value,
        Err(error) => {
            return Err(Rejection::Invalid(vec![FieldError {
                field: format!("body[{}]", offset(text, &error)),
                message: "JSON decode error".into(),
                kind: "json_invalid",
            }]));
        }
    };
    let object = match value {
        Value::Object(object) => object,
        // FastAPI takes a null body for no body.
        Value::Null => return Err(single("body", "Field required", "missing")),
        _ => return Err(single("body", MODEL_TYPE, "model_attributes_type")),
    };
    let mut errors = Errors(Vec::new());
    let field = |name: &str| object.get(name);

    errors.choice(
        "body.repository_type",
        field("repository_type"),
        &["s3"],
        "s3",
    );
    let container_name = errors.required_string("body.container_name", field("container_name"));
    let vector_path = errors.required_string("body.vector_path", field("vector_path"));
    if let Some(path) = &vector_path
        && !matches_vector_path(path)
    {
        errors.push(
            "body.vector_path",
            format!("String should match pattern '{VECTOR_PATH_PATTERN}'"),
            "string_pattern_mismatch",
        );
    }
    let doc_id_path = errors.required_string("body.doc_id_path", field("doc_id_path"));
    let tenant_id = match field("tenant_id") {
        None => Some(String::new()),
        Some(value) => errors.string("body.tenant_id", value),
    };
    let dimension = errors.count("body.dimension", field("dimension"), 0);
    let doc_count = errors.count("body.doc_count", field("doc_count"), 4);
    let data_type = errors
        .choice(
            "body.data_type",
            field("data_type"),
            &["float", "half_float", "byte", "binary"],
            "float",
        )
        .and_then(|choice| match choice {
            "float" => Some(DataType::Float),
            "half_float" => Some(DataType::HalfFloat),
            "byte" => Some(DataType::Byte),
            other => {
                errors.unsupported("body.data_type", &format!("{other} vectors"));
                None
            }
        });
    errors.choice("body.engine", field("engine"), &["faiss"], "faiss");
    let index = errors.index_parameters(field("index_parameters"));
    let skip_stored_vectors = match field("skip_stored_vectors") {
        None => Some(false),
        Some(value) => errors.boolean("body.skip_stored_vectors", value),
    };
    for name in object
        .keys()
        .filter(|name| !FIELDS.contains(&name.as_str()))
    {
        errors.push(
            &format!("body.{name}"),
            "Extra inputs are not permitted".into(),
            "extra_forbidden",
        );
    }

    // Every field read when no error was pushed.
    let request = (|| {
        let index = index?;
        Some(BuildRequest {
            container_name: container_name?,
            vector_path: vector_path?,
            doc_id_path: doc_id_path?,
            tenant_id: tenant_id?,
            dimension: dimension?,
            doc_count: doc_count?,
            data_type: data_type?,
            space_type: index.space_type,
            ef_construction: index.ef_construction,
            ef_search: index.ef_search,
            m: index.m,
            other_algorithm_parameters: index.others,
            skip_stored_vectors: skip_stored_vectors?,
        })
    })();
    match request {
        Some(request) if errors.0.is_empty() => Ok(request),
        _ => Err(Rejection::Invalid(errors.0)),
    }
}

/// `.+\.knnvec$` as the service's regex engine reads it: at least one
/// character other than a newline, then `.knnvec` at the very end.
fn matches_vector_path(path: &str) -> bool {
    path.strip_suffix(".knnvec")
        .and_then(|stem| stem.chars().last())
        .is_some_and(|last| last != '\n')
}

/// Where a JSON error is, for the service's `body[offset]`: Python's
/// offset, in characters from 0, of the character it stopped at, or of
/// the end. serde gives the line and the column, in bytes from 1, of the
/// last byte it read. The two agree on the errors in the table; on
/// others, an invalid escape say, they can differ by a character or two.
fn offset(text: &str, error: &serde_json::Error) -> usize {
    let line_start: usize = text
        .split('\n')
        .take(error.line().saturating_sub(1))
        .map(|line| line.len() + 1)
        .sum();
    let past_end = usize::from(error.is_eof());
    let at = (line_start + error.column() + past_end).saturating_sub(1);
    // Characters before byte `at`, which may fall inside one.
    text.char_indices().take_while(|&(i, _)| i < at).count()
}

fn single(field: &str, message: &str, kind: &'static str) -> Rejection {
    Rejection::Invalid(vec![FieldError {
        field: field.into(),
        message: message.into(),
        kind,
    }])
}

/// `index_parameters`, defaults filled in.
struct Index {
    space_type: SpaceType,
    ef_construction: i64,
    ef_search: i64,
    m: i64,
    others: Map<String, Value>,
}

/// The errors so far; each reader returns `None` after pushing one.
struct Errors(Vec<FieldError>);

impl Errors {
    fn push(&mut self, field: &str, message: String, kind: &'static str) {
        self.0.push(FieldError {
            field: field.into(),
            message,
            kind,
        });
    }

    fn unsupported(&mut self, field: &str, what: &str) {
        self.push(
            field,
            format!(
                "knn-builder does not build {what} yet; the data node builds this segment itself"
            ),
            "unsupported",
        );
    }

    fn required_string(&mut self, field: &str, value: Option<&Value>) -> Option<String> {
        match value {
            None => {
                self.push(field, "Field required".into(), "missing");
                None
            }
            Some(value) => self.string(field, value),
        }
    }

    fn string(&mut self, field: &str, value: &Value) -> Option<String> {
        if let Value::String(text) = value {
            Some(text.clone())
        } else {
            self.push(
                field,
                "Input should be a valid string".into(),
                "string_type",
            );
            None
        }
    }

    /// A required integer above `floor`.
    fn count(&mut self, field: &str, value: Option<&Value>, floor: u64) -> Option<u64> {
        let Some(value) = value else {
            self.push(field, "Field required".into(), "missing");
            return None;
        };
        let number = self.integer(field, value)?;
        if number <= i128::from(floor) {
            self.push(
                field,
                format!("Input should be greater than {floor}"),
                "greater_than",
            );
            return None;
        }
        self.within(field, number, u64::MIN, u64::MAX)
    }

    /// An optional integer with a default.
    fn int_or(&mut self, field: &str, value: Option<&Value>, default: i64) -> Option<i64> {
        let Some(value) = value else {
            return Some(default);
        };
        let number = self.integer(field, value)?;
        self.within(field, number, i64::MIN, i64::MAX)
    }

    /// `number` as a `T`. The service's model has no bounds here; a number
    /// past what knn-builder holds is refused as pydantic refuses one past
    /// a bound.
    fn within<T>(&mut self, field: &str, number: i128, min: T, max: T) -> Option<T>
    where
        T: TryFrom<i128> + Into<i128>,
    {
        let (min, max) = (min.into(), max.into());
        if number < min {
            let message = format!("Input should be greater than or equal to {min}");
            self.push(field, message, "greater_than_equal");
        } else if number > max {
            let message = format!("Input should be less than or equal to {max}");
            self.push(field, message, "less_than_equal");
        }
        T::try_from(number).ok()
    }

    /// An integer as pydantic's lax mode takes one: a number with no
    /// fractional part, a bool, or a string of one, trimmed, signed or
    /// with underscores between digits and a zero fraction.
    fn integer(&mut self, field: &str, value: &Value) -> Option<i128> {
        match value {
            Value::Number(number) => {
                if let Some(int) = number.as_i64() {
                    return Some(int.into());
                }
                if let Some(int) = number.as_u64() {
                    return Some(int.into());
                }
                // serde_json reads an integer past 64 bits as a float, so
                // this refuses one the service takes (tests/request.rs).
                let float = number.as_f64().unwrap_or(f64::NAN);
                if float.fract() != 0.0 {
                    self.push(
                        field,
                        "Input should be a valid integer, got a number with a fractional part"
                            .into(),
                        "int_from_float",
                    );
                    return None;
                }
                integral(float).or_else(|| {
                    self.push(
                        field,
                        "Unable to parse input string as an integer, exceeded maximum size".into(),
                        "int_parsing_size",
                    );
                    None
                })
            }
            Value::Bool(flag) => Some(i128::from(*flag)),
            Value::String(text) => parse_int(text).or_else(|| {
                self.push(field, INT_PARSING.into(), "int_parsing");
                None
            }),
            _ => {
                self.push(field, "Input should be a valid integer".into(), "int_type");
                None
            }
        }
    }

    /// A bool as pydantic's lax mode takes one: a bool, 0 or 1, or a word
    /// such as `yes` or `off`.
    fn boolean(&mut self, field: &str, value: &Value) -> Option<bool> {
        let parsed = match value {
            Value::Bool(flag) => Some(*flag),
            Value::Number(number) => {
                let float = number.as_f64().unwrap_or(f64::NAN);
                if float.fract() != 0.0 {
                    self.push(field, "Input should be a valid boolean".into(), "bool_type");
                    return None;
                }
                match integral(float) {
                    Some(0) => Some(false),
                    Some(1) => Some(true),
                    _ => None,
                }
            }
            Value::String(text) => match text.to_ascii_lowercase().as_str() {
                "0" | "off" | "f" | "false" | "n" | "no" => Some(false),
                "1" | "on" | "t" | "true" | "y" | "yes" => Some(true),
                _ => None,
            },
            _ => {
                self.push(field, "Input should be a valid boolean".into(), "bool_type");
                return None;
            }
        };
        if parsed.is_none() {
            self.push(
                field,
                "Input should be a valid boolean, unable to interpret input".into(),
                "bool_parsing",
            );
        }
        parsed
    }

    /// An optional string enum: one of `choices`, `default` when absent,
    /// `None` after an error.
    fn choice<'a>(
        &mut self,
        field: &str,
        value: Option<&'a Value>,
        choices: &[&str],
        default: &'static str,
    ) -> Option<&'a str> {
        let Some(value) = value else {
            return Some(default);
        };
        match value.as_str() {
            Some(text) if choices.contains(&text) => Some(text),
            _ => {
                self.push(
                    field,
                    format!("Input should be {}", or_list(choices)),
                    "enum",
                );
                None
            }
        }
    }

    fn index_parameters(&mut self, value: Option<&Value>) -> Option<Index> {
        let empty = Map::new();
        let object = match value {
            None => &empty,
            Some(Value::Object(object)) => object,
            Some(_) => {
                self.push(
                    "body.index_parameters",
                    MODEL_TYPE.into(),
                    "model_attributes_type",
                );
                return None;
            }
        };
        // Unknown fields here are ignored, as the service's model does.
        self.choice(
            "body.index_parameters.algorithm",
            object.get("algorithm"),
            &["hnsw"],
            "hnsw",
        );
        let space_type = self
            .choice(
                "body.index_parameters.space_type",
                object.get("space_type"),
                &["l2", "innerproduct", "hamming"],
                "l2",
            )
            .and_then(|choice| match choice {
                "l2" => Some(SpaceType::L2),
                "innerproduct" => Some(SpaceType::InnerProduct),
                other => {
                    self.unsupported(
                        "body.index_parameters.space_type",
                        &format!("the {other} space"),
                    );
                    None
                }
            });
        let parameters = match object.get("algorithm_parameters") {
            None => &empty,
            Some(Value::Object(parameters)) => parameters,
            Some(_) => {
                self.push(
                    "body.index_parameters.algorithm_parameters",
                    MODEL_TYPE.into(),
                    "model_attributes_type",
                );
                return None;
            }
        };
        let path = |name: &str| format!("body.index_parameters.algorithm_parameters.{name}");
        let ef_construction = self.int_or(
            &path("ef_construction"),
            parameters.get("ef_construction"),
            100,
        );
        let ef_search = self.int_or(&path("ef_search"), parameters.get("ef_search"), 100);
        let m = self.int_or(&path("m"), parameters.get("m"), 16);
        let others = parameters
            .iter()
            .filter(|(name, _)| !matches!(name.as_str(), "ef_construction" | "ef_search" | "m"))
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect();
        Some(Index {
            space_type: space_type?,
            ef_construction: ef_construction?,
            ef_search: ef_search?,
            m: m?,
            others,
        })
    }
}

/// An integral float as an integer, as pydantic takes one: within 64
/// signed bits.
fn integral(float: f64) -> Option<i128> {
    // 2^63, exactly.
    const BOUND: f64 = 9_223_372_036_854_775_808.0;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "used only when integral and within 64 signed bits, so exact"
    )]
    (float.fract() == 0.0 && (-BOUND..BOUND).contains(&float)).then_some(float as i128)
}

const INT_PARSING: &str = "Input should be a valid integer, unable to parse string as an integer";

/// `'a', 'b' or 'c'`, as pydantic lists an enum's values.
fn or_list(choices: &[&str]) -> String {
    let quoted: Vec<String> = choices.iter().map(|c| format!("'{c}'")).collect();
    match quoted.split_last() {
        Some((last, rest)) if !rest.is_empty() => format!("{} or {last}", rest.join(", ")),
        Some((only, _)) => only.clone(),
        None => String::new(),
    }
}

/// A string as pydantic reads an integer from one: trimmed, an optional
/// sign, digits with single underscores between them, and an optional
/// fraction of zeros.
fn parse_int(text: &str) -> Option<i128> {
    let text = text.trim();
    let (negative, rest) = match text.as_bytes().first() {
        Some(b'-') => (true, text.get(1..)?),
        Some(b'+') => (false, text.get(1..)?),
        _ => (false, text),
    };
    let (whole, fraction) = rest.split_once('.').unwrap_or((rest, "0"));
    if fraction.is_empty() || !fraction.bytes().all(|b| b == b'0') {
        return None;
    }
    if whole.is_empty()
        || whole.starts_with('_')
        || whole.ends_with('_')
        || whole.contains("__")
        || !whole.bytes().all(|b| b.is_ascii_digit() || b == b'_')
    {
        return None;
    }
    let digits: String = whole.chars().filter(|&c| c != '_').collect();
    // Digits only, so the parse fails only past i128; Python takes any
    // length, so such a string is an integer past every bound, not text.
    let magnitude: i128 = digits.parse().unwrap_or(i128::MAX);
    Some(if negative { -magnitude } else { magnitude })
}

#[cfg(test)]
mod tests {
    use super::{is_json, matches_vector_path, or_list, parse_int};

    #[test]
    fn reads_integers_from_strings_as_pydantic() {
        assert_eq!(parse_int(" 768 "), Some(768));
        assert_eq!(parse_int("+768"), Some(768));
        assert_eq!(parse_int("-3"), Some(-3));
        assert_eq!(parse_int("7_68"), Some(768));
        assert_eq!(parse_int("768.0"), Some(768));
        assert_eq!(parse_int("768.00"), Some(768));
        let past_i128 = "9".repeat(40);
        assert_eq!(parse_int(&past_i128), Some(i128::MAX));
        assert_eq!(parse_int(&format!("-{past_i128}")), Some(-i128::MAX));
        for bad in [
            "abc", "1.5", "7__68", "_768", "768_", "", "+", "768.", "0x10",
        ] {
            assert_eq!(parse_int(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn matches_the_vector_path_pattern() {
        assert!(matches_vector_path("a/b.knnvec"));
        assert!(matches_vector_path("x.knnvec"));
        assert!(!matches_vector_path(".knnvec"));
        assert!(!matches_vector_path("a/b.vec"));
        assert!(!matches_vector_path("a\n.knnvec"));
        assert!(!matches_vector_path("a.knnvec\n"));
    }

    #[test]
    fn reads_json_by_its_media_type_as_fastapi() {
        for json in [
            None,
            Some(&b""[..]),
            Some(b"application/json"),
            Some(b"application/json; charset=utf-8"),
            Some(b"  Application/JSON ;x=y"),
            Some(b"application/vnd.api+json"),
        ] {
            assert!(is_json(json), "{json:?}");
        }
        for other in [
            &b"text/plain"[..],
            b"application",
            b" ",
            b"application/json/x",
            b"application/x-www-form-urlencoded",
            b"application /json",
            b"text/json",
        ] {
            assert!(!is_json(Some(other)), "{other:?}");
        }
    }

    #[test]
    fn lists_choices_as_pydantic() {
        assert_eq!(or_list(&["faiss"]), "'faiss'");
        assert_eq!(or_list(&["a", "b"]), "'a' or 'b'");
        assert_eq!(or_list(&["a", "b", "c"]), "'a', 'b' or 'c'");
    }
}
