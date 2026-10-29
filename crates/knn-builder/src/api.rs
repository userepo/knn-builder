//! The HTTP API k-NN calls, answered as the upstream service answers it:
//! `POST /_build`, `GET /_status/{job_id}` and `GET /_heart_beat`, with its
//! status codes and bodies. As FastAPI does, a route answers another
//! method, HEAD included, with 405, its `Allow` and
//! `{"detail":"Method Not Allowed"}`; a path no route takes with 404 and
//! `{"detail":"Not Found"}`, or with a 307 to the route's path when only
//! trailing slashes are in the way. Like Starlette, it routes on the
//! percent-decoded path, so `%2F` separates as `/` does.
//! `tests/data/routes.json` holds the service's own answers, recorded by
//! `tools/oracle/routes.py`.
//!
//! The service also serves FastAPI's documentation (`/docs`, `/redoc`,
//! `/openapi.json`) and a debug list of every job (`/_jobs`); here they are
//! paths like any other, 404. A build request's body may be 2 MB at most,
//! axum's default; a larger one gets 413, where the service reads any
//! size. k-NN's are a few hundred bytes.
//!
//! With [`Credentials`] set, every route asks for them by Basic Auth, and
//! answers 401 without them; unset, the `Authorization` header is
//! ignored, as the service ignores it.

use std::fmt::Write as _;
use std::sync::Arc;
use std::time::Instant;

use axum::body::Bytes;
use axum::extract::{Path, Request, State};
use axum::http::uri::PathAndQuery;
use axum::http::{HeaderMap, HeaderValue, StatusCode, Uri, header};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use base64::Engine as _;
use serde_json::{Value, json};
use tower::ServiceExt as _;
use tower::util::MapRequest;

use crate::config::Credentials;
use crate::request::{self, JobId};
use crate::service::{Service, Status};

/// What `GET /_heart_beat` answers: the service's name.
pub const SERVICE_NAME: &str = "remote-vector-index-builder-api";

#[derive(Debug, Clone)]
struct Api {
    service: Arc<Service>,
    auth: Option<Arc<Credentials>>,
}

/// The API as a service: the routes, behind the path decoding.
pub type App = MapRequest<Router, fn(Request) -> Request>;

/// The API on `service`, asking for `auth` when it is set. Serve it with
/// [`axum::ServiceExt::into_make_service`].
#[must_use]
pub fn app(service: Arc<Service>, auth: Option<Credentials>) -> App {
    router(service, auth).map_request(decode_path as fn(Request) -> Request)
}

/// The request with its path as Starlette routes on it: percent-decoded,
/// bytes that are not UTF-8 replaced, as uvicorn decodes it; then encoded
/// again only where a URI needs it, so an encoded slash routes as a slash
/// and an encoded percent stays a percent.
fn decode_path(mut request: Request) -> Request {
    let uri = request.uri();
    if !uri.path().contains('%') {
        return request;
    }
    let path = encoded(&decoded(uri.path()));
    let path_and_query = match uri.query() {
        Some(query) => format!("{path}?{query}"),
        None => path,
    };
    let mut parts = uri.clone().into_parts();
    if let Ok(path_and_query) = PathAndQuery::try_from(path_and_query) {
        parts.path_and_query = Some(path_and_query);
        if let Ok(uri) = Uri::from_parts(parts) {
            *request.uri_mut() = uri;
        }
    }
    request
}

/// `path` percent-decoded, as Python's `unquote` decodes it: a `%` not
/// followed by two hex digits stays, and bytes that are not UTF-8 become
/// U+FFFD.
fn decoded(path: &str) -> String {
    let bytes = path.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut at = 0;
    while let Some(&byte) = bytes.get(at) {
        let escaped = bytes
            .get(at + 1..at + 3)
            .filter(|_| byte == b'%')
            .and_then(|hex| Some(hex_digit(*hex.first()?)? << 4 | hex_digit(*hex.get(1)?)?));
        if let Some(value) = escaped {
            out.push(value);
            at += 3;
        } else {
            out.push(byte);
            at += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn hex_digit(byte: u8) -> Option<u8> {
    char::from(byte)
        .to_digit(16)
        .and_then(|digit| u8::try_from(digit).ok())
}

/// `path` with every byte a URI path cannot hold as it is
/// percent-encoded: all but the unreserved characters, the
/// sub-delimiters, `:`, `@` and `/`.
fn encoded(path: &str) -> String {
    let mut out = String::with_capacity(path.len());
    for &byte in path.as_bytes() {
        if byte.is_ascii_alphanumeric() || b"-._~!$&'()*+,;=:@/".contains(&byte) {
            out.push(char::from(byte));
        } else {
            let _ = write!(out, "%{byte:02X}");
        }
    }
    out
}

/// The routes, on `service`, asking for `auth` when it is set.
fn router(service: Arc<Service>, auth: Option<Credentials>) -> Router {
    let api = Api {
        service,
        auth: auth.map(Arc::new),
    };
    Router::new()
        .route("/_build", post(build).fallback(not_allowed("POST")))
        .route(
            "/_status/{job_id}",
            get(status)
                .head(not_allowed("GET"))
                .fallback(not_allowed("GET")),
        )
        .route(
            "/_heart_beat",
            get(heart_beat)
                .head(not_allowed("GET"))
                .fallback(not_allowed("GET")),
        )
        .route_layer(middleware::from_fn_with_state(api.clone(), authenticate))
        .fallback(not_found)
        .layer(middleware::from_fn(log))
        .with_state(api)
}

async fn build(State(api): State<Api>, headers: HeaderMap, body: Bytes) -> Response {
    let content_type = headers.get(header::CONTENT_TYPE).map(HeaderValue::as_bytes);
    let request = match request::parse_as(content_type, &body) {
        Ok(request) => request,
        Err(rejection) => {
            tracing::info!(errors = ?rejection.errors(), "build request refused");
            return answer(rejection.status(), &rejection.body());
        }
    };
    match api.service.submit(request) {
        Ok(job_id) => Json(json!({ "job_id": job_id.as_str() })).into_response(),
        Err(refusal) => detail(refusal.status(), &refusal.to_string()),
    }
}

async fn status(State(api): State<Api>, Path(job_id): Path<String>) -> Response {
    match api.service.status(&JobId::from(job_id.as_str())) {
        Some(status) => Json(status_body(&status)).into_response(),
        None => detail(404, "Job not found"),
    }
}

async fn heart_beat() -> Json<&'static str> {
    Json(SERVICE_NAME)
}

/// The status response: every field, null where it does not apply.
fn status_body(status: &Status) -> Value {
    let (file_name, error_message) = match status {
        Status::Running => (None, None),
        Status::Completed { file_name } => (Some(file_name), None),
        Status::Failed { error_message } => (None, Some(error_message)),
    };
    json!({
        "task_status": status.task_status(),
        "file_name": file_name,
        "error_message": error_message,
    })
}

/// A 405 for a route that takes only `allow`. Axum sends a HEAD's answer
/// without its body, as HTTP has it.
fn not_allowed(
    allow: &'static str,
) -> impl Fn() -> std::future::Ready<Response> + Clone + Send + 'static {
    move || {
        let mut response = detail(405, "Method Not Allowed");
        response
            .headers_mut()
            .insert(header::ALLOW, HeaderValue::from_static(allow));
        std::future::ready(response)
    }
}

/// A path no route takes: 404, or a 307 to a route's path when only
/// trailing slashes are in the way, as Starlette redirects it (all of
/// them stripped, to an absolute URL on the request's `Host`, the query
/// kept).
async fn not_found(headers: HeaderMap, uri: Uri) -> Response {
    let path = uri.path();
    let route = path.trim_end_matches('/');
    if route.len() < path.len() && is_route(route) {
        let query = uri.query().map(|q| format!("?{q}")).unwrap_or_default();
        let host = headers.get(header::HOST).and_then(|h| h.to_str().ok());
        let location = match host {
            Some(host) => format!("http://{host}{route}{query}"),
            None => format!("{route}{query}"),
        };
        if let Ok(location) = HeaderValue::try_from(location) {
            return (
                StatusCode::TEMPORARY_REDIRECT,
                [(header::LOCATION, location)],
            )
                .into_response();
        }
    }
    detail(404, "Not Found")
}

/// Whether a route takes `path`.
fn is_route(path: &str) -> bool {
    matches!(path, "/_build" | "/_heart_beat")
        || path
            .strip_prefix("/_status/")
            .is_some_and(|id| !id.is_empty() && !id.contains('/'))
}

/// Basic Auth, when it is set: 401 for a request without the credentials,
/// with FastAPI's messages.
async fn authenticate(State(api): State<Api>, request: Request, next: Next) -> Response {
    let Some(credentials) = &api.auth else {
        return next.run(request).await;
    };
    match basic(request.headers()) {
        Basic::Missing => unauthorized("Not authenticated"),
        Basic::Unreadable => unauthorized("Invalid authentication credentials"),
        Basic::Given(username, password) => {
            // Both compared in full, whichever differs.
            let user = same(&username, credentials.username.as_bytes());
            let pass = same(&password, credentials.password.as_bytes());
            if user & pass {
                next.run(request).await
            } else {
                unauthorized("Incorrect username or password")
            }
        }
    }
}

/// What a request's `Authorization` header holds.
enum Basic {
    /// No header, or another scheme than Basic.
    Missing,
    /// Basic, but not base64 of `username:password`.
    Unreadable,
    /// A username and password.
    Given(Vec<u8>, Vec<u8>),
}

fn basic(headers: &HeaderMap) -> Basic {
    let Some(value) = headers.get(header::AUTHORIZATION) else {
        return Basic::Missing;
    };
    let value = value.as_bytes();
    let Some(space) = value.iter().position(|&b| b == b' ') else {
        return Basic::Missing;
    };
    let (scheme, encoded) = value.split_at(space);
    if !scheme.eq_ignore_ascii_case(b"basic") {
        return Basic::Missing;
    }
    let Ok(decoded) = base64::engine::general_purpose::STANDARD.decode(encoded.trim_ascii()) else {
        return Basic::Unreadable;
    };
    match decoded.iter().position(|&b| b == b':') {
        Some(colon) => Basic::Given(
            decoded.get(..colon).unwrap_or_default().to_vec(),
            decoded.get(colon + 1..).unwrap_or_default().to_vec(),
        ),
        None => Basic::Unreadable,
    }
}

/// `a == b`, in a time that tells nothing of where they differ.
fn same(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0, |differ, (x, y)| differ | (x ^ y)) == 0
}

fn unauthorized(message: &str) -> Response {
    let mut response = detail(401, message);
    response
        .headers_mut()
        .insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Basic"));
    response
}

/// `{"detail": message}` with `status`, as FastAPI's errors answer.
fn detail(status: u16, message: &str) -> Response {
    answer(status, &json!({ "detail": message }))
}

fn answer(status: u16, body: &Value) -> Response {
    let status = StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    (status, Json(body)).into_response()
}

/// A line per request, as the service's server logs one.
async fn log(request: Request, next: Next) -> Response {
    let (method, path) = (request.method().clone(), request.uri().path().to_owned());
    let start = Instant::now();
    let response = next.run(request).await;
    tracing::info!(
        %method,
        path,
        status = response.status().as_u16(),
        ms = start.elapsed().as_millis(),
        "request"
    );
    response
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::{decoded, encoded, is_route, same, status_body};
    use crate::service::Status;

    /// `tools/oracle/routes.py`'s record of the service's answers.
    const ROUTES: &str = include_str!("../tests/data/routes.json");

    #[test]
    fn answers_a_status_as_the_service_does() {
        let table: Value = serde_json::from_str(ROUTES).unwrap();
        let statuses = &table["statuses"];
        for (name, status) in [
            ("running", Status::Running),
            (
                "completed",
                Status::Completed {
                    file_name: "_0_states.faiss".into(),
                },
            ),
            (
                "failed",
                Status::Failed {
                    error_message: "it failed".into(),
                },
            ),
        ] {
            let body = serde_json::to_string(&status_body(&status)).unwrap();
            assert_eq!(body, statuses[name]["response_body"], "{name}");
        }
    }

    #[test]
    fn knows_its_routes() {
        for route in ["/_build", "/_heart_beat", "/_status/abc"] {
            assert!(is_route(route), "{route}");
        }
        for path in [
            "/",
            "/_status",
            "/_status/",
            "/_status/a/b",
            "/_jobs",
            "/_build/x",
        ] {
            assert!(!is_route(path), "{path}");
        }
    }

    #[test]
    fn decodes_a_path_as_python() {
        assert_eq!(decoded("/a%2Fb%2541"), "/a/b%41");
        assert_eq!(decoded("/%C3%A9"), "/\u{e9}");
        assert_eq!(decoded("/%FF"), "/\u{fffd}");
        // Not two hex digits: left as it is.
        for kept in ["/%", "/%4", "/%G1", "/%+1", "/100%"] {
            assert_eq!(decoded(kept), kept);
        }
    }

    #[test]
    fn encodes_what_a_path_cannot_hold() {
        assert_eq!(encoded("/a/b-._~!$&'()*+,;=:@"), "/a/b-._~!$&'()*+,;=:@");
        assert_eq!(encoded("/%41 ?#"), "/%2541%20%3F%23");
        assert_eq!(encoded("/\u{e9}"), "/%C3%A9");
        assert_eq!(decoded(&encoded("/%41\u{fffd}")), "/%41\u{fffd}");
    }

    #[test]
    fn compares_in_full() {
        assert!(same(b"secret", b"secret"));
        assert!(!same(b"secret", b"secreT"));
        assert!(!same(b"secret", b"secrets"));
        assert!(same(b"", b""));
    }
}
