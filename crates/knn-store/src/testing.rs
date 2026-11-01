//! Test support: a stand-in for S3 on a local port, for faults LocalStack
//! cannot make (one object served under every key, HEAD and GET, ranged
//! or whole, the first GET bodies cut off halfway on demand, PUT taken
//! and dropped, a bucket in another region than the client's, and a
//! record of every GET and of every request's path and the region it
//! was signed for; or a server that reads requests and never answers),
//! and the stores the tests run on it.

use std::io::{BufRead, BufReader, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use aws_config::ConfigLoader;
use aws_sdk_s3::config::{Credentials, Region};

use crate::store::loader;
use crate::{Settings, Store, Transfers};

/// The service's SDK settings, with a region and static credentials, so
/// a test looks nothing up.
pub(crate) fn test_loader() -> ConfigLoader {
    loader()
        .region(Region::from_static("us-east-1"))
        .credentials_provider(Credentials::new("test", "test", None, None, "test"))
}

/// A store with [`test_loader`]'s settings, on `fake` when given.
pub(crate) async fn test_store(fake: Option<&FakeS3Server>, transfers: Transfers) -> Store {
    let settings = Settings {
        endpoint_url: fake.map(|fake| fake.endpoint.clone()),
        region: None,
    };
    Store::with_config(&test_loader().load().await, &settings, transfers)
}

const E_TAG: &str = "\"fake-e-tag\"";

/// A GET's headers, as the fake saw them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Get {
    pub(crate) range: Option<String>,
    pub(crate) if_match: Option<String>,
}

/// A bucket in another region than the client's: the fake answers a
/// request signed for any region but `region` with `status`, and S3's
/// header for the bucket's region holds the `names` entry for that
/// answer (the last one again once they run out; an empty entry, or
/// none left, leaves the header out). A `HeadBucket` signed for the
/// region is answered 200 with the header, as S3 answers it.
pub(crate) struct Elsewhere {
    pub(crate) region: String,
    pub(crate) status: u16,
    pub(crate) names: Vec<String>,
}

/// A request as the fake saw it: its method, its path without the
/// query, and the region its signature names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Signed {
    pub(crate) method: String,
    pub(crate) path: String,
    pub(crate) region: Option<String>,
}

/// Every request so far, in order of arrival.
type Requests = Arc<Mutex<Vec<Signed>>>;

pub(crate) struct FakeS3Server {
    /// `http://127.0.0.1:<port>`.
    pub(crate) endpoint: String,
    gets: Arc<Mutex<Vec<Get>>>,
    signed: Requests,
}

impl FakeS3Server {
    /// Serves `data`, cutting the first `cuts` GET bodies off halfway.
    pub(crate) fn serving(data: Vec<u8>, cuts: usize) -> Self {
        Self::serving_from(data, cuts, None)
    }

    /// Serves `data` from a bucket in `elsewhere`'s region.
    pub(crate) fn serving_elsewhere(data: Vec<u8>, elsewhere: Elsewhere) -> Self {
        Self::serving_from(data, 0, Some(elsewhere))
    }

    fn serving_from(data: Vec<u8>, cuts: usize, elsewhere: Option<Elsewhere>) -> Self {
        let gets = Arc::new(Mutex::new(Vec::new()));
        let signed = Requests::default();
        let object = Object {
            data,
            cuts: AtomicUsize::new(cuts),
            elsewhere,
            answered_elsewhere: AtomicUsize::new(0),
            gets: Arc::clone(&gets),
            signed: Arc::clone(&signed),
        };
        Self::start(Some(object), gets, signed)
    }

    /// Reads requests and never answers them.
    pub(crate) fn silent() -> Self {
        Self::start(None, Arc::default(), Requests::default())
    }

    /// The GETs so far, in order of arrival.
    pub(crate) fn gets(&self) -> Vec<Get> {
        self.gets.lock().unwrap().clone()
    }

    /// Every request so far, in order of arrival.
    pub(crate) fn signed(&self) -> Vec<Signed> {
        self.signed.lock().unwrap().clone()
    }

    fn start(object: Option<Object>, gets: Arc<Mutex<Vec<Get>>>, signed: Requests) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let object = object.map(Arc::new);
        thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { return };
                let object = object.clone();
                thread::spawn(move || serve(stream, object.as_deref()));
            }
        });
        Self {
            endpoint,
            gets,
            signed,
        }
    }
}

struct Object {
    data: Vec<u8>,
    /// GET bodies still to cut off.
    cuts: AtomicUsize,
    elsewhere: Option<Elsewhere>,
    /// Answers given for the wrong region so far.
    answered_elsewhere: AtomicUsize,
    gets: Arc<Mutex<Vec<Get>>>,
    signed: Requests,
}

impl Object {
    /// Answers one request. Returns false once the connection is closed.
    fn respond(&self, request: &Request, out: &mut TcpStream) -> bool {
        let signed = request.signed_region();
        self.signed.lock().unwrap().push(Signed {
            method: request.method.clone(),
            path: request.path.clone(),
            region: signed.map(str::to_owned),
        });
        if let Some(elsewhere) = &self.elsewhere {
            if signed != Some(elsewhere.region.as_str()) {
                let reason = match elsewhere.status {
                    301 => "Moved Permanently",
                    400 => "Bad Request",
                    _ => "Elsewhere",
                };
                let answered = self.answered_elsewhere.fetch_add(1, Ordering::SeqCst);
                let named = match elsewhere.names.get(answered).or(elsewhere.names.last()) {
                    Some(name) if !name.is_empty() => {
                        format!("x-amz-bucket-region: {name}\r\n")
                    }
                    _ => String::new(),
                };
                let head = format!(
                    "HTTP/1.1 {} {reason}\r\n{named}content-length: 0\r\n\r\n",
                    elsewhere.status
                );
                return out.write_all(head.as_bytes()).is_ok();
            }
            if request.is_bucket() {
                let head = format!(
                    "HTTP/1.1 200 OK\r\nx-amz-bucket-region: {}\r\ncontent-length: 0\r\n\r\n",
                    elsewhere.region
                );
                return out.write_all(head.as_bytes()).is_ok();
            }
        }
        if request.method == "PUT" {
            let head = format!("HTTP/1.1 200 OK\r\ncontent-length: 0\r\netag: {E_TAG}\r\n\r\n");
            return out.write_all(head.as_bytes()).is_ok();
        }
        let len = self.data.len();
        if request.method == "HEAD" {
            let head = format!("HTTP/1.1 200 OK\r\ncontent-length: {len}\r\netag: {E_TAG}\r\n\r\n");
            return out.write_all(head.as_bytes()).is_ok();
        }
        let range = request.header("range");
        self.gets.lock().unwrap().push(Get {
            range: range.map(str::to_owned),
            if_match: request.header("if-match").map(str::to_owned),
        });
        let (status, start, end) = match range.and_then(parse_range) {
            Some((first, last)) => ("206 Partial Content", first, last + 1),
            None => ("200 OK", 0, len),
        };
        let body = &self.data[start..end];
        let content_range = match range {
            Some(_) => format!("content-range: bytes {start}-{}/{len}\r\n", end - 1),
            None => String::new(),
        };
        let head = format!(
            "HTTP/1.1 {status}\r\ncontent-length: {}\r\netag: {E_TAG}\r\n{content_range}\r\n",
            body.len()
        );
        let cut = self
            .cuts
            .try_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
            .is_ok();
        let sent = if cut { &body[..body.len() / 2] } else { body };
        let written = out
            .write_all(head.as_bytes())
            .and_then(|()| out.write_all(sent))
            .and_then(|()| out.flush())
            .is_ok();
        if cut {
            // The declared length never arrives: the client sees the body
            // end early.
            _ = out.shutdown(Shutdown::Both);
            return false;
        }
        written
    }
}

/// `bytes=<first>-<last>`.
fn parse_range(value: &str) -> Option<(usize, usize)> {
    let (first, last) = value.strip_prefix("bytes=")?.split_once('-')?;
    Some((first.parse().ok()?, last.parse().ok()?))
}

struct Request {
    method: String,
    /// Without the query.
    path: String,
    /// Names in lowercase.
    headers: Vec<(String, String)>,
}

impl Request {
    /// Whether the path names a bucket alone, as path-style
    /// `HeadBucket` does, and no key.
    fn is_bucket(&self) -> bool {
        self.path
            .trim_start_matches('/')
            .split('/')
            .nth(1)
            .is_none_or(str::is_empty)
    }

    /// The region the `SigV4` signature names in its credential scope,
    /// `<key>/<date>/<region>/s3/aws4_request`.
    fn signed_region(&self) -> Option<&str> {
        let authorization = self.header("authorization")?;
        let (_, scope) = authorization.split_once("Credential=")?;
        scope.split('/').nth(2)
    }

    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, value)| value.as_str())
    }
}

/// Reads a request's body past: `content-length` bytes, or chunks and
/// trailers under `transfer-encoding: chunked`.
fn drop_body(reader: &mut impl BufRead, request: &Request) -> Option<()> {
    let chunked = request
        .header("transfer-encoding")
        .is_some_and(|value| value.eq_ignore_ascii_case("chunked"));
    if !chunked {
        let len: usize = request
            .header("content-length")
            .and_then(|len| len.parse().ok())
            .unwrap_or(0);
        return drop_bytes(reader, len);
    }
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).ok()?;
        let size = line.trim_end().split(';').next()?;
        let size = usize::from_str_radix(size, 16).ok()?;
        if size == 0 {
            // The trailers, up to the empty line.
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).ok()?;
                if line.trim_end().is_empty() {
                    return Some(());
                }
            }
        }
        // The chunk and the line end after it.
        drop_bytes(reader, size + 2)?;
    }
}

fn drop_bytes(reader: &mut impl BufRead, len: usize) -> Option<()> {
    let mut dropped = vec![0; len];
    reader.read_exact(&mut dropped).ok()
}

/// Answers the requests on one connection until either side closes it;
/// without an object, reads them and answers none.
fn serve(stream: TcpStream, object: Option<&Object>) {
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut out = stream;
    while let Some(request) = read_request(&mut reader) {
        if let Some(object) = object
            && !object.respond(&request, &mut out)
        {
            return;
        }
    }
}

/// A request's line and headers, its body (a PUT's, sized by its
/// `content-length` or chunked) read and dropped.
fn read_request(reader: &mut impl BufRead) -> Option<Request> {
    let mut line = String::new();
    if reader.read_line(&mut line).ok()? == 0 {
        return None;
    }
    let mut words = line.split_whitespace();
    let method = words.next()?.to_owned();
    let path = words.next()?.split('?').next()?.to_owned();
    let mut headers = Vec::new();
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).ok()? == 0 {
            return None;
        }
        let line = line.trim_end();
        if line.is_empty() {
            let request = Request {
                method,
                path,
                headers,
            };
            drop_body(reader, &request)?;
            return Some(request);
        }
        let (name, value) = line.split_once(':')?;
        headers.push((name.trim().to_ascii_lowercase(), value.trim().to_owned()));
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::read_request;

    /// Every request on a connection parses, whatever body the one before
    /// it carried.
    #[test]
    fn reads_bodies_past() {
        let requests = "PUT /bucket/a?x-id=PutObject HTTP/1.1\r\ncontent-length: 5\r\n\r\nhello\
            PUT /bucket/b HTTP/1.1\r\ntransfer-encoding: chunked\r\n\r\n\
            3\r\nabc\r\n2;ext=1\r\nde\r\n0\r\nx-amz-checksum-crc32: AAAAAA==\r\n\r\n\
            HEAD /bucket/ HTTP/1.1\r\n\r\n";
        let mut reader = Cursor::new(requests.as_bytes());
        let paths: Vec<(String, String, bool)> = std::iter::from_fn(|| read_request(&mut reader))
            .map(|request| {
                let bucket = request.is_bucket();
                (request.method, request.path, bucket)
            })
            .collect();
        assert_eq!(
            paths,
            [
                ("PUT".to_owned(), "/bucket/a".to_owned(), false),
                ("PUT".to_owned(), "/bucket/b".to_owned(), false),
                ("HEAD".to_owned(), "/bucket/".to_owned(), true),
            ]
        );
    }
}
