//! Test support: a stand-in for S3 on a local port, for faults LocalStack
//! cannot make (one object served under every key, HEAD and GET, ranged
//! or whole, the first GET bodies cut off halfway on demand, and a record
//! of every GET; or a server that reads requests and never answers), and
//! the stores the tests run on it.

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

pub(crate) struct FakeS3Server {
    /// `http://127.0.0.1:<port>`.
    pub(crate) endpoint: String,
    gets: Arc<Mutex<Vec<Get>>>,
}

impl FakeS3Server {
    /// Serves `data`, cutting the first `cuts` GET bodies off halfway.
    pub(crate) fn serving(data: Vec<u8>, cuts: usize) -> Self {
        let gets = Arc::new(Mutex::new(Vec::new()));
        let object = Object {
            data,
            cuts: AtomicUsize::new(cuts),
            gets: Arc::clone(&gets),
        };
        Self::start(Some(object), gets)
    }

    /// Reads requests and never answers them.
    pub(crate) fn silent() -> Self {
        Self::start(None, Arc::default())
    }

    /// The GETs so far, in order of arrival.
    pub(crate) fn gets(&self) -> Vec<Get> {
        self.gets.lock().unwrap().clone()
    }

    fn start(object: Option<Object>, gets: Arc<Mutex<Vec<Get>>>) -> Self {
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
        Self { endpoint, gets }
    }
}

struct Object {
    data: Vec<u8>,
    /// GET bodies still to cut off.
    cuts: AtomicUsize,
    gets: Arc<Mutex<Vec<Get>>>,
}

impl Object {
    /// Answers one request. Returns false once the connection is closed.
    fn respond(&self, request: &Request, out: &mut TcpStream) -> bool {
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
    /// Names in lowercase.
    headers: Vec<(String, String)>,
}

impl Request {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, value)| value.as_str())
    }
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

/// A request's line and headers; HEAD and GET carry no body.
fn read_request(reader: &mut impl BufRead) -> Option<Request> {
    let mut line = String::new();
    if reader.read_line(&mut line).ok()? == 0 {
        return None;
    }
    let method = line.split_whitespace().next()?.to_owned();
    let mut headers = Vec::new();
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).ok()? == 0 {
            return None;
        }
        let line = line.trim_end();
        if line.is_empty() {
            return Some(Request { method, headers });
        }
        let (name, value) = line.split_once(':')?;
        headers.push((name.trim().to_ascii_lowercase(), value.trim().to_owned()));
    }
}
