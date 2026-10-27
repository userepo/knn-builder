//! Serializes the Faiss index file a remote build hands back to OpenSearch
//! k-NN: an ID map over an `IndexHNSWCagra`, written from a CAGRA graph,
//! the vectors and their ids without Faiss, in file order and with the
//! total length known before the first byte, so it can be streamed into
//! an upload.
//!
//! Nothing is implemented yet.
