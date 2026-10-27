//! With the `gpu` feature, links this crate's targets (its tests among
//! them) so they find libcuvs and the CUDA runtime at run time. cuvs-sys
//! reports the conda environment's library directory as `DEP_CUVS_LIB`;
//! the cuvs crate adds it as an rpath only to its own targets. A binary
//! in another crate that links knn-gpu needs the same rpath.

fn main() {
    println!("cargo::rerun-if-env-changed=DEP_CUVS_LIB");
    if let Ok(lib) = std::env::var("DEP_CUVS_LIB") {
        println!("cargo::rustc-link-search=native={lib}");
        println!("cargo::rustc-link-arg=-Wl,-rpath,{lib}");
    }
}
