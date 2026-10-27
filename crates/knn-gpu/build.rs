//! With the `gpu` feature, builds the C++ shim over cuVS (`cpp/`) with
//! CMake against the libcuvs of the active conda environment
//! (`CONDA_PREFIX`, see `environment.yml`), and links it with libcuvs and
//! the libraries its inlined code calls. The rpath it adds covers this
//! crate's own targets (its tests and examples); a binary in another crate
//! that links knn-gpu needs the same rpath.

fn main() {
    #[cfg(feature = "gpu")]
    gpu();
}

#[cfg(feature = "gpu")]
fn gpu() {
    println!("cargo::rerun-if-changed=cpp");
    println!("cargo::rerun-if-env-changed=CONDA_PREFIX");
    let prefix = std::env::var("CONDA_PREFIX").expect(
        "the gpu feature builds against the conda environment in environment.yml; activate it",
    );

    let shim = cmake::Config::new("cpp")
        .define("CMAKE_PREFIX_PATH", &prefix)
        .build();
    println!(
        "cargo::rustc-link-search=native={}",
        shim.join("lib").display()
    );
    println!("cargo::rustc-link-lib=static=knn_cagra");

    let lib = format!("{prefix}/lib");
    println!("cargo::rustc-link-search=native={lib}");
    for name in ["cuvs", "rmm", "rapids_logger", "cudart", "stdc++"] {
        println!("cargo::rustc-link-lib=dylib={name}");
    }
    println!("cargo::rustc-link-arg=-Wl,-rpath,{lib}");
}
