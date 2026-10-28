//! With the `gpu` feature, adds libcuvs's directory in the active conda
//! environment (`CONDA_PREFIX`) to this crate's rpath: knn-gpu's own rpath
//! covers only its own targets, and this crate's binary and tests link it
//! too.

fn main() {
    #[cfg(feature = "gpu")]
    gpu();
}

#[cfg(feature = "gpu")]
fn gpu() {
    println!("cargo::rerun-if-env-changed=CONDA_PREFIX");
    let prefix = std::env::var("CONDA_PREFIX").expect(
        "the gpu feature builds against the conda environment in environment.yml; activate it",
    );
    println!("cargo::rustc-link-arg=-Wl,-rpath,{prefix}/lib");
}
