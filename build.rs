fn main() {
    // Link to the LAPACK library
    println!("cargo:rustc-link-lib=lapack");
    // Link to the BLAS library
    println!("cargo:rustc-link-lib=blas");
    // Link to the OpenBLAS library
    println!("cargo:rustc-link-lib=openblas");
}
