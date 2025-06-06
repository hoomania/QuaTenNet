use ndarray::{arr1, Array1, Array2, ArrayD, IxDyn};
use ndarray_linalg::SVD;
use rand::Rng;

pub struct SVDResult {
    pub u: Array2<f64>,
    pub sigma_f64: Box<[f64]>,
    pub sigma: Array1<f64>,
    pub vt: Array2<f64>,
}

/// Creates a tensor of the specified shape, filled with the given value.
///
/// # Arguments
///
/// * `shape` - A slice of `usize` representing the dimensions of the tensor.
/// * `fill` - A `f64` value to fill the tensor with.
///
/// # Returns
///
/// An `ArrayD<f64>` representing the tensor.
pub fn tensor(shape: &[usize], fill: f64) -> ArrayD<f64> {
    let size = shape.iter().product();
    ArrayD::from_shape_vec(IxDyn(shape), vec![fill; size]).expect("ShapeError!")
}

/// Creates a tensor of the specified shape, filled with zeros.
///
/// # Arguments
///
/// * `shape` - A slice of `usize` representing the dimensions of the tensor.
///
/// # Returns
///
/// An `ArrayD<f64>` representing the tensor filled with zeros.
pub fn zeros(shape: &[usize]) -> ArrayD<f64> {
    tensor(shape, 0.0)
}

/// Creates a tensor of the specified shape, filled with ones.
///
/// # Arguments
///
/// * `shape` - A slice of `usize` representing the dimensions of the tensor.
///
/// # Returns
///
/// An `ArrayD<f64>` representing the tensor filled with ones.
pub fn ones(shape: &[usize]) -> ArrayD<f64> {
    tensor(shape, 1.0)
}

/// Creates an identity matrix of the specified size.
///
/// # Arguments
///
/// * `size` - The size of the identity matrix (number of rows and columns).
///
/// # Returns
///
/// An `Array2<f64>` representing the identity matrix.
pub fn identity(size: usize) -> Array2<f64> {
    Array2::eye(size)
}

/// Creates a diagonal matrix from the given diagonal elements.
///
/// # Arguments
///
/// * `diag` - A slice of `f64` representing the diagonal elements.
///
/// # Returns
///
/// An `Array2<f64>` representing the diagonal matrix.
pub fn diagonal(diag: &[f64]) -> Array2<f64> {
    Array2::from_diag(&arr1(diag))
}

/// Creates a tensor of the specified shape, filled with random values in the range [0.0, 1.0].
///
/// # Arguments
///
/// * `shape` - A slice of `usize` representing the dimensions of the tensor.
///
/// # Returns
///
/// An `ArrayD<f64>` representing the tensor filled with random values.
pub fn random(shape: &[usize]) -> ArrayD<f64> {
    let size = shape.iter().product();
    let mut rng = rand::thread_rng(); // Use thread_rng for random number generation
    let rnd_values: Vec<f64> = (0..size).map(|_| rng.random_range(0.0..=1.0)).collect();
    ArrayD::from_shape_vec(IxDyn(shape), rnd_values).expect("ShapeError!")
}

/// Performs Singular Value Decomposition (SVD) on the given 2D array.
///
/// # Arguments
///
/// * `arr` - A 2D array of type `Array2<f64>` to perform SVD on.
///
/// # Returns
///
/// A `Result<SVDResult, String>` where:
/// - `Ok(SVDResult)` contains the SVD results (U, sigma, VT).
/// - `Err(String)` contains an error message if the SVD operation fails.
pub fn svd(arr: Array2<f64>) -> Result<SVDResult, String> {
    let (u, sigma, vt) = arr
        .svd(true, true)
        .map_err(|err| format!("SVD error: {:?}", err))?;

    let u = u.ok_or_else(|| "U matrix is None".to_string())?;
    let vt = vt.ok_or_else(|| "VT matrix is None".to_string())?;
    let sigma_f64: &[f64] = sigma
        .as_slice()
        .ok_or_else(|| "Sigma is empty".to_string())?;

    Ok(SVDResult {
        u,
        sigma_f64: sigma_f64.into(),
        sigma,
        vt,
    })
}
