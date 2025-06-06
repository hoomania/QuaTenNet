use ndarray::{Array2, ArrayD, Axis, IxDyn};

/// Computes the tensor dot product of two tensors along specified axes.
///
/// This function takes two tensors and a vector of axes to contract over. The axes
/// must be specified in pairs, where each pair consists of an axis from the first
/// tensor and an axis from the second tensor. The function checks for shape compatibility
/// along the specified axes and performs the dot product accordingly.
///
/// # Parameters
///
/// - `a`: A reference to a tensor of type `ArrayD<f64>`. This is the first tensor
///   involved in the dot product.
/// - `b`: A reference to a tensor of type `ArrayD<f64>`. This is the second tensor
///   involved in the dot product.
/// - `axis_vec`: A vector of `usize` representing the axes to contract over. The length
///   of this vector must be even, as it specifies pairs of axes (one from `a` and one from `b`).
///
/// # Returns
///
/// - `Result<ArrayD<f64>, String>`: Returns a `Result` containing either:
///   - `Ok(ArrayD<f64>)`: The resulting tensor after performing the dot product.
///   - `Err(String)`: An error message if the input is invalid or if there is a shape mismatch
///     along the specified axes.
///
/// # Errors
///
/// The function may return an error in the following cases:
/// - If the length of `axis_vec` is not an even number.
/// - If the shapes of the specified axes in tensors `a` and `b` do not match.
pub fn tensor_dot(
    a: &ArrayD<f64>,
    b: &ArrayD<f64>,
    axis_vec: Vec<usize>,
) -> Result<ArrayD<f64>, String> {
    // Check if the length of axis_vec is even
    if axis_vec.len() % 2 != 0 {
        return Err("Axis length is not even number!".to_string());
    }

    // Create a 2D array from axis_vec to separate axes for a and b
    let axis = Array2::from_shape_vec((2, axis_vec.len() / 2), axis_vec).unwrap();
    let axes_a = axis.index_axis(Axis(0), 0).to_vec();
    let axes_b = axis.index_axis(Axis(0), 1).to_vec();

    let ash = a.shape();
    let bsh = b.shape();

    // Check for shape compatibility along the specified axes
    for k in 0..axes_a.len() {
        if ash[axes_a[k] as usize] != bsh[axes_b[k] as usize] {
            return Err(format!(
                "Shape mismatch along specified axes: a[{}] = {}, b[{}] = {}",
                axes_a[k], ash[axes_a[k]], axes_b[k], bsh[axes_b[k]]
            )
            .to_string());
        }
    }

    // Identify axes in tensor A that are not involved in the contraction
    let notin_a: Vec<usize> = (0..a.ndim())
        .filter(|&k| !axes_a.contains(&(k as usize)))
        .collect();

    // Calculate the product of sizes for linked and unlinked axes in tensor A
    let a_mpl_linked: usize = axes_a.iter().map(|&ndx| ash[ndx]).product();
    let a_mpl_unlinked: usize = notin_a.iter().map(|&ndx| ash[ndx]).product();

    // Create a new axes order for tensor A
    let newaxes_a = [notin_a.clone(), axes_a].concat();

    // Do same for tensor B:
    let notin_b: Vec<usize> = (0..b.ndim())
        .filter(|&k| !axes_b.contains(&(k as usize)))
        .collect();

    let b_mpl_linked: usize = axes_b.iter().map(|&ndx| bsh[ndx]).product();
    let b_mpl_unlinked: usize = notin_b.iter().map(|&ndx| bsh[ndx]).product();

    let newaxes_b = [axes_b, notin_b.clone()].concat();

    // Permute and reshape tensor A to a 2D matrix for dot product
    let a_permute = a.view().permuted_axes(IxDyn(&newaxes_a));
    let a_reshape = a_permute.to_shape((a_mpl_unlinked, a_mpl_linked)).unwrap();

    // Do same for tensor B:
    let b_permute = b.view().permuted_axes(IxDyn(&newaxes_b));
    let b_reshape = b_permute.to_shape((b_mpl_linked, b_mpl_unlinked)).unwrap();

    // Compute the dot product of the reshaped matrices
    let res = a_reshape.dot(&b_reshape).into_owned();

    // Determine the output shape based on the unlinked axes
    let old_a: Vec<_> = notin_a.iter().map(|&ndx| ash[ndx]).collect();
    let old_b: Vec<_> = notin_b.iter().map(|&ndx| bsh[ndx]).collect();

    let output = res
        .to_shape([old_a, old_b].concat())
        .expect("Failed to reshape output")
        .into_owned();

    Ok(output)
}
