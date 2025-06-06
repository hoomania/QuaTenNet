use ndarray::{s, Array1, ArrayD, IxDyn};

/// Computes the trace of a tensor along specified axes.
///
/// The trace is calculated by summing the diagonal elements of the tensor
/// along the specified axes. The function requires exactly two axes to
/// be specified, which must have the same size.
///
/// # Parameters
///
/// - `tensor`: A reference to a tensor of type `ArrayD<f64>`. This is the tensor
///   for which the trace will be calculated.
/// - `axes`: A vector of `usize` containing exactly two axes indices along which
///   the trace will be computed.
///
/// # Returns
///
/// - `Result<ArrayD<f64>, String>`: Returns a `Result` containing either:
///   - `Ok(ArrayD<f64>)`: The resulting tensor after computing the trace.
///   - `Err(String)`: An error message if the input is invalid or if there is a shape mismatch.
///
/// # Errors
///
/// The function may return an error in the following cases:
/// - If the length of `axes` is not exactly 2.
/// - If the sizes of the specified axes in the tensor do not match.
pub fn trace(tensor: &ArrayD<f64>, axes: Vec<usize>) -> Result<ArrayD<f64>, String> {
    // Check if exactly two axes are provided
    if axes.len() != 2 {
        return Err(format!(
            "Trace calculation need two axes index. (Axes length is {}!)",
            axes.len()
        )
        .to_string());
    }

    let t_shape = tensor.shape().to_vec();

    // Check if the sizes of the specified axes are the same
    if t_shape[axes[0]] != t_shape[axes[1]] {
        return Err(format!(
            "Shape mismatch along specified axes: tenosr[{}] = {}, tensor[{}] = {}",
            axes[0], t_shape[axes[0]], axes[1], t_shape[axes[1]]
        )
        .to_string());
    }

    // Identify axes in the tensor that are not involved in the trace calculation
    let notin: Vec<usize> = (0..tensor.ndim())
        .filter(|&k| !axes.contains(&(k as usize)))
        .collect();

    // Get the shapes of the axes that are not involved in the trace
    let notin_shape: Vec<_> = notin.iter().map(|&ndx| t_shape[ndx]).collect();

    let r_shape_dim: Vec<usize> = vec![notin.iter().map(|&ndx| t_shape[ndx]).product()];

    let new_arrange = [axes.clone(), notin.clone()].concat();

    let t_permuted = tensor
        .view()
        .permuted_axes(IxDyn(&new_arrange))
        .to_shape(
            [
                vec![t_shape[axes[0]], t_shape[axes[0]]],
                r_shape_dim.clone(),
            ]
            .concat(),
        )
        .expect("Failed to reshape permuted tensor")
        .into_owned();

    let mut result = Array1::<f64>::zeros(r_shape_dim[0]);

    for i in 0..t_shape[axes[0]] {
        let slice = t_permuted.slice(s![i, i, ..]);
        result = &result + &slice;
    }

    Ok(result
        .to_shape(notin_shape)
        .expect("Failed to reshape output (trace)")
        .into_owned())
}
