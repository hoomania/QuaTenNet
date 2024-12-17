use ndarray::*;

pub fn tensor_dot(
    a: &ArrayD<f64>,
    b: &ArrayD<f64>,
    axis_vec: Vec<usize>,
) -> Result<ArrayD<f64>, String> {
    if axis_vec.len() % 2 != 0 {
        return Err("Axis length is not even number!".to_string());
    }

    let axis = Array2::from_shape_vec((2, axis_vec.len() / 2), axis_vec).unwrap();
    let axes_a = axis.index_axis(Axis(0), 0).to_vec();
    let axes_b = axis.index_axis(Axis(0), 1).to_vec();

    let ash = a.shape();
    let bsh = b.shape();

    for k in 0..axes_a.len() {
        if ash[axes_a[k] as usize] != bsh[axes_b[k] as usize] {
            return Err(format!(
                "Shape mismatch along specified axes: a[{}] = {}, b[{}] = {}",
                axes_a[k], ash[axes_a[k]], axes_b[k], bsh[axes_b[k]]
            )
            .to_string());
        }
    }

    // tensor a:
    let notin_a: Vec<usize> = (0..a.ndim())
        .filter(|&k| !axes_a.contains(&(k as usize)))
        .collect();

    let a_mpl_linked: usize = axes_a.iter().map(|&ndx| ash[ndx]).product();
    let a_mpl_unlinked: usize = notin_a.iter().map(|&ndx| ash[ndx]).product();

    let newaxes_a = [notin_a.clone(), axes_a].concat();

    // tensor b:
    let notin_b: Vec<usize> = (0..b.ndim())
        .filter(|&k| !axes_b.contains(&(k as usize)))
        .collect();

    let b_mpl_linked: usize = axes_b.iter().map(|&ndx| bsh[ndx]).product();
    let b_mpl_unlinked: usize = notin_b.iter().map(|&ndx| bsh[ndx]).product();

    let newaxes_b = [axes_b, notin_b.clone()].concat();

    // permutation & reshape to matrix
    let a_permute = a.view().permuted_axes(IxDyn(&newaxes_a));
    let a_reshape = a_permute.to_shape((a_mpl_unlinked, a_mpl_linked)).unwrap();

    let b_permute = b.view().permuted_axes(IxDyn(&newaxes_b));
    let b_reshape = b_permute.to_shape((b_mpl_linked, b_mpl_unlinked)).unwrap();

    // dot product
    let res = a_reshape.dot(&b_reshape).into_owned();

    // output shape
    let old_a: Vec<_> = notin_a.iter().map(|&ndx| ash[ndx]).collect();
    let old_b: Vec<_> = notin_b.iter().map(|&ndx| bsh[ndx]).collect();

    let output = res
        .to_shape([old_a, old_b].concat())
        .expect("Failed to reshape output")
        .into_owned();

    Ok(output)
}
