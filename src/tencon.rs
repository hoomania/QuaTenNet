use crate::tendot::*;
use crate::trace::*;
use ndarray::{Array2, ArrayD, IxDyn};
use std::collections::{HashMap, HashSet};

/// Contracts a list of tensors according to a specified contraction order.
///
/// This function performs tensor contractions based on the provided contraction order,
/// which specifies how the tensors should be combined. It validates the indices, processes
/// the tensors to ensure they have compatible shapes, and executes the contractions in the
/// specified order using a greedy approach to minimize the computational cost. The result is
/// a single tensor that is the result of all specified contractions.
///
/// # Arguments
/// - `tensors`: A vector of `ArrayD<f64>` representing the tensors to be contracted.
/// - `contraction_order`: A vector of vectors containing integers that specify the order of contraction.
///   Each integer should appear exactly twice for shared indices (for contraction) and at most once for
///   unique indices (for non-contracted dimensions).
///
/// # Returns
/// A `Result<ArrayD<f64>, String>` where:
/// - `Ok(ArrayD<f64>)` contains the resulting tensor after all contractions are performed.
/// - `Err(String)` contains an error message if the contraction order is invalid or if any other error occurs.
///
/// # Errors
/// This function may return an error if:
/// - The indices in `contraction_order` are not valid (e.g., an index appears the wrong number of times).
/// - There are issues during tensor operations such as shape mismatches or invalid contractions.
pub fn contract(
    mut tensors: Vec<ArrayD<f64>>,
    mut contraction_order: Vec<Vec<i32>>,
) -> Result<ArrayD<f64>, String> {
    indices_validation(&contraction_order)?;
    prepare_contraction_data(&mut tensors, &mut contraction_order);

    let mut ten_list = tensors;
    let mut cnt_order = contraction_order;

    // Generate a contraction plan using a greedy algorithm
    let contraction_plan = contract_map(&ten_list, &cnt_order);

    for pair in contraction_plan {
        for &i in &pair {
            trace_check(&mut ten_list[i], &mut cnt_order[i])?;
        }

        let axes = order_to_index(&cnt_order, &pair);
        let contraction = tensor_dot(&ten_list[pair[0]], &ten_list[pair[1]], axes)?;

        ten_list[pair[0]] = contraction;
        ten_list.remove(pair[1]);
        order_reformat(&mut cnt_order, &pair);
    }

    Ok(final_order(ten_list.remove(0), cnt_order))
}

/// Validates the indices in the contraction order for tensor operations.
///
/// This function checks that the indices specified in the contraction order meet the required
/// conditions for valid tensor contractions. Specifically, it ensures that:
/// - Positive indices appear exactly twice, indicating that they are shared dimensions for contraction.
/// - Negative indices appear at most once, indicating that they are unique dimensions that should not be contracted.
///
/// # Arguments
/// - `order`: A reference to a vector of vectors containing integers that specify the contraction
///   order for the tensors. Each inner vector represents the order of dimensions for a specific tensor.
///
/// # Returns
/// A `Result<(), String>` where:
/// - `Ok(())` indicates that the validation was successful and all indices are valid.
/// - `Err(String)` contains an error message if any index fails the validation checks.
///
/// # Notes
/// This function is crucial for ensuring that tensor contractions are performed correctly,
/// preventing runtime errors due to invalid index configurations. It should be called before
/// attempting to perform any tensor contractions.
fn indices_validation(order: &[Vec<i32>]) -> Result<(), String> {
    let mut counts = HashMap::new();

    // Count how many times each index appears in the contraction order
    for &val in order.iter().flatten() {
        *counts.entry(val).or_insert(0) += 1;
    }
    for (&key, &count) in &counts {
        // Check if positive indices appear exactly twice
        if key > 0 && count != 2 {
            return Err(format!(
                "Index {} must appear exactly twice in contraction order list.",
                key
            ));
        }

        // Check if negative indices appear at most once
        if key < 0 && count > 1 {
            return Err(format!(
                "Index {} must appear at most once in contraction order list.",
                key
            ));
        }
    }
    Ok(())
}

/// Prepares the tensors and contraction orders for tensor contraction operations.
///
/// This function modifies the input tensors and their corresponding contraction orders to ensure
/// that they are compatible for contraction. It performs the following tasks:
/// - Expands the shapes of the tensors to include extra dimensions, ensuring that all tensors
///   have compatible shapes for the contraction process.
/// - Updates the contraction orders to reflect the new shapes of the tensors, adding new indices
///   for the extra dimensions.
///
/// # Arguments
/// - `tensors`: A mutable reference to a vector of `ArrayD<f64>` representing the tensors to be
///   contracted. This will be modified to include extra dimensions as needed.
/// - `orders`: A mutable reference to a vector of vectors containing integers that specify the
///   order of dimensions for each tensor. This will be updated to reflect the new contraction
///   orders after processing.
///
/// # Notes
/// This function assumes that the contraction orders are valid and that the tensors are properly
/// initialized. It modifies the tensors and orders in place, so the original vectors will be
/// updated directly. This function should be called before performing any tensor contractions.
fn prepare_contraction_data(tensors: &mut Vec<ArrayD<f64>>, orders: &mut Vec<Vec<i32>>) {
    let max_idx = orders.iter().flatten().cloned().max().unwrap_or(0);
    let ten_len = tensors.len();
    let extra_dims: Vec<usize> = vec![1; ten_len - 1];

    // Expand the shape of each tensor except the last one to include an extra dimension
    for i in 0..ten_len - 1 {
        tensors[i] = tensors[i]
            .to_shape([tensors[i].shape(), &[1]].concat())
            .unwrap()
            .to_owned();
    }

    let last_shape = tensors.last().unwrap().shape().to_vec();

    // Expand the last tensor's shape to include extra dimensions
    tensors[ten_len - 1] = tensors
        .last_mut()
        .unwrap()
        .to_shape([&last_shape, &extra_dims[..]].concat())
        .unwrap()
        .to_owned();

    let mut new_orders = Vec::new();
    let mut new_dims = Vec::new();

    // Update the contraction orders to reflect the new shapes of the tensors
    for (i, order) in orders.iter().enumerate().take(orders.len() - 1) {
        let ext = max_idx + i as i32 + 1;
        new_orders.push([order.clone(), vec![ext]].concat());
        new_dims.push(ext);
    }

    // Append the last order with its new dimensions
    new_orders.push([orders.last().unwrap().clone(), new_dims].concat());
    *orders = new_orders;
}

/// Generates a contraction plan for a list of tensors based on their shapes and contraction orders.
///
/// This function creates a plan for contracting tensors by selecting pairs of tensors to be
/// contracted based on a greedy algorithm. It evaluates the shapes and orders of the tensors
/// to determine the best pairs to contract in order to minimize computational cost. The function
/// continues to generate pairs until only one tensor remains.
///
/// # Arguments
/// - `tensors`: A reference to a slice of `ArrayD<f64>` representing the tensors to be contracted.
/// - `orders`: A reference to a vector of vectors containing integers that specify the order of
///   dimensions for each tensor. This is used to guide the contraction process.
///
/// # Returns
/// A `Vec<Vec<usize>>` representing the contraction plan, where each inner vector contains the
/// indices of the tensors to be contracted in each step of the plan.
///
/// # Notes
/// This function assumes that the shapes and orders of the tensors are valid and that the
/// tensors are properly initialized. The contraction plan generated by this function should be
/// used to guide the actual contraction operations in a subsequent step.
pub fn contract_map(tensors: &[ArrayD<f64>], orders: &[Vec<i32>]) -> Vec<Vec<usize>> {
    let mut shapes = shape_vec(tensors);
    let mut contraction_orders = orders.to_vec();
    let mut plan = Vec::new();

    while contraction_orders.len() > 1 {
        let (ratios, row_sums) = ratio_matrix(&shapes, &contraction_orders);
        let nodes = select_best_nodes(&ratios, &row_sums);
        plan.push(nodes.clone());
        shape_reformat(&mut shapes, &contraction_orders, &nodes);
        order_reformat(&mut contraction_orders, &nodes);
    }

    plan
}

/// Checks for indices that need to be traced in the given tensor and performs the trace operation.
///
/// The function identifies pairs of indices in the contraction order that appear exactly twice,
/// indicating that a trace operation should be performed. It then traces the tensor along these
/// indices and removes them from the contraction order.
///
/// # Arguments
/// - `tensor`: A mutable reference to an `ArrayD<f64>` representing the tensor to be traced.
/// - `order`: A mutable reference to a vector of integers representing the contraction order of the tensor.
///
/// # Returns
/// - `Result<(), String>`: Returns `Ok(())` if the trace operation is successful. If an error occurs,
///   it returns an `Err` containing a descriptive error message.
///
/// # Errors
/// This function may return an error if the trace operation cannot be performed due to invalid
/// indices or if the tensor does not have the required dimensions for tracing.
///
/// # Note
/// The function modifies the input tensor and contraction order in place. It is important to ensure
/// that the contraction order is correctly specified before calling this function, as incorrect
/// orders may lead to runtime errors or unexpected behavior. Additionally, the tensor must have
/// dimensions that correspond to the indices being traced.
fn trace_check(tensor: &mut ArrayD<f64>, order: &mut Vec<i32>) -> Result<(), String> {
    let mut index_map = HashMap::new();
    for (i, &val) in order.iter().enumerate() {
        index_map.entry(val).or_insert_with(Vec::new).push(i);
    }

    // Check for pairs of indices that need to be traced
    for indices in index_map.values() {
        // If an index appears exactly twice, it indicates a trace operation
        if indices.len() == 2 {
            let trace_axes: Vec<usize> = indices.iter().map(|&i| i).collect();
            *tensor = trace(tensor, trace_axes)?;

            // Remove the traced indices from the order
            for &i in indices.iter().rev() {
                order.remove(i);
            }
        }
    }
    Ok(())
}

/// Computes a matrix of ratios representing the shared dimensions between pairs of tensors,
/// along with a vector of row sums.
///
/// The ratio for each pair of tensors is calculated as the sum of the dimensions of their
/// shared indices divided by the product of their sizes. This helps in determining the best
/// pairs of tensors to contract based on their shared dimensions.
///
/// # Arguments
/// - `shapes`: A slice of vectors, where each vector contains the dimensions of a tensor.
/// - `orders`: A slice of vectors representing the contraction orders for each tensor.
///
/// # Returns
/// - `(Array2<f64>, Vec<f64>)`: A tuple containing:
///   - An `Array2<f64>` representing the matrix of ratios for each pair of tensors.
///   - A `Vec<f64>` containing the sum of the ratios for each row in the matrix that contains
///     the maximum ratio value.
///
/// # Note
/// The function assumes that the input shapes and orders are valid and correspond to the same
/// number of tensors. The resulting ratio matrix is symmetric, as the ratio between tensor A and
/// tensor B is the same as that between tensor B and tensor A. The row sums are particularly useful
/// for selecting the best nodes for contraction, as they highlight which tensors share the most
/// dimensions relative to their sizes.
fn ratio_matrix(shapes: &[Vec<i32>], orders: &[Vec<i32>]) -> (Array2<f64>, Vec<f64>) {
    let n = orders.len();
    let mut matrix = Array2::zeros((n, n));

    // Calculate the ratio of shared dimensions for each pair of tensors
    for i in 0..n {
        for j in i + 1..n {
            // Find shared indices between the two tensors
            let shared: Vec<_> = orders[i].iter().filter(|x| orders[j].contains(x)).collect();
            let si: i32 = shapes[i].iter().product();
            let sj: i32 = shapes[j].iter().product();
            let mut val = 0.0;

            // Sum the dimensions of the shared indices
            for &s in &shared {
                if let Some(idx) = orders[i].iter().position(|x| *x == *s) {
                    val += shapes[i][idx] as f64;
                }
            }

            // Calculate the ratio of shared dimensions to the product of tensor sizes
            let r = val / ((si * sj) as f64);
            matrix[[i, j]] = r;
            matrix[[j, i]] = r;
        }
    }

    let max_val = matrix.iter().fold(f64::MIN, |a, &b| a.max(b));

    // Create a vector to store the sum of rows that contain the maximum value
    let row_sums: Vec<f64> = (0..n)
        .map(|i| {
            if matrix.row(i).iter().any(|&x| x == max_val) {
                matrix.row(i).iter().sum()
            } else {
                0.0
            }
        })
        .collect();

    (matrix, row_sums)
}

/// Selects the best pair of tensors to contract based on the maximum row sum.
///
/// This function identifies the two tensors with the highest ratio of shared
/// dimensions, which are the best candidates for contraction. It first finds
/// the index of the row with the maximum sum in the ratio matrix, and then
/// selects the column with the maximum value in that row to determine the
/// second tensor to contract.
///
/// # Arguments
///
/// * `ratio_matrix` - A 2D array representing the ratio matrix of tensor pairs.
/// * `row_sum` - A vector of floats representing the sum of ratios for each row.
///
/// # Returns
///
/// Returns a vector containing the indices of the two tensors selected for contraction.
///
/// # Note
/// The function assumes that the input `scores` vector is non-empty and that the `matrix` is
/// square (i.e., the number of rows equals the number of columns). It selects the best nodes
/// based on the maximum score found in the `scores` vector and the corresponding maximum value
/// in the respective row of the `matrix`. If there are multiple pairs with the same maximum
/// score, the first encountered pair will be selected.
fn select_best_nodes(matrix: &Array2<f64>, scores: &[f64]) -> Vec<usize> {
    let i = scores
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    let j = matrix
        .row(i)
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(j, _)| j)
        .unwrap();
    vec![i, j]
}

/// Converts contraction orders into indices for shared dimensions between two specified orders.
///
/// This function identifies the common indices shared between two contraction orders and returns
/// their positions in the respective orders. It is useful for determining which dimensions are
/// involved in a contraction operation.
///
/// # Arguments
/// - `order_list`: A slice of vectors representing the contraction orders for each tensor.
/// - `indexes`: A slice of indices indicating which two orders to compare.
///
/// # Returns
/// - `Vec<usize>`: A vector containing the indices of the common dimensions in the specified
///   contraction orders.
///
/// # Note
/// The function assumes that the provided indexes correspond to valid contraction orders in the
/// `order_list`. If the specified orders do not share any common indices, the resulting vector
/// will be empty. Care should be taken to ensure that the orders being compared are relevant to
/// the contraction operation being performed.
fn order_to_index(order_list: &[Vec<i32>], indexes: &[usize]) -> Vec<usize> {
    // Create a set of common indices shared between the two specified orders
    let common: HashSet<i32> = order_list[indexes[0]]
        .iter()
        .cloned()
        .collect::<HashSet<_>>()
        .intersection(&order_list[indexes[1]].iter().cloned().collect())
        .cloned()
        .collect();

    let mut idx = vec![];

    // Iterate over the specified indexes to find their positions in the order list
    for i in indexes {
        for ce in &common {
            if let Some(ndx) = order_list[*i].iter().position(|value| value == ce) {
                idx.push(ndx);
            }
        }
    }

    idx
}

/// Merges and reformats the contraction orders after a contraction operation.
///
/// This function combines the contraction orders of two tensors that have been contracted,
/// removing duplicates and ensuring that the resulting order is unique. The order of the
/// remaining indices is preserved, and the second order is removed from the list.
///
/// # Arguments
/// - `orders`: A mutable reference to a vector of vectors representing the contraction orders
///   for each tensor.
/// - `indices`: A slice of indices indicating which two orders to merge.
///
/// # Note
/// The function assumes that the indices provided correspond to valid orders in the `orders`
/// vector. After merging, the resulting order will only contain unique indices, which is crucial
/// for maintaining the integrity of the contraction process. Care should be taken to ensure that
/// the orders being merged are relevant to the current contraction operation.
fn order_reformat(orders: &mut Vec<Vec<i32>>, indices: &[usize]) {
    let (a, b) = (indices[0], indices[1]);
    let mut combined = orders[a].clone();
    combined.extend(orders[b].iter());

    let mut seen = HashSet::new();
    let mut merged = Vec::new();
    let mut duplicates = HashSet::new();

    // Iterate over the combined order to filter out duplicates
    for &val in &combined {
        // If the value is unique, add it to the merged vector
        // else it's a duplicate, add it to the duplicates set
        if seen.insert(val) {
            merged.push(val);
        } else {
            duplicates.insert(val);
        }
    }
    merged.retain(|x| !duplicates.contains(x));

    orders[a] = merged;
    orders.remove(b);
}

/// Reformats the shapes of tensors based on the specified contraction indices.
///
/// This function modifies the shapes of the tensors by merging the shapes of two tensors
/// that are being contracted. It removes any common dimensions from the shapes of the tensors
/// specified by the given indices, ensuring that the resulting shape reflects the contraction
/// operation correctly.
///
/// # Arguments
/// - `shapes`: A mutable reference to a vector of vectors, where each inner vector represents
///   the shape of a tensor. This will be modified to reflect the new shapes after contraction.
/// - `orders`: A reference to a vector of vectors containing integers that specify the order of
///   dimensions for each tensor. This is used to identify common dimensions between tensors.
/// - `indices`: A slice of indices indicating which two tensors' shapes are being reformatted.
///   The first index will be updated to include the merged shape, while the second index will be removed.
///
/// # Notes
/// This function assumes that the specified indices correspond to valid tensors in the shapes vector
/// and that the orders vector correctly represents the dimensions of those tensors. It modifies the
/// shapes in place, so the original shapes vector will be updated directly.
fn shape_reformat(shapes: &mut Vec<Vec<i32>>, orders: &[Vec<i32>], indices: &[usize]) {
    // Create a set of common indices shared between the two specified orders
    let common: HashSet<_> = orders[indices[0]]
        .iter()
        .cloned()
        .collect::<HashSet<_>>()
        .intersection(&orders[indices[1]].iter().cloned().collect())
        .cloned()
        .collect();

    // Iterate over the specified indices to remove common dimensions from their shapes
    for &i in indices {
        // Find the indices of the common elements in the current order
        let mut to_remove: Vec<_> = orders[i]
            .iter()
            .enumerate()
            .filter(|(_, x)| common.contains(x))
            .map(|(i, _)| i)
            .collect();

        to_remove.sort();
        // Remove the common dimensions from the shape of the current tensor
        for &idx in to_remove.iter().rev() {
            shapes[i].remove(idx);
        }
    }

    // Merge the shapes of the two tensors being contracted
    let merged: Vec<_> = shapes[indices[0]]
        .iter()
        .chain(&shapes[indices[1]])
        .cloned()
        .collect();
    shapes[indices[0]] = merged;
    shapes.remove(indices[1]);
}

/// Converts the shapes of tensors into a vector of integer vectors.
///
/// This function takes a slice of tensors and extracts their shapes, converting each shape
/// from a dimension type to a vector of integers. This is useful for managing tensor dimensions
/// in a format that is easier to work with during contraction operations.
///
/// # Arguments
/// - `tensors`: A slice of `ArrayD<f64>` representing the tensors whose shapes are to be extracted.
///
/// # Returns
/// - `Vec<Vec<i32>>`: A vector of vectors, where each inner vector contains the dimensions of
///   a corresponding tensor, represented as integers.
///
/// # Note
/// The function assumes that the input tensors are valid and have defined shapes. The resulting
/// vector of shapes will have the same length as the input tensor slice, and each inner vector
/// will correspond to the dimensions of the respective tensor. This format is particularly useful
/// for operations that require knowledge of tensor dimensions, such as contraction and reshaping.
fn shape_vec(tensors: &[ArrayD<f64>]) -> Vec<Vec<i32>> {
    tensors
        .iter()
        .map(|t| t.shape().iter().map(|&d| d as i32).collect())
        .collect()
}

/// Rearranges the axes of the final tensor based on the contraction order.
///
/// This function takes a tensor and its contraction order, sorts the order, and permutes the
/// tensor's axes accordingly. The resulting tensor will have its dimensions arranged in a
/// specified order, which is important for ensuring that the output tensor matches the expected
/// layout after contraction.
///
/// # Arguments
/// - `tensor`: An `ArrayD<f64>` representing the final contracted tensor to be rearranged.
/// - `order`: A vector of vectors representing the contraction order, which indicates how the
///   dimensions should be permuted.
///
/// # Returns
/// - `ArrayD<f64>`: The tensor with its axes permuted according to the specified order.
///
/// # Note
/// The function assumes that the contraction order is valid and corresponds to the dimensions of
/// the input tensor. The output tensor will have its axes rearranged based on the sorted order,
/// which is crucial for maintaining the correct structure of the tensor after contraction. Care
/// should be taken to ensure that the order provided accurately reflects the desired output layout.
fn final_order(tensor: ArrayD<f64>, order: Vec<Vec<i32>>) -> ArrayD<f64> {
    let mut sorted = order[0].clone();
    sorted.sort_by(|a, b| b.cmp(a));

    let axis_order: Vec<_> = sorted
        .iter()
        .map(|x| order[0].iter().position(|y| y == x).unwrap())
        .collect();

    tensor.permuted_axes(IxDyn(&axis_order))
}
