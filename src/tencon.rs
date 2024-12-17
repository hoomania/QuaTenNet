use crate::tendot::*;
use ndarray::*;
use std::collections::HashSet;

pub fn contract(tensors: Vec<ArrayD<f64>>, contraction_order: Vec<Vec<i32>>) -> ArrayD<f64> {
    let mut ten_list = tensors.clone();
    let mut cnt_order = contraction_order.clone();

    let map = contract_map(tensors, contraction_order);
    for pair in map {
        let cnt_axis = order_to_index(&cnt_order, &pair);

        let contraction = tensor_dot(&ten_list[pair[0]], &ten_list[pair[1]], cnt_axis);
        match contraction {
            Ok(result) => {
                ten_list[pair[0]] = result;
                ten_list.remove(pair[1]);
                order_reformat(&mut cnt_order, &pair);
            }
            Err(err) => {
                println!("Error: {}", err);
            }
        }
    }

    ten_list[0].clone()
}

pub fn contract_map(
    tensors: Vec<ArrayD<f64>>,
    contraction_order: Vec<Vec<i32>>,
) -> Vec<Vec<usize>> {
    let mut ten_shape = shape_vec(&tensors);
    let mut cnt_order = contraction_order.clone();

    let mut map = vec![];
    while cnt_order.len() > 1 {
        let (rt_mat, rw_sum) = ratio_matrix(&ten_shape, &cnt_order);
        let nodes = nodes_selection(rt_mat, rw_sum);
        map.push(nodes.clone());
        shape_reformat(&mut ten_shape, &cnt_order, &nodes);
        order_reformat(&mut cnt_order, &nodes);
    }

    map
}

fn ratio_matrix(
    shapes: &Vec<Vec<i32>>,
    contraction_order: &Vec<Vec<i32>>,
) -> (Array2<f64>, Vec<f64>) {
    let n = contraction_order.len();
    let mut rt_mat: Array2<f64> = Array2::zeros((n, n));

    for i in 0..n {
        for j in (i + 1)..n {
            let t_i_shape = &shapes[i];
            let t_j_shape = &shapes[j];

            let s_i: i32 = t_i_shape.iter().product();
            let s_j: i32 = t_j_shape.iter().product();

            let same_shape: Vec<_> = contraction_order[i]
                .iter()
                .filter(|&k| contraction_order[j].contains(&k))
                .collect();

            let mut same_val: f64 = 0.0;
            for &ss in &same_shape {
                if let Some(same_ndx) = contraction_order[i].iter().position(|&r| r == *ss) {
                    same_val += t_i_shape[same_ndx] as f64;
                }
            }

            rt_mat[[i, j]] = same_val / (s_i as f64 * s_j as f64);
            rt_mat[[j, i]] = rt_mat[[i, j]];
        }
    }

    let max_item = rt_mat.iter().cloned().fold(f64::MIN, f64::max);

    let mut has_max = vec![0.0; n];
    for i in 0..n {
        let i_row = rt_mat.index_axis(Axis(0), i).to_vec();
        if i_row.contains(&max_item) {
            has_max[i] = i_row.iter().sum();
        }
    }

    (rt_mat, has_max)
}

fn order_reformat(order_list: &mut Vec<Vec<i32>>, merge_order: &Vec<usize>) {
    let base_node = merge_order[0];
    let second_node = merge_order[1];

    let both: Vec<i32> = order_list[base_node]
        .iter()
        .chain(order_list[second_node].iter())
        .cloned()
        .collect();

    let mut merge = Vec::new();
    let mut seen = HashSet::new();

    let mut target = Vec::new();
    for i in both {
        if seen.insert(i) {
            merge.push(i);
        } else {
            target.push(i)
        }
    }

    let set: HashSet<_> = target.iter().cloned().collect();
    merge.retain(|x| !set.contains(x));

    order_list[base_node] = merge.clone();
    order_list.remove(second_node);
}

fn shape_reformat(
    shape_list: &mut Vec<Vec<i32>>,
    order_list: &Vec<Vec<i32>>,
    indexes: &Vec<usize>,
) {
    let set1: HashSet<_> = order_list[indexes[0]].iter().cloned().collect();
    let set2: HashSet<_> = order_list[indexes[1]].iter().cloned().collect();

    let common_elements: HashSet<_> = set1.intersection(&set2).collect();

    for i in &common_elements {
        for l in 0..2 {
            for (j, value) in order_list[indexes[l]].iter().enumerate() {
                if i == &value {
                    shape_list[indexes[l]].remove(j);
                }
            }
        }
    }

    shape_list[indexes[0]] = shape_list[indexes[0]]
        .iter()
        .chain(shape_list[indexes[1]].iter())
        .cloned()
        .collect();

    shape_list.remove(indexes[1]);
}

fn nodes_selection(ratio_matrix: Array2<f64>, row_sum: Vec<f64>) -> Vec<usize> {
    let (i_idx, _) =
        row_sum
            .iter()
            .enumerate()
            .fold((0, f64::NEG_INFINITY), |(max_idx, max_val), (i, &val)| {
                if val > max_val {
                    (i, val)
                } else {
                    (max_idx, max_val)
                }
            });

    let row = ratio_matrix.index_axis(Axis(0), i_idx).to_vec();

    let (j_idx, _) =
        row.iter()
            .enumerate()
            .fold((0, f64::NEG_INFINITY), |(max_idx, max_val), (i, &val)| {
                if val > max_val {
                    (i, val)
                } else {
                    (max_idx, max_val)
                }
            });

    vec![i_idx, j_idx]
}

fn order_to_index(order_list: &Vec<Vec<i32>>, indexes: &Vec<usize>) -> Vec<usize> {
    let set1: HashSet<_> = order_list[indexes[0]].iter().cloned().collect();
    let set2: HashSet<_> = order_list[indexes[1]].iter().cloned().collect();

    let common_elements: HashSet<_> = set1.intersection(&set2).collect();

    let mut idx = vec![];

    for i in indexes {
        for (j, &value) in order_list[*i].iter().enumerate() {
            if common_elements.contains(&value) {
                idx.push(j);
            }
        }
    }

    idx
}

fn shape_vec(tensors_list: &Vec<ArrayD<f64>>) -> Vec<Vec<i32>> {
    tensors_list
        .iter()
        .map(|tensor| tensor.shape().iter().map(|&dim| dim as i32).collect())
        .collect()
}
