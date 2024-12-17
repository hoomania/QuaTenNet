use ndarray::Array;
use qua_ten_net::tencon::{contract, contract_map};

fn main() {
    let vec_a: Vec<f64> = (0..12).map(|x| x as f64).collect();
    let a = Array::from_shape_vec(vec![2, 3, 2], vec_a).expect("ShapeError!");

    let vec_b: Vec<f64> = (0..4).map(|x| x as f64).collect();
    let b = Array::from_shape_vec(vec![2, 2], vec_b).expect("ShapeError!");

    let vec_b = (0..81).map(|x| x as f64).collect();
    let c = Array::from_shape_vec(vec![3, 3, 3, 3], vec_b).expect("ShapeError!");

    let order: Vec<Vec<i32>> = vec![vec![1, 2, -1], vec![1, -2], vec![2, -3, -4, -5]];

    let con = contract(vec![a.clone(), b.clone(), c.clone()], order.clone());
    let con_map = contract_map(vec![a, b, c], order);
    println!("\n Contraction Output: {:?}", con);
    println!("\n Contraction Map: {:?}", con_map);
}
