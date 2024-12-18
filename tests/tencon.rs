use ndarray::*;
use qua_ten_net::tencon::*;

#[test]
fn test_contract() {
    let vec_a: Vec<f64> = (0..12).map(|x| x as f64).collect();
    let a = Array::from_shape_vec(vec![2, 3, 2], vec_a).expect("ShapeError!");

    let vec_b = (0..81).map(|x| x as f64).collect();
    let b = Array::from_shape_vec(vec![3, 3, 3, 3], vec_b).expect("ShapeError!");

    let order: Vec<Vec<i32>> = vec![vec![-1, 1, 2], vec![2, 3, -2], vec![1, 3, -3, -4]];

    let con = contract(vec![a.clone(), a, b], order);

    let correct_vec = vec![
        12852.0, 13104.0, 13356.0, 13608.0, 13860.0, 14112.0, 14364.0, 14616.0, 14868.0, 15120.0,
        15417.0, 15714.0, 16011.0, 16308.0, 16605.0, 16902.0, 17199.0, 17496.0, 33588.0, 34380.0,
        35172.0, 35964.0, 36756.0, 37548.0, 38340.0, 39132.0, 39924.0, 39744.0, 40689.0, 41634.0,
        42579.0, 43524.0, 44469.0, 45414.0, 46359.0, 47304.0,
    ]
    .into_iter()
    .map(|x| x as f64)
    .collect();

    let correct = Array::from_shape_vec(vec![2, 2, 3, 3], correct_vec).expect("ShapeError!");

    assert_eq!(con, correct);
}

#[test]
fn test_contract_map() {
    let vec_a: Vec<f64> = (0..12).map(|x| x as f64).collect();
    let a = Array::from_shape_vec(vec![2, 3, 2], vec_a).expect("ShapeError!");

    let vec_b = (0..81).map(|x| x as f64).collect();
    let b = Array::from_shape_vec(vec![3, 3, 3, 3], vec_b).expect("ShapeError!");

    let order: Vec<Vec<i32>> = vec![vec![-1, 1, 2], vec![2, 3, -2], vec![1, 3, -3, -4]];

    let con_map = contract_map(vec![a.clone(), a, b], order);
    let correct = vec![vec![0, 1], vec![0, 1]];
    assert_eq!(con_map, correct);
}
