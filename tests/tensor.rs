use ndarray::{Array, Array1, Array2, IxDyn};
use qua_ten_net::tensor::*;

#[test]
fn test_tensor() {
    let tst = tensor(&[2, 2, 2], 13.0);
    let rslt = Array::from_shape_vec(IxDyn(&[2, 2, 2]), vec![13.0; 8]).expect("ShapeError!");
    assert_eq!(tst, rslt);
}

#[test]
fn test_zeros() {
    let tst = zeros(&[2, 2, 2]);
    let rslt = Array::from_shape_vec(IxDyn(&[2, 2, 2]), vec![0.0; 8]).expect("ShapeError!");
    assert_eq!(tst, rslt);
}

#[test]
fn test_ones() {
    let tst = ones(&[2, 2, 2]);
    let rslt = Array::from_shape_vec(IxDyn(&[2, 2, 2]), vec![1.0; 8]).expect("ShapeError!");
    assert_eq!(tst, rslt);
}

#[test]
fn test_identity() {
    let tst = identity(2);

    let rslt = Array::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).expect("ShapeError!");
    assert_eq!(tst, rslt);
}

#[test]
fn test_diagonal() {
    let tst = diagonal(&[3.14159, 2.71828, 1.38064]);

    let rslt = Array::from_shape_vec(
        (3, 3),
        vec![3.14159, 0.0, 0.0, 0.0, 2.71828, 0.0, 0.0, 0.0, 1.38064],
    )
    .expect("ShapeError!");
    assert_eq!(tst, rslt);
}

#[test]
fn test_svd() {
    let tnsr =
        Array2::from_shape_vec((2, 2), (0..4).map(|x| x as f64).collect()).expect("ShapeError!");
    let svd = svd(tnsr).unwrap();

    let u = Array2::from_shape_vec(
        (2, 2),
        vec![
            -0.22975292054736118,
            0.9732489894677303,
            -0.9732489894677303,
            -0.22975292054736118,
        ],
    )
    .expect("ShapeError!");

    let sigma_f64: Box<[f64]> = [3.702459173643832, 0.540181513475453].into();

    let sigma =
        Array1::from_shape_vec(2, vec![3.702459173643832, 0.540181513475453]).expect("ShapeError!");

    let vt = Array2::from_shape_vec(
        (2, 2),
        vec![
            -0.5257311121191336,
            -0.8506508083520399,
            -0.8506508083520399,
            0.5257311121191336,
        ],
    )
    .expect("ShapeError!");

    assert_eq!(svd.u, u);
    assert_eq!(svd.sigma_f64, sigma_f64);
    assert_eq!(svd.sigma, sigma);
    assert_eq!(svd.vt, vt);
}
