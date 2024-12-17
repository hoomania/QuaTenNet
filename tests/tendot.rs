use ndarray::*;
use qua_ten_net::tendot::tensor_dot;

#[test]
fn test_tensor_dot() {
    let vec_a: Vec<f64> = (0..6).map(|x| x as f64).collect();
    let a = Array::from_shape_vec(vec![2, 3], vec_a).expect("ShapeError!");

    let vec_b = (0..12).map(|x| x as f64).collect();
    let b = Array::from_shape_vec(vec![3, 2, 2], vec_b).expect("ShapeError!");

    let dot = tensor_dot(&a, &b, vec![1, 0]);

    match dot {
        Ok(result) => {
            let rslt = Array::from_shape_vec(
                vec![2, 2, 2],
                vec![20.0, 23.0, 26.0, 29.0, 56.0, 68.0, 80.0, 92.0],
            )
            .expect("ShapeError!");
            assert_eq!(result, rslt);
        }
        Err(err) => {
            println!("{}", err);
        }
    }
}

#[test]
fn test_tensor_dot_fail_axis_length() {
    let vec_a: Vec<f64> = (0..6).map(|x| x as f64).collect();
    let a = Array::from_shape_vec(vec![2, 3], vec_a).expect("ShapeError!");

    let vec_b = (0..12).map(|x| x as f64).collect();
    let b = Array::from_shape_vec(vec![3, 2, 2], vec_b).expect("ShapeError!");

    let dot = tensor_dot(&a, &b, vec![1, 0, 3]);

    match dot {
        Ok(_) => {}
        Err(err) => {
            assert_eq!("Axis length is not even number!", err);
        }
    }
}

#[test]
fn test_tensor_dot_fail_index() {
    let vec_a: Vec<f64> = (0..6).map(|x| x as f64).collect();
    let a = Array::from_shape_vec(vec![2, 3], vec_a).expect("ShapeError!");

    let vec_b = (0..12).map(|x| x as f64).collect();
    let b = Array::from_shape_vec(vec![3, 2, 2], vec_b).expect("ShapeError!");

    let dot = tensor_dot(&a, &b, vec![1, 1]);

    match dot {
        Ok(_) => {}
        Err(err) => {
            assert_eq!(
                "Shape mismatch along specified axes: a[1] = 3, b[1] = 2",
                err
            );
        }
    }
}
