use ndarray::Array;
use qua_ten_net::trace::trace;

#[test]
fn test_trace() {
    let vec_a: Vec<f64> = (0..16).map(|x| x as f64).collect();
    let a = Array::from_shape_vec(vec![2, 2, 2, 2], vec_a).expect("ShapeError!");

    let trc = trace(&a, vec![1, 3]);

    match trc {
        Ok(result) => {
            let rslt =
                Array::from_shape_vec(vec![2, 2], vec![5.0, 9.0, 21.0, 25.0]).expect("ShapeError!");
            assert_eq!(result, rslt);
        }
        Err(err) => {
            println!("Error: {}", err);
        }
    }
}

#[test]
fn test_trace_fail_axes_len() {
    let vec_a: Vec<f64> = (0..16).map(|x| x as f64).collect();
    let a = Array::from_shape_vec(vec![2, 2, 2, 2], vec_a).expect("ShapeError!");

    let trc = trace(&a, vec![1, 3, 2]);

    match trc {
        Ok(_) => {}
        Err(err) => {
            assert_eq!(
                "Trace calculation need two axes index. (Axes length is 3!)",
                err
            );
        }
    }
}

#[test]
fn test_trace_fail_axes_dim() {
    let vec_a: Vec<f64> = (0..24).map(|x| x as f64).collect();
    let a = Array::from_shape_vec(vec![2, 3, 2, 2], vec_a).expect("ShapeError!");

    let trc = trace(&a, vec![1, 3]);

    match trc {
        Ok(_) => {}
        Err(err) => {
            assert_eq!(
                "Shape mismatch along specified axes: tenosr[1] = 3, tensor[3] = 2",
                err
            );
        }
    }
}
