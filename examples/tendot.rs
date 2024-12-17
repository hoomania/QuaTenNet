use ndarray::Array;
use qua_ten_net::tendot::tensor_dot;

fn main() {
    let vec_a: Vec<f64> = (0..6).map(|x| x as f64).collect();
    let a = Array::from_shape_vec(vec![2, 3], vec_a).expect("ShapeError!");

    let vec_b = (0..12).map(|x| x as f64).collect();
    let b = Array::from_shape_vec(vec![3, 2, 2], vec_b).expect("ShapeError!");

    let dot = tensor_dot(&a, &b, vec![1, 0]);

    match dot {
        Ok(result) => {
            println!("Dot Product: {:?}", result);
        }
        Err(err) => {
            println!("Error: {}", err);
        }
    }
}
