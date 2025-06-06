use ndarray::Array;
use qua_ten_net::trace::trace;

fn main() {
    let vec_a: Vec<f64> = (0..16).map(|x| x as f64).collect();
    let a = Array::from_shape_vec(vec![2, 2, 2, 2], vec_a).expect("ShapeError!");

    let trc = trace(&a, vec![1, 3]);

    match trc {
        Ok(result) => {
            println!("\nInput tensor: \n{:?}", a);
            println!("\nTrace on second and fourth indexes: \n{:?}", result);
        }
        Err(err) => {
            println!("Error: {}", err);
        }
    }
}
