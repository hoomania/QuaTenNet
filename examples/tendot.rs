use qua_ten_net::tendot::tensor_dot;
use qua_ten_net::tensor::random;

fn main() {
    let tensor_a = random(&[3, 2]);
    let tensor_b = random(&[2, 4, 2]);

    match tensor_dot(&tensor_a, &tensor_b, vec![1, 0]) {
        Ok(result) => {
            println!("\nMatrix A: \n{:?}", tensor_a);
            println!("\nMatrix B: \n{:?}", tensor_b);
            println!(
                "\nDot product of A and B along the second index of A and the first index of B: \n{:?}",
                result
            );
        }
        Err(err) => {
            println!("\nError on tensor dot product: \n{}", err);
        }
    }
}
