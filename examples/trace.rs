use qua_ten_net::tensor::random;
use qua_ten_net::trace::trace;

fn main() {
    let tensor = random(&[2, 3, 2, 3]);

    match trace(&tensor, vec![1, 3]) {
        Ok(result) => {
            println!("\nInput tensor: \n{:?}", tensor);
            println!("\nTrace on second and fourth indexes: \n{:?}", result);
        }
        Err(err) => {
            println!("Error: {}", err);
        }
    }
}
