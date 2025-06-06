use qua_ten_net::tencon::contract;
use qua_ten_net::tensor::random;

fn main() {
    let tensor_a = random(&[2, 2, 3, 2]);
    let tensor_b = random(&[3, 2, 4]);
    let tensor_c = random(&[3, 4]);

    match contract(
        &[tensor_a, tensor_b, tensor_c],
        &[&[1, -1, 2, -2], &[2, 1, 3], &[-3, 3]],
    ) {
        Ok(result) => println!("\nContraction result: \n{:?}", result),
        Err(err) => eprintln!("\nError during contraction: \n{}", err),
    }
}
