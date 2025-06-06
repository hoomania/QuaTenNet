use ndarray::Array2;
use qua_ten_net::tensor::*;

fn main() {
    let tnsr = tensor(&[2, 2, 2], 13.0);
    println!("\nTensor filled with arbitrary number: \n{:?}", tnsr);

    let zeros = zeros(&[2, 2, 2]);
    println!("\nTensor filled with zeros: \n{:?}", zeros);

    let ones = ones(&[2, 2, 2]);
    println!("\nTensor filled with zeros: \n{:?}", ones);

    let identity = identity(3);
    println!("\n3x3 idnetity matrix: \n{:?}", identity);

    let diag = diagonal(&[3.14159, 2.71828, 1.38064]);
    println!("\n3x3 diagonal matrix: \n{:?}", diag);

    let rnd = random(&[2, 2, 2]);
    println!(
        "\nTensor filled with random numbers between 0.0 and 1.0: \n{:?}",
        rnd
    );

    let sample =
        Array2::from_shape_vec((3, 3), (0..9).map(|x| x as f64).collect()).expect("ShapeError!");

    match svd(sample.clone()) {
        Ok(svd) => {
            println!("\n3x3 matrix as SVD input: \n{:?}", sample);
            println!("\nU matrix: \n{:?}", svd.u);
            println!("\nSigma matrix: \n{:?}", svd.sigma);
            println!("\nV^T matrix: \n{:?}", svd.vt);
        }
        Err(err) => {
            println!("\nError on SVD: \n{}", err);
        }
    }
}
