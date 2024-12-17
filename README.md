# QuaTenNet

**QuaTenNet** is a Rust package providing essential tools for working with tensor networks in computational quantum physics. This library aims to facilitate the development and analysis of tensor network algorithms, making it easier for researchers and developers to implement computational quantum simulations.

## Features

- General tools for constructing and manipulating tensor networks.
- Utilizes `ndarray` for efficient manipulation of high-dimensional tensors.
- Use Greedy algorithm for tensor contraction.
- User-friendly API for seamless integration into quantum physics projects.

## Installation

To include **QuaTenNet** in your Rust project, add the following line to your `Cargo.toml` file:

```toml
[dependencies]
qua_ten_net = "0.1.0"  # Replace with the latest version
```

## Examples

You can find examples of how to use **QuaTenNet** in the examples directory. \
To run an example, use the following command:

```bash
cargo run --example tencon
cargo run --example tendot
```

## Sample code for `Tensor Dot Product` and `Tensor Contraction`:

```rust
// Tensro Dot Product:
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

// Tensor Contraction:
let vec_a: Vec<f64> = (0..12).map(|x| x as f64).collect();
let a = Array::from_shape_vec(vec![2, 3, 2], vec_a).expect("ShapeError!");

let vec_b: Vec<f64> = (0..4).map(|x| x as f64).collect();
let b = Array::from_shape_vec(vec![2, 2], vec_b).expect("ShapeError!");

let vec_b = (0..81).map(|x| x as f64).collect();
let c = Array::from_shape_vec(vec![3, 3, 3, 3], vec_b).expect("ShapeError!");

let order: Vec<Vec<i32>> = vec![vec![1, 2, -1], vec![1, -2], vec![2, -3, -4, -5]];

let con = contract(vec![a.clone(), b.clone(), c.clone()], order.clone());
let con_map = contract_map(vec![a, b, c], order);
println!("\n Contraction Output: {:?}", con);
println!("\n Contraction Map: {:?}", con_map);
```

## License
This project is licensed under the GPLv3.
