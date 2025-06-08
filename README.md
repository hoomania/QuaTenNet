<div align="center">
<img src="https://raw.githubusercontent.com/hoomania/QuaTenNet/master/assets/qtn_logo.jpg" width="250px" style="border-radius: 10px;
"/>
</div>
## QuaTenNet (QTN)

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
qua_ten_net = "0.2.0"  # Replace with the latest version
```

## Contraction Graph

Tensor contraction is fundamental to tensor networks. In the graphical representation of tensor contraction, each tensor is depicted as a geometric object (usually a circle), and the indices of the tensor are represented as legs that connect to the tensor. This visualization helps in understanding the relationships and operations between tensors in a network.

In the `tencon` module, free legs are addressed using negative numbers, while each positive number must be repeated twice in the contraction order. This approach ensures clarity in the representation of tensor connections and contractions.


![Contraction Graph](https://raw.githubusercontent.com/hoomania/QuaTenNet/master/assets/contraction_graph.jpg)

The contraction graph above can be represented in Rust code as follows, allowing us to contract our tensors with a simple syntax:
```rust
let tensor_a = tensor::random(&[3, 2, 2]); // Define A with a shape of (3, 2, 2)
let tensor_b = tensor::random(&[2, 2, 4, 5]); // Define B with a shape of (2, 2, 4, 5)
let tensor_c = tencon::contract(
        &[tensor_a, tensor_b],
        &[&[-1, 1, 2], &[1, 2, -2, -3]],
    )
```

## Examples

You can find examples of how to use **QuaTenNet** in the examples directory. \
To run an example, use the following command:

```bash
cargo run --example tencon
cargo run --example tendot
cargo run --example trace
cargo run --example tensor
```

## Contribution
Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

## License
This project is licensed under the GPLv3. See the LICENSE file for more details.
