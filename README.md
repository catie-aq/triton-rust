# Triton-rust : A gRPC library for Nvidia Triton Inference Server

Triton-rust is a gRPC library to interact with [Nvidia Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server).

## Building

You can compile the library together with all examples with:

```bash
make all
```

## Examples

You can find several examples of neural network inference using Triton Inference Server and Rust. These examples could be found [here](examples/README.md).

## Known bugs

- CUDA shared memory is not functionnal yet
- Rust's ndarrays are to be in standard layout

## Contact

Boris Albar (b.albar@catie.fr)

## Ackowledgements

 This work has been done in frame of the [Vaniila platform](http://vaniila.ai/).

[<img width="200" src="https://www.vaniila.ai/wp-content/uploads/2020/02/Vaniila_bleu_horizontal.png">](http://vaniila.ai/)