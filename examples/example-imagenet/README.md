# Triton-rust : An ImageNet inference example

This example show you how to use the library to infer a vision

## Setting up Triton Inference model

The first step is to convert a vision model to ONNX format.
You can use [this python script](generate_onnx_resnet.py) to export the model to ONNX. It requires torchvision.

The associated Triton configuration for this model is given [here](config/config.pbtxt).

## Building the example

You can build the example using the following command :

```
make triton-example-imagenet
```

## Running the example

```bash
target/release/examples/triton-example-imagenet
```