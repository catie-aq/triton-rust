# Triton-rust : A Huggingface inference example

This example show you how to use the library to infer a language model using Huggingface's library [transformers](https://github.com/huggingface/transformers)

## Setting up Triton Inference model

The first step is to install the huggingface python library and convert a model to onnx.

```bash
python -m transformers.onnx -model=distilbert-base-uncased onnx/
```
Here we use distilbert-base-uncased as an example.

An example of config file for this model is given [here](config/config.pbtxt)

## Building the example

Please refer to library building.

## Running the example

```bash
examples/triton
```