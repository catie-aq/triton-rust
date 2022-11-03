clean:
	rm -rf *~ dist *.egg-info build target

lib:
	cd src/shared_memory && make all && cd ../../
	cargo build --release

triton-example-huggingface:
	cargo build --release --example triton-example-huggingface

triton-example-imagenet:
	cargo build --release --example triton-example-imagenet

examples: triton-example-huggingface triton-example-imagenet

all: lib examples
