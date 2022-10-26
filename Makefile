clean:
	rm -rf *~ dist *.egg-info build target

lib:
	cd src/shared_memory && make all && cd ../../
	cargo build --release

triton-example-huggingface:
	cargo build --release --example triton-example-huggingface

examples: triton-example-huggingface

all: lib triton-example-huggingface
