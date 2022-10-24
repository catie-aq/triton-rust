clean:
	rm -rf *~ dist *.egg-info build target

build:
	cd src/shared_memory && make all && cd ../../
	maturin build --release --manylinux=off

all: build
