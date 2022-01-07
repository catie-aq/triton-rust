clean:
	rm -rf *~ dist *.egg-info build target

build:
	maturin build --release --manylinux=off

all: build
