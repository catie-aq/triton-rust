CC = g++  # C compiler
CFLAGS = -Wall -Wextra -O3 -g -I/opt/cuda/include/
LDFLAGS = -L/opt/cuda/targets/x86_64-linux/lib/ -lcuda -lcudart -lrt -shared
RM = rm -f   # rm command
TARGET_LIB = libcudasharedmemory.so

SRCS = cuda_shared_memory/cuda_shared_memory.cc cuda_shared_memory/cencode.cc system_shared_memory/shared_memory.cc  # source files
OBJS = $(SRCS:.cc=.o)

.PHONY: all
all: ${TARGET_LIB}

$(TARGET_LIB): $(SRCS)
	$(CC) -shared $(CFLAGS) $(LDFLAGS) -o $@ $^

.PHONY: clean
clean:
	-${RM} ${OBJS} $(SRCS:.cc=.d)
