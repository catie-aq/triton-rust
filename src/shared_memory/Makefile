CC = g++  # C compiler
CFLAGS = -fPIC -Wall -Wextra -O3 -g -I/opt/cuda/include/ -DTRITON_ENABLE_GPU
LDFLAGS = -lcuda -lcudart -lrt -shared
RM = rm -f   # rm command
TARGET_LIB = libcudasharedmemory.so

SRCS = cuda_shared_memory/cuda_shared_memory.cc cuda_shared_memory/cencode.cc system_shared_memory/shared_memory.cc  # source files
OBJS = $(SRCS:.cc=.o)

.PHONY: all
all: ${TARGET_LIB}

$(TARGET_LIB): $(OBJS)
	$(CC) -shared -fPIC $(LDFLAGS) -o $@ $^

$(OBJS): $(SRCS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

$(SRCS:.cc=.d):%.d:%.cc
	$(CC) $(CFLAGS) -MM $< >$@

include $(SRCS:.cc=.d)

.PHONY: clean
clean:
	-${RM} ${OBJS} $(SRCS:.cc=.d)
