CC = g++  # C compiler
CFLAGS = -Wall -Wextra -O3 -g -I/opt/cuda/include/
LDFLAGS = -L/opt/cuda/targets/x86_64-linux/lib/ -lcuda -lcudart -lrt
RM = rm -f   # rm command
TARGET_LIB = libcudasharedmemory.so
TARGET_LIB_STATIC = libcudasharedmemory.a

SRCS = cuda_shared_memory/cuda_shared_memory.cc cuda_shared_memory/cencode.cc system_shared_memory/shared_memory.cc  # source files
OBJS = $(SRCS:.cc=.o)

.PHONY: all
all: ${TARGET_LIB} ${TARGET_LIB_STATIC}

$(TARGET_LIB): $(SRCS)
	$(CC) -shared $(CFLAGS) $(LDFLAGS) -o $@ $^

$(TARGET_LIB_STATIC): $(OBJS)
	ar rcs $(TARGET_LIB_STATIC) *.o

$(OBJS): $(SRCS)
	$(CC) -c $(CFLAGS) $(LDFLAGS) $^

.PHONY: clean
clean:
	-${RM} ${OBJS} $(SRCS:.cc=.d)
