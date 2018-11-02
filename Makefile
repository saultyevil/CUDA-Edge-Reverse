PROJECT_NAME = cuReverse
BIN_DIR = ./bin
SRC_DIR = ./src
OBJ_DIR = ./$(SRC_DIR)/obj

# Set the C and CUDA compilers, assumed to be in the system PATH
CC = gcc
NVCC = nvcc
CFLAGS =
NVCCFLAGS =
LFLAGS = -lm


all: build clean

build: gobj cobj
	mkdir -p $(BIN_DIR)
	$(NVCC) $(LFLAGS) -o $(BIN_DIR)/$(PROJECT_NAME) *.o

gobj: 
	$(NVCC) $(NVCCFLAGS) -c $(SRC_DIR)/*.cu

cobj:
	$(CC) $(CFLAGS) -c $(SRC_DIR)/*.c

clean:
	rm *.o
