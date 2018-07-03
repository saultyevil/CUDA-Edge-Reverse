# CUDA Edge Reverse

Based on my previous repository, "MPI_image_processing". Currently a WIP, but it should compile and run. The output will be a bunch of 34's.

## Building

You will need the `nvcc` compiler part of the CUDA toolkit to build the program. You will also obvioulsy need a CUDA capable GPU otherwise the program will output a white tile.

To build the program, type into the terminal the following

```
$ nvcc -x cu main.c pgmio.c image_process.cu -o edge_reverse
```

The switch `-x cu` tells the compiler to treat all the input files as `.cu` files, which should simplify linking and compilation in general. If you are feeling more adventurous, you can attempt to create linked objects using `nvcc` for the GPU code and `gcc` or `clang` (or whatever else) for the CPU code. Then you should be able to compile in the normal way. Note that `nvcc` will output an x64 object and thus you will also need to tell your C compiler to do the same.

