# CUDA Edge Reverse

Based on my previous repository, "MPI-Edge-Reverse". Takes in an edge image and re-creates the original image. The current implementation isn't perfect, but is a neat demonstration of using CUDA.

## Building

You will need the `nvcc` compiler part of the CUDA toolkit to build the program. You will also obviously need a CUDA capable GPU otherwise the program *should* output just a white tile.

To build the program, type into the terminal the following

```bash
$ nvcc -x cu src/main.c src/pgmio.c src/read_par.c src/image_process.cu -o edge_reverse
```

The switch `-x cu` tells the compiler to treat all the input files as `.cu` files, which should simplify linking and compilation in general. If you are feeling more adventurous, you can attempt to create linked objects using `nvcc` for the GPU code and `gcc` or `clang` (or whatever else) for the CPU code. Then you should be able to compile in the normal way. Note that `nvcc` will output an x64 object and thus you will also need to tell your C compiler to do the same.
