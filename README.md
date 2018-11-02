# CUDA Edge Reverse

Based on my previous repository, "MPI-Edge-Reverse". Takes in an edge image and re-creates the original image. The current implementation isn't perfect, but is a neat demonstration of using CUDA.

## Building

The program can be built using the provided Makefile.

You will need the `nvcc` compiler part of the CUDA toolkit to build the program. You will also obviously need a CUDA capable GPU otherwise the program *should* output just a white tile.
