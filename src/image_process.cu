#include <stdio.h>
#include <math.h>

#include "main.h"

__global__
void CUDA_edge_reverse(FLOAT_PRECISION *cuda_buff, FLOAT_PRECISION *cuda_old,
    int nx, int ny, int max_iter)
{
    int iter;
    FLOAT_PRECISION cuda_new;

    /*
     * Find the thread number of a CUDA core and the corresponding index for a
     * 2D array
     */
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int colID = blockDim.x * blockIdx.x + tx;
    int rowID = blockDim.y * blockIdx.y + ty;

    if ((colID > 0 ) && (colID < nx+1) && (rowID > 0) && (rowID < ny+1))
    {
        /*
         * Image processing iterations
         */
        for (iter = 0; iter < max_iter; iter++)
        {
            cuda_new = 0.25 *
            (
                cuda_old[colID * (ny + 2) + (rowID - 1)] +
                cuda_old[colID * (ny + 2) + (rowID + 1)] +
                cuda_old[(colID - 1) * (ny + 2) + rowID] +
                cuda_old[(colID + 1) * (ny + 2) + rowID] -
                cuda_buff[(colID - 1) * ny + (rowID - 1)]
            );

            /*
            * Set the new values to be the old values for the next iteration
            */
            cuda_old[colID * (ny + 2) + rowID] = cuda_new;

            /*
            * Sync the threads to make sure that all the cells have updated
            * correctly for the next iteration
            */
            __syncthreads();
        }

        /*
        * Once interations are complete, copy the "old" values, i.e. the
        * reversed pixel values, into the image buffer
        */
        cuda_buff[(colID - 1) * ny + (rowID - 1)] =
            cuda_old[colID * (ny + 2) + rowID];
    }

}

extern "C" int CUDA_image_processing(FLOAT_PRECISION *image_buff, int nx,
    int ny, int max_iter)
{
    int i, buff_size, image_size;
    float cuda_runtime;  // has to be a float for the CUDA functions :^)
    cudaEvent_t cuda_start, cuda_stop;
    FLOAT_PRECISION *host_old = NULL,
                    *cuda_old = NULL,
                    *cuda_new = NULL,
                    *cuda_buff = NULL;

    cudaEventCreate(&cuda_start);
    cudaEventCreate(&cuda_stop);

    /*
     * Allocate memory on the device and copy the normalised host buff into the
     * device memory
     */
    image_size = nx * ny * sizeof(FLOAT_PRECISION);
    buff_size = (nx + 2) * (ny + 2) * sizeof(FLOAT_PRECISION);

    host_old = (FLOAT_PRECISION *) malloc(buff_size);

    for (i = 0; i < (nx + 2) * (ny + 2); i++)
        host_old[i] = 255.0;

    cudaMalloc((void **) &cuda_old, buff_size);
    cudaMalloc((void **) &cuda_new, buff_size);
    cudaMalloc((void **) &cuda_buff, image_size);

    /*
     * Copy the image to the cuda image buff
     */
    cudaMemcpy(cuda_old, host_old, buff_size, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_buff, image_buff, image_size, cudaMemcpyHostToDevice);

    free(host_old);

    /*
     * Begin image processing using CUDA
     *  - (16, 8) is 128 threads per block
     */
    dim3 n_threads(16, 8);
    dim3 n_blocks((nx + 2)/n_threads.x + 1, (ny + 2)/n_threads.y + 1);

    cudaEventRecord(cuda_start, MASTER_GPU);

    /*
     * Call the CUDA kernel to do the image processing. The number of blocks
     * created exceeds the number of threads required, but guarantees that all
     * pixels get a CUDA code
     */
    CUDA_edge_reverse<<<n_blocks, n_threads>>>(cuda_buff, cuda_old, nx, ny,
        max_iter);

    cudaEventRecord(cuda_stop, MASTER_GPU);
    cudaEventSynchronize(cuda_stop);
    cudaEventElapsedTime(&cuda_runtime, cuda_start, cuda_stop);

    /*
     * Copy the device result into host memory and free the pointer
     */
    cudaMemcpy(image_buff, cuda_buff, image_size, cudaMemcpyDeviceToHost);
    cudaFree(cuda_buff);

    printf("\n---------------------------------------\n");
    printf("\nKernel runtime: %5.3f ms\n", cuda_runtime);
    printf("\n---------------------------------------\n");

    return 0;
}

