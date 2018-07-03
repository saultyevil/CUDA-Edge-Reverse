#include "main.h"

__global__
void CUDA_edge_reverse(double *cuda_buff, int nx, int ny)
{
    int i,
        j,
        iter;

    /*
     * Find the thread number of a CUDA core and the corresponding index for a
     * 2D array
     */
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int rowID = blockIdx.y * blockDim.y + tx;
    int colID = blockIdx.x * blockDim.x + ty;

    for (i = 0; i < ny + 2; i++)
    {
        for (j = 0; j < nx + 2; j++)
        {
            cuda_buff[i * ny + j] = 34;
        }
    }
}

extern "C" int CUDA_image_processing(double *host_buff, double *cuda_buff,
    int buff_size, int nx, int ny)
{
    /*
     * Allocate memory on the device and copy the normalised host buff into the
     * device memory
     */
    cudaMalloc((void **) &cuda_buff, buff_size);
    cudaMemcpy(cuda_buff, host_buff, buff_size, cudaMemcpyHostToDevice);

    /*
     * Begin image processing using a CUDA kernel
     */
    CUDA_edge_reverse<<<1,1>>>(cuda_buff, nx, ny);

    /*
     * Copy the device result into host memory and free the pointer
     */
    cudaMemcpy(host_buff, cuda_buff, buff_size, cudaMemcpyDeviceToHost);
    cudaFree(cuda_buff);

    return 0;
}

