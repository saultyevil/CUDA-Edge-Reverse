#include <stdlib.h>
#include <stdio.h>

#include "main.h"

int main(void)
{
    int i,
        j,
        nx,
        ny,
        i_buff_size,
        p_buff_size;
    char *input = (char *) malloc(sizeof(*input) * MAX_LINE),
         *output = (char *) malloc(sizeof(*input) * MAX_LINE);
    double *image_buff = NULL,
           *host_p_buff = NULL,
           *cuda_p_buff = NULL;

    input = "edge768x768.pgm";
    output = "image768x768.pgm";

    /*
     * Read in the image/image dimensions and allocate memory for the host
     * image and process buffers
     *  - i_buff_size is the size required for the image buffer
     *  - p_buff_size is the size required for the process buffer, note that it
     *    is slighty larger as there are an extra ring of cells on the edge of
     *    the buffer
     */
    pgmsize(input, &nx, &ny);
    i_buff_size = nx * ny * sizeof(*image_buff);
    p_buff_size = (nx + 2) * (ny + 2) * sizeof(*host_p_buff);
    image_buff = (double *) malloc(i_buff_size);
    host_p_buff = (double *) malloc(p_buff_size);
    pgmread(input, image_buff, nx, ny);

    /*
     * Normalise the host process buffer as being completely white. This serves
     * as a good first starting point for the iterations as well as acts as an
     * edge source for the edge pixels
     */
    for (i = 0; i < ny + 2; i++)
    {
        for (j = 0; j < nx + 2; j++)
        {
            host_p_buff[i * ny + j] = 255.0;
        }
    }

    /*
     * Call the CUDA dummy function to begin image processing
     */
    CUDA_image_processing(host_p_buff, cuda_p_buff, p_buff_size, nx, ny);

    /*
     * Write out the final image
     */
    pgmwrite(output, host_p_buff, nx, ny);

    return 0;
}
