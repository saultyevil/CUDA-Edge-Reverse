#include <stdlib.h>
#include <stdio.h>

#include "main.h"

void cuda_wrapper(FLOAT_PRECISION *image_buff, int nx, int ny, int max_iter);

int main(int argc, char *argv[])
{
    int nx, ny, i_size, max_iter;
    char *input_name = (char *) malloc(sizeof(char) * MAX_LINE),
         *output_name = (char *) malloc(sizeof(char) * MAX_LINE),
         *ini_file = (char *) malloc(sizeof(char) * MAX_LINE);
    FLOAT_PRECISION *image_buff = NULL;

    printf("\n---------------------------------------\n\n");
    printf("      Beginning image conversion!\n");
    printf("            Yipee!!!!!\n");
    printf("\n---------------------------------------\n\n");

    /*
     * Read in the ini file filename from file
     */
    if (argc < 2)
    {
        printf("No ini file provided, using default.\n");
        ini_file = "gpu_edge.ini";
    }
    else
    {
        ini_file = argv[1];
    }

    /*
     * Read in program paramaters from the ini file
     */
    read_int(ini_file, "MAX_ITER", &max_iter);
    read_string(ini_file, "INPUT_FILE", input_name);
    read_string(ini_file, "OUTPUT_FILE", output_name);

    printf("\n---------------------------------------\n\n");
    printf("MAX_ITER    : %d\n", max_iter);
    printf("INTPUT_NAME : %s\n", input_name);
    printf("OUTPUT_NAME : %s\n", output_name);

    /*
     * Read in the image/image dimensions and allocate memory for the host
     * image buffer and read in the image
     */
    pgmsize(input_name, &nx, &ny);
    i_size = nx * ny * sizeof(FLOAT_PRECISION);
    image_buff = (FLOAT_PRECISION *) malloc(i_size);
    pgmread(input_name, image_buff, nx, ny);

    printf("\nImage resolution: %d x %d\n", nx, ny);
    printf("\n---------------------------------------\n");

    /*
     * Call the CUDA dummy function to begin image processing
     */
    cuda_wrapper(image_buff, nx, ny, max_iter);

    /*
     * Write out the final image
     */
    pgmwrite(output_name, image_buff, nx, ny);

    return 0;
}
