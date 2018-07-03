/* ****************************************************************************
 * Image processing header file:
 *  - contains constants and function prototypes for IO and dummy CUDA funcs
 * ************************************************************************** */

#define MAX_LINE 128
#define MASTER_GPU 0

/*
 * IO functions
 */
void pgmsize(char *filename, int *nx, int *ny);
void pgmread(char *filename, void *vx, int nx, int ny);
void pgmwrite(char *filename, void *vx, int nx, int ny);

/*
 * Dummy CUDA functions
 */
extern "C" int CUDA_image_processing(double *host_buff, double *cuda_buff,
    int buff_size, int nx, int ny);
