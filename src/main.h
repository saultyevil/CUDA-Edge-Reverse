/* ****************************************************************************
 * Image processing header file:
 *  - contains constants and function prototypes for IO and dummy CUDA funcs
 * ************************************************************************** */

#define TRUE 1
#define FALSE 0
#define MAX_LINE 128
#define MASTER_GPU 0
#define NO_PAR_CONST 0
#define STRING_NO_PAR_CONST '\0'

#define FLOAT_PRECISION double

/*
 * IO functions
 */
void pgmsize(char *filename, int *nx, int *ny);
void pgmread(char *filename, void *vx, int nx, int ny);
void pgmwrite(char *filename, void *vx, int nx, int ny);

int get_int_CL(char *par_name, int *parameter);
int get_string_CL(char *par_name, char *parameter);
int read_int(char *ini_file, char *par_name, int *parameter);
int read_string(char *ini_file, char *par_string, char *parameter);

/*
 * Dummy CUDA functions
 */
extern "C" int CUDA_image_processing(FLOAT_PRECISION *image_buff, int nx,
    int ny, int max_iter);
