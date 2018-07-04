/*
 * Borrowed from David Henty at the EPCC :^)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "main.h"

/*
 *  Routine to get the size of a PGM data file
 *
 *  Note that this assumes a single line comment and no other white space.
 */
void pgmsize(char *filename, int *nx, int *ny)
{
  FILE *fp;

  char *cret;
  int iret;

  char dummy[MAX_LINE];
  int n = MAX_LINE;

  if (NULL == (fp = fopen(filename,"r")))
  {
    fprintf(stderr, "pgmsize: cannot open <%s>\n", filename);
    exit(-1);
  }

  cret = fgets(dummy, n, fp);
  cret = fgets(dummy, n, fp);

  iret = fscanf(fp,"%d %d", nx, ny);

  fclose(fp);
}

/*
 *  Routine to read a PGM data file into a 2D floating point array
 *  x[nx][ny]. Because of the way C handles (or fails to handle!)
 *  multi-dimensional arrays we have to cast the pointer to void.
 *
 *  Note that this assumes a single line comment and no other white space.
 */
void pgmread(char *filename, void *vx, int nx, int ny)
{
  FILE *fp;

  int nxt, nyt, i, j, t;
  char dummy[MAX_LINE];
  int n = MAX_LINE;

  char *cret;
  int iret;

  FLOAT_PRECISION *x = (FLOAT_PRECISION *) vx;

  if (NULL == (fp = fopen(filename,"r")))
  {
    fprintf(stderr, "pgmread: cannot open <%s>\n", filename);
    exit(-1);
  }

  cret = fgets(dummy, n, fp);
  cret = fgets(dummy, n, fp);

  iret = fscanf(fp,"%d %d",&nxt,&nyt);

  if (nx != nxt || ny != nyt)
  {
    fprintf(stderr,
            "pgmread: size mismatch, (nx,ny) = (%d,%d) expected (%d,%d)\n",
            nxt, nyt, nx, ny);
    exit(-1);
  }

  iret = fscanf(fp,"%d",&i);

  /*
   *  Must cope with the fact that the storage order of the data file
   *  is not the same as the storage of a C array, hence the pointer
   *  arithmetic to access x[i][j].
   */
  for (j=0; j<ny; j++)
  {
    for (i=0; i<nx; i++)
    {
      iret = fscanf(fp,"%d", &t);
      x[(ny-j-1)+ny*i] = t;
    }
  }

  fclose(fp);
}


/*
 *  Routine to write a PGM image file from a 2D floating point array
 *  x[nx][ny]. Because of the way C handles (or fails to handle!)
 *  multi-dimensional arrays we have to cast the pointer to void.
 */
void pgmwrite(char *filename, void *vx, int nx, int ny)
{
  FILE *fp;

  int i, j, k, grey;

  FLOAT_PRECISION xmin, xmax, tmp, fval;
  FLOAT_PRECISION thresh = 255.0;

  FLOAT_PRECISION *x = (FLOAT_PRECISION *) vx;

  if (NULL == (fp = fopen(filename,"w")))
  {
    fprintf(stderr, "pgmwrite: cannot create <%s>\n", filename);
    exit(-1);
  }

  printf("\nWriting %d x %d picture into file: %s\n", nx, ny, filename);
  printf("\n---------------------------------------\n\n");

  /*
   *  Find the max and min absolute values of the array
   */
  xmin = fabs(x[0]);
  xmax = fabs(x[0]);

  for (i=0; i < nx*ny; i++)
  {
    if (fabs(x[i]) < xmin) xmin = fabs(x[i]);
    if (fabs(x[i]) > xmax) xmax = fabs(x[i]);
  }

  if (xmin == xmax) xmin = xmax-1.0;

  fprintf(fp, "P2\n");
  fprintf(fp, "# Written by pgmio::pgmwrite\n");
  fprintf(fp, "%d %d\n", nx, ny);
  fprintf(fp, "%d\n", (int) thresh);

  k = 0;

  for (j=ny-1; j >=0 ; j--)
  {
    for (i=0; i < nx; i++)
    {
      /*
       *  Access the value of x[i][j]
       */
      tmp = x[j+ny*i];

      /*
       *  Scale the value appropriately so it lies between 0 and thresh
       */
      fval = thresh*((fabs(tmp)-xmin)/(xmax-xmin))+0.5;
      grey = (int) fval;

      fprintf(fp, "%3d ", grey);

      if (0 == (k+1)%18) fprintf(fp, "\n");

      k++;
    }
  }

  if (0 != k%18) fprintf(fp, "\n");
  fclose(fp);
}
