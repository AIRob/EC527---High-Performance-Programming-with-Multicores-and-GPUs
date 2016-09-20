/************************************************************************/
// gcc -O1 -fopenmp -o test_mmm_inter_omp test_mmm_inter_omp.c -lrt

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#define GIG 1000000000
#define CPG 3.61           // Cycles per GHz -- Adjust to your computer

#define BASE  100
#define ITERS 50
#define DELTA 100
#define bsize 25

#define OPTIONS 3
#define IDENT 0

#define OMEGA 1.90       // TO BE DETERMINED

#define MINVAL   0.0
#define MAXVAL  10.0

#define TOL 0.00001

typedef float data_t;

/* Create abstract data type for matrix */
typedef struct {
  long int len;
  data_t *data;
} matrix_rec, *matrix_ptr;
/************************************************************************/
main(int argc, char *argv[])
{
  int OPTION;
  struct timespec diff(struct timespec start, struct timespec end);
  struct timespec time1, time2;
  struct timespec time_stamp[OPTIONS][ITERS+1];
  int clock_gettime(clockid_t clk_id, struct timespec *tp);
  matrix_ptr new_matrix(long int len);
  int set_matrix_length(matrix_ptr m, long int index);
  long int get_matrix_length(matrix_ptr m);
  int init_matrix(matrix_ptr m, long int len);
  int zero_matrix(matrix_ptr m, long int len);
  void mmm_ijk(matrix_ptr a, matrix_ptr b, matrix_ptr c);
  void mmm_ijk_omp(matrix_ptr a, matrix_ptr b, matrix_ptr c);
  void mmm_kij(matrix_ptr a, matrix_ptr b, matrix_ptr c);
  void mmm_kij_omp(matrix_ptr a, matrix_ptr b, matrix_ptr c);
  void mmm_jki(matrix_ptr a, matrix_ptr b, matrix_ptr c);
  void SOR(matrix_ptr a, int *iterations);
  void SOR_omp(matrix_ptr a, int *iterations);
  void bijk( matrix_ptr a, matrix_ptr b, matrix_ptr c);

  int *iterations;

  iterations = (int *)malloc(sizeof(int));

  long int i, j, k;
  long int time_sec, time_ns;
  long int MAXSIZE = BASE+(ITERS+1)*DELTA;

  printf("\n Hello World -- MMM \n");

  // declare and initialize the matrix structure
  matrix_ptr a0 = new_matrix(MAXSIZE);
  init_matrix(a0, MAXSIZE);
  matrix_ptr b0 = new_matrix(MAXSIZE);
  init_matrix(b0, MAXSIZE);
  matrix_ptr c0 = new_matrix(MAXSIZE);
  zero_matrix(c0, MAXSIZE);

  OPTION = 0;
  for (i = 0; i < ITERS; i++) {
    set_matrix_length(a0,BASE+(i+1)*DELTA);
    clock_gettime(CLOCK_REALTIME, &time1);
    SOR(a0, iterations);
    clock_gettime(CLOCK_REALTIME, &time2);
    time_stamp[OPTION][i] = diff(time1,time2);
    printf("\niter = %d\n", i);
  }
 
  OPTION++;
  for (i = 0; i < ITERS; i++) {
    set_matrix_length(a0,BASE+(i+1)*DELTA);
    clock_gettime(CLOCK_REALTIME, &time1);
    SOR_omp(a0, iterations);
    clock_gettime(CLOCK_REALTIME, &time2);
    time_stamp[OPTION][i] = diff(time1,time2);
    printf("\niter = %d", i);
  }
 
  OPTION++;
  for (i = 0; i < ITERS; i++) {
    set_matrix_length(a0,BASE+(i+1)*DELTA);
    set_matrix_length(b0,BASE+(i+1)*DELTA);
    set_matrix_length(c0,BASE+(i+1)*DELTA);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
    bijk(a0,b0,c0);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
    time_stamp[OPTION][i] = diff(time1,time2);
    printf("\niter = %d", i);
  }
  
  printf("\nlength, ijk, kij, jki");
  for (i = 0; i < ITERS; i++) {
    printf("\n%d, ", BASE+(i+1)*DELTA);
    for (j = 0; j < OPTIONS; j++) {
      if (j != 0) printf(", ");
      printf("%ld", (long int)((double)(CPG)*(double)
		 (GIG * time_stamp[j][i].tv_sec + time_stamp[j][i].tv_nsec)));
    }
  }

  printf("\n");
  
}/* end main */

/**********************************************/

/* Create matrix of specified length */
matrix_ptr new_matrix(long int len)
{
  long int i;

  /* Allocate and declare header structure */
  matrix_ptr result = (matrix_ptr) malloc(sizeof(matrix_rec));
  if (!result) return NULL;  /* Couldn't allocate storage */
  result->len = len;

  /* Allocate and declare array */
  if (len > 0) {
    data_t *data = (data_t *) calloc(len*len, sizeof(data_t));
    if (!data) {
	  free((void *) result);
	  printf("\n COULDN'T ALLOCATE STORAGE \n", result->len);
	  return NULL;  /* Couldn't allocate storage */
	}
	result->data = data;
  }
  else result->data = NULL;

  return result;
}

/* Set length of matrix */
int set_matrix_length(matrix_ptr m, long int index)
{
  m->len = index;
  return 1;
}

/* Return length of matrix */
long int get_matrix_length(matrix_ptr m)
{
  return m->len;
}

/* initialize matrix */
int init_matrix(matrix_ptr m, long int len)
{
  long int i;

  if (len > 0) {
    m->len = len;
    for (i = 0; i < len*len; i++)
      m->data[i] = (data_t)(i);
    return 1;
  }
  else return 0;
}

/* initialize matrix */
int zero_matrix(matrix_ptr m, long int len)
{
  long int i,j;

  if (len > 0) {
    m->len = len;
    for (i = 0; i < len*len; i++)
      m->data[i] = (data_t)(IDENT);
    return 1;
  }
  else return 0;
}

data_t *get_matrix_start(matrix_ptr m)
{
  return m->data;
}

/*************************************************/

struct timespec diff(struct timespec start, struct timespec end)
{
  struct timespec temp;
  if ((end.tv_nsec-start.tv_nsec)<0) {
    temp.tv_sec = end.tv_sec-start.tv_sec-1;
    temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
  } else {
    temp.tv_sec = end.tv_sec-start.tv_sec;
    temp.tv_nsec = end.tv_nsec-start.tv_nsec;
  }
  return temp;
}

/*************************************************/

/* SOR */
void SOR_omp(matrix_ptr a, int *iterations)
{

  data_t *data = get_matrix_start(a);
  omp_set_num_threads(4);

  #pragma omp parallel shared(data)
 {
  
  long int i, j;
  long int length = get_matrix_length(a);
  double change; 
  int mean_change = 1000;   
  int iters = 0;
  while ((mean_change/(double)(length*length)) > (double)TOL) {
    iters++;
    mean_change = 0;
    #pragma omp for 
    for (i = 1; i < length-1; i++) 
      for (j = 1; j < length-1; j++) {
	change = data[i*length+j] - .25 * (data[(i-1)*length+j] +
					  data[(i+1)*length+j] +
					  data[i*length+j+1] +
					  data[i*length+j-1]);
	data[i*length+j] -= change * OMEGA;
	if (change < 0){
	  change = -change;
	}
	mean_change += change;
      }
    if (abs(data[(length-2)*(length-2)]) > 10.0*(MAXVAL - MINVAL)) {
      break;
    }
  }
   *iterations = iters;
 }
}


void SOR(matrix_ptr a, int *iterations)
{
  long int i, j;
  long int length = get_matrix_length(a);
  data_t *data = get_matrix_start(a);
  double change, mean_change = 1000;   
  int iters = 0;
    
  while ((mean_change/(double)(length*length)) > (double)TOL) {
    iters++;
    mean_change = 0;
    for (i = 1; i < length-1; i++) 
      for (j = 1; j < length-1; j++) {
	change = data[i*length+j] - .25 * (data[(i-1)*length+j] +
					  data[(i+1)*length+j] +
					  data[i*length+j+1] +
					  data[i*length+j-1]);
	data[i*length+j] -= change * OMEGA;
	if (change < 0){
	  change = -change;
	}
	mean_change += change;
      }
    if (abs(data[(length-2)*(length-2)]) > 10.0*(MAXVAL - MINVAL)) {
      break;
    }
  }
   *iterations = iters;
 
}

void bijk( matrix_ptr a, matrix_ptr b, matrix_ptr c)

{
  	data_t *A = get_matrix_start(a);
  	data_t *B = get_matrix_start(b);
  	data_t *C = get_matrix_start(c);

    omp_set_num_threads(8);
    #pragma omp parallel shared(A,B,C)
	{
     
        long int i, j, k, kk, jj;
        double sum;
        long int get_matrix_length(matrix_ptr m);
        data_t *get_matrix_start(matrix_ptr m);
        long int n = get_matrix_length(a);
        long int en = bsize*(n/bsize);
    #pragma omp for
        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
                C[i*n+ j] = 0.0;
        
        #pragma omp for
        for (kk = 0; kk < en; kk += bsize) {
            for (jj = 0; jj < en; jj += bsize) {
                for (i = 0; i < n; i++) {
                    for (j = jj; j < jj + bsize; j++) {
                        sum = C[i*n+j];
                        for (k = kk; k < kk + bsize; k++) {
                            sum += A[i*n+k]*B[k*n+j];
                        }
                        C[i*n+j] = sum;
                    }
                }
            }
        }
    }

}
