/*****************************************************************************/

// gcc -o test_eval test_eval.c -lrt

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>

#define SIZE 10000000
#define ITERS 20
#define DELTA 10
#define BASE 0

#define GIG 1000000000
#define CPG 2.9           // Cycles per GHz -- Adjust to your computer

#define UNROLL 2

#define OPTIONS 4
#define IDENT 1.0
#define OP *

#define VALUE_X 1

typedef float data_t;

/* Create abstract data type for vector */
typedef struct {
  long int len;
  data_t *data;
} vec_rec, *vec_ptr;

//struct timespec {
//  time_t tv_sec; /* seconds */
//  long tv_nsec;  /* nanoseconds */
//};

/*****************************************************************************/
main(int argc, char *argv[])
{
  int OPTION;
  struct timespec diff(struct timespec start, struct timespec end);
  struct timespec time1, time2;
  struct timespec time_stamp[OPTIONS][ITERS+1];
  int clock_gettime(clockid_t clk_id, struct timespec *tp);
  vec_ptr new_vec(long int len);
  int get_vec_element(vec_ptr v, long int index, data_t *dest);
  long int get_vec_length(vec_ptr v);
  int set_vec_length(vec_ptr v, long int index);
  int init_vector(vec_ptr v, long int len);
  data_t *x_value, *result;
   
  data_t *get_vec_start(vec_ptr v);
  
  //Function declaration for poly
  void poly(vec_ptr v, data_t *x, data_t *result);
  void poly_unrolling(vec_ptr v, data_t *x, data_t *result);
  void poly_accumulator(vec_ptr v, data_t *x, data_t *result);
  void poly_associative(vec_ptr v, data_t *x, data_t *result);
  

  long int i, j, k;
  long long int time_sec, time_ns;
  long int MAXSIZE = BASE+(ITERS+1)*DELTA;

  printf("\n Hello World -- psum examples\n");

  // declare and initialize the vector structure
  vec_ptr v0 = new_vec(MAXSIZE);
  x_value = (data_t *) malloc(sizeof(data_t));
  result = (data_t *) malloc(sizeof(data_t));
  init_vector(v0, MAXSIZE);

//  vec_ptr check = new_vec(8);
//  data_t *temp = get_vec_start(check);
//  long int check_length = get_vec_length(check);
//  for(i=0; i<check_length; i++)
//  {
//    temp[i] = i+1;
//    printf("\n%ld\n", temp[i]);
//  } 
 

  *x_value = (data_t)VALUE_X;
  //poly_accumulator(check, x_value, result);

//  printf("Result:%ld\n", *result); 
  // execute and time all 7 options from B&O 
  OPTION = 0;
  for (i = 0; i < ITERS; i++) {
    set_vec_length(v0,BASE+(i+1)*DELTA);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
    poly(v0, x_value, result);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
    time_stamp[OPTION][i] = diff(time1,time2);
  }

  OPTION++;
  for (i = 0; i < ITERS; i++) {
    set_vec_length(v0,BASE+(i+1)*DELTA);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
    poly_unrolling(v0, x_value, result);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
    time_stamp[OPTION][i] = diff(time1,time2);
  }

  OPTION++;
  for (i = 0; i < ITERS; i++) {
    set_vec_length(v0,BASE+(i+1)*DELTA);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
    poly_accumulator(v0, x_value, result);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
    time_stamp[OPTION][i] = diff(time1,time2);
  }
  
  OPTION++;
  for (i = 0; i < ITERS; i++) {
    set_vec_length(v0,BASE+(i+1)*DELTA);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
    poly_associative(v0, x_value, result);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
    time_stamp[OPTION][i] = diff(time1,time2);
  }

  /* output times */
  for (i = 0; i < ITERS; i++) {
    printf("\n%d,  ", BASE+(i+1)*DELTA);
    for (j = 0; j < OPTIONS; j++) {
      if (j != 0) printf(", ");
      printf("%ld", (long int)((double)(CPG)*(double)(GIG * time_stamp[j][i].tv_sec + time_stamp[j][i].tv_nsec)));
    }
  }


  printf("\n");
  
}/* end main */

/**********************************************/
/* Create vector of specified length */
vec_ptr new_vec(long int len)
{
  long int i;

  /* Allocate and declare header structure */
  vec_ptr result = (vec_ptr) malloc(sizeof(vec_rec));
  if (!result) return NULL;  /* Couldn't allocate storage */
  result->len = len;

  /* Allocate and declare array */
  if (len > 0) {
    data_t *data = (data_t *) calloc(len, sizeof(data_t));
    if (!data) {
	  free((void *) result);
	  return NULL;  /* Couldn't allocate storage */
	}
	result->data = data;
  }
  else result->data = NULL;

  return result;
}

/* Retrieve vector element and store at dest.
   Return 0 (out of bounds) or 1 (successful)
*/
int get_vec_element(vec_ptr v, long int index, data_t *dest)
{
  if (index < 0 || index >= v->len) return 0;
  *dest = v->data[index];
  return 1;
}

/* Return length of vector */
long int get_vec_length(vec_ptr v)
{
  return v->len;
}

/* Set length of vector */
int set_vec_length(vec_ptr v, long int index)
{
  v->len = index;
  return 1;
}

/* initialize vector */
int init_vector(vec_ptr v, long int len)
{
  long int i;

  if (len > 0) {
    v->len = len;
    for (i = 0; i < len; i++) v->data[i] = (data_t)(i);
    return 1;
  }
  else return 0;
}

data_t *get_vec_start(vec_ptr v)
{
  return v->data;
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
void poly(vec_ptr v, data_t *x, data_t *result)
{
    long int i;
    long int get_vec_length(vec_ptr v);
    data_t *get_vec_start(vec_ptr v);
    long int length = get_vec_length(v);
    data_t *a = get_vec_start(v);
    data_t temp;
    temp = a[0];
    data_t xpwr = *x;

    

    for(i=1; i<length; i++)
    {
        temp = temp + a[i]*xpwr;
        xpwr = (*x) * xpwr;
    }
    *result = temp;
}

void poly_unrolling(vec_ptr v, data_t *x, data_t *result)
{
    long int i;
    long int get_vec_length(vec_ptr v);
    data_t *get_vec_start(vec_ptr v);
    long int length = get_vec_length(v);
    data_t *a = get_vec_start(v);
    long int limit = length - UNROLL + 1;
    data_t temp;
    temp = a[0];
    data_t xpwr = *x;

    for(i=1; i<limit; i+=UNROLL)
    {
        temp = (temp + (a[i]*xpwr)) + (a[i+1] * ((*x)*xpwr));
        xpwr = (*x) * (*x) * xpwr;   
    }

  /* Finish remaining elements */
  for (; i < length; i++) 
  {
    temp += a[i] * xpwr;
    xpwr = (*x) * xpwr;
  }

    *result = temp;
}


void poly_accumulator(vec_ptr v, data_t *x, data_t *result)
{
    long int i;
    long int get_vec_length(vec_ptr v);
    data_t *get_vec_start(vec_ptr v);
    long int length = get_vec_length(v);
    data_t *a = get_vec_start(v);
    long int limit = length - UNROLL + 1;
    data_t temp1;
    data_t temp2;
    temp1 = a[0];
    temp2 = a[1];
    data_t xpwr = *x;

    for(i=2; i<limit; i+=UNROLL)
    {
        temp1 += (a[i] * xpwr);
        temp2 += (a[i+1] * ((*x)*xpwr));
        xpwr = (*x) * (*x) * xpwr;   
    }
  /* Finish remaining elements */
  for (; i < length; i++) 
  {
    temp1 += a[i] * xpwr;
    xpwr = (*x) * xpwr;
  }
    *result = temp1 + temp2;
}

void poly_associative(vec_ptr v, data_t *x, data_t *result)
{
    long int i;
    long int get_vec_length(vec_ptr v);
    data_t *get_vec_start(vec_ptr v);
    long int length = get_vec_length(v);
    data_t *a = get_vec_start(v);
    long int limit = length - UNROLL + 1;
    data_t temp;
    temp = a[0];
    data_t xpwr = *x;

    for(i=1; i<limit; i+=UNROLL)
    {
        temp = temp + ((a[i]*xpwr) + (a[i+1] * ((*x)*xpwr)));
        xpwr = (*x) * (*x) * xpwr;   
    }

  /* Finish remaining elements */
  for (; i < length; i++) 
  {
    temp += a[i] * xpwr;
    xpwr = (*x) * xpwr;
  }

    *result = temp;
}
