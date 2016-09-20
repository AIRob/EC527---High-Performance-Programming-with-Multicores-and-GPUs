/*****************************************************************************/

// gcc -o test_combine1-7 test_combine1-7.c -lrt

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>

#define GIG 1000000000
#define CPG 2.9
#define SIZE 10000000
#define ITERS 20
#define DELTA 10
#define BASE 0
#define OPTIONS 3
#define IDENT 1.0
#define OP *

typedef double data_t;

/* Create abstract data type for vector */
typedef struct {
  long int len;
  data_t *data;
} vec_rec, *vec_ptr;

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
  data_t *data_holder;
  
  void dot_prod(vec_ptr v0,vec_ptr v1,data_t *dest);
  void dot_prod_unroll(vec_ptr v0,vec_ptr v1, data_t *dest);
  void dot_prod_paral(vec_ptr v0,vec_ptr v1, data_t *dest);

  long int i, j, k;
  long long int time_sec, time_ns;
  long int MAXSIZE = BASE+(ITERS+1)*DELTA;

  printf("\n Hello World -- psum examples\n");

  // declare and initialize the vector structure
  vec_ptr v0 = new_vec(MAXSIZE);
  vec_ptr v1 = new_vec(MAXSIZE);
  data_holder = (data_t *) malloc(sizeof(data_t));
  init_vector(v0, MAXSIZE);
  init_vector(v1, MAXSIZE);

  // execute and time all 7 options from B&O 
  OPTION = 0;
  for (i = 0; i < ITERS; i++) {
    set_vec_length(v0,BASE+(i+1)*DELTA);
    set_vec_length(v1,BASE+(i+1)*DELTA);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
    dot_prod(v0,v1,data_holder);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
    time_stamp[OPTION][i] = diff(time1,time2);
  }

  OPTION++;
  for (i = 0; i < ITERS; i++) {
    set_vec_length(v0,BASE+(i+1)*DELTA);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
    dot_prod_unroll(v0,v1,data_holder);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
    time_stamp[OPTION][i] = diff(time1,time2);
  }

  OPTION++;
  for (i = 0; i < ITERS; i++) {
    set_vec_length(v0,BASE+(i+1)*DELTA);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
    dot_prod_paral(v0,v1,data_holder);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
    time_stamp[OPTION][i] = diff(time1,time2);
  }
  /* output times */
  for (i = 0; i < ITERS; i++) {
    printf("\n%d,  ", BASE+(i+1)*DELTA);
    for (j = 0; j < OPTIONS; j++) {
      if (j != 0) printf(", ");
     // printf("%d%d", time_stamp[j][i].tv_sec, time_stamp[j][i].tv_nsec);
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
/* DOT PRODUCT METHOD AS IN COMBINE4 */
void dot_prod(vec_ptr v0,vec_ptr v1, data_t *dest)
{
  long int i;
  long int get_vec_length(vec_ptr v0);
  data_t *get_vec_start(vec_ptr v0);
  long int length = get_vec_length(v0);
  data_t *data1 = get_vec_start(v0);
  data_t *data2 = get_vec_start(v1);
  data_t acc = IDENT;

  for (i = 0; i < length; i++) {
    acc += data1[i] OP data2[i];
  }
  *dest = acc;
}

/* DOT PRODUCT LOOP UNROLLING METHOD */  /* BEST METHOD TO PERFORM DOT PRODUCT */
void dot_prod_unroll(vec_ptr v0,vec_ptr v1,data_t *dest)
{
  long int i;
  long int get_vec_length(vec_ptr v0);
  data_t *get_vec_start(vec_ptr v0);
  long int length = get_vec_length(v0);
  long int limit = length - 1;
  data_t *data1 = get_vec_start(v0);
  data_t *data2 = get_vec_start(v1);
  data_t acc = IDENT;

  /* Combine two elements at a time */
  for (i = 0; i < limit; i+=2) {
    acc += (data1[i] OP data2[i]) + (data1[i+1] OP data2[i+1]);
  }

  /* Finish remaining elements */
  for (; i < length; i++) {
    acc += data1[i] OP data2[i];
  }
  *dest = acc;
}

/* DOT PRODUCT PARALLELIZATION METHOD */
void dot_prod_paral(vec_ptr v0,vec_ptr v1,data_t *dest)
{
  long int i;
  long int get_vec_length(vec_ptr v0);
  data_t *get_vec_start(vec_ptr v0);
  long int length = get_vec_length(v0);
  long int limit = length - 1;
  data_t *data1 = get_vec_start(v0);
  data_t *data2 = get_vec_start(v1);
  data_t acc0 = IDENT;
  data_t acc1 = IDENT;

  /* Combine two elements at a time w/ 2 acculators */
  for (i = 0; i < limit; i+=2) {
    acc0 += data1[i] OP data2[i];
    acc1 += data1[i+1] OP data2[i+1];
  }

  /* Finish remaining elements */
  for (; i < length; i++) {
    acc0 += data1[i] OP data2[i];
  }
  *dest = acc0 + acc1;
  }
