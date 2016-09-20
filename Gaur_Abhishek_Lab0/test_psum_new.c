/******************************************************************************/

// gcc -O0 -o test_psum test_psum.c -lrt

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>

#define SIZE 1000000000

#define CPU_SPEED 0.395e-9

/******************************************************************************/
void psum1(float a[], float p[], long int n)
{
  long int i;

  p[0] = a[0];
  for (i = 1; i < n; i++)
    p[i] = p[i-1] + a[i];

}

void psum2(float a[], float p[], long int n)
{
  long int i;

  p[0] = a[0];
  for (i = 1; i < n-1; i+=2) {
    float mid_val = p[i-1] + a[i];
    p[i] = mid_val;
    p[i+1] = mid_val + a[i+1];
  }

  /* For odd n, finish remaining element */
  if (i < n)
    p[i] = p[i-1] + a[i];
}

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

main(int argc, char *argv[])
{
  int clock_gettime(clockid_t clk_id, struct timespec *tp);
  void psum1(float a[], float p[], long int n);
  void psum2(float a[], float p[], long int n);
  float *in, *out;
  long int i, j, k, m;
  struct timespec start_time, stop_time, elapsed_time; 
  int num_of_cycles;

  // initialize
  in = (float *) malloc(SIZE * sizeof(*in));
  out = (float *) malloc(SIZE * sizeof(*out));
  for (i = 0; i < SIZE; i++) in[i] = (float)(i);
  
  // process psum1 for various array sizes and collect timing
  
  printf("For psum1:\n");
  for(m=0; m<=400; m=m+20)
  {
  	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start_time);
  	psum1(in, out, m);
  	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop_time);
  	elapsed_time = diff(start_time, stop_time);
  	printf("%d \t %d \n", m,(int)( elapsed_time.tv_nsec/((double)SIZE * CPU_SPEED)));

  }
  // process psum2 for various array sizes and collect timing

  printf("For psum2:\n");


  for(m=0; m<=400; m=m+20)
  {
  	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start_time);
  	psum2(in, out, m);
  	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop_time);
  	elapsed_time = diff(start_time, stop_time);	
	
  	printf("%d \t %d \n", m, (int)( elapsed_time.tv_nsec/((double)SIZE * CPU_SPEED)));
  }
  // output timing

  
}/* end main */

