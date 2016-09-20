/*********************************************************************/
// To compile:
// gcc -O0 -o test_timers test_timers.c
//
// 

#include <stdio.h>
#include <string.h>		// memset
#include <sys/time.h>		// get time of day
#include <sys/times.h>		// get time of day
#include <sys/mman.h>		// mmap
#include <unistd.h>		// getpid

// Used by gettimeofday --> NEED TO CHECK THIS AND POSSIBLY CHANGE FOR CORRECT TIMING
#define GET_TOD_TICS 1000000

// Used by RDTSC --> NEED TO CHECK THIS AND POSSIBLY CHANGE FOR CORRECT TIMING
#define CLK_RATE 2.53e9 

// Used by get_seconds --> NEED TO CHECK THIS AND POSSIBLY CHANGE FOR CORRECT TIMING
#define GET_SECONDS_TICS 100	

struct timeval tim;    // used by gettimeofday

struct timezone tzp;

typedef union {        // used by RDTSC
  unsigned long long int64;
  struct {unsigned int lo, hi;} int32;
} tsc_counter;

// RDTSC is an intrinsic that executes the assembly language instruction rdtsc
#define RDTSC(cpu_c)              \
  __asm__ __volatile__ ("rdtsc" : \
  "=a" ((cpu_c).int32.lo),        \
  "=d"((cpu_c).int32.hi))

inline double usecs_of_timeval(struct timeval * p_tv)
{
    return((double) p_tv->tv_sec) * 1e6 + ((double) p_tv->tv_usec);
}

double get_seconds() { 	/* routine to read time */
    struct tms rusage;
    times(&rusage);	/* UNIX utility: time in clock ticks */
    return (double)(rusage.tms_utime)/(double)(GET_SECONDS_TICS);
}

/*********************************************************************/
int main(int argc, char *argv[])
{
  long long int i, j, k, delta_int[20], steps = 0;
  double sec0, sec,t1, t2, delta[10];	/* timing variables */
  struct timeval tv_start, tv_stop;     /* used by gettimeofday */
  tsc_counter t_0,t_1;                  /* used by RDTSC */

  // Test gettimeofday().  Run 6 times, each with a different number of iterations
  // in the delay loop.  Each time, call gettimeofday before and after the delay loop.
  // Find the difference in the usecond part of the struct and save.
  // Note:  there is a potential bug here:  If we roll over seconds (high order part
  // of struct) the microsecond count would be wrong.  We fix that with a call to
  // "usecs_of_timeval()".

  printf("\n Using gettimeofday: \n");
  
  gettimeofday(&tv_start, NULL);
  for (i = 0; i <= 100000000; i++) steps = steps + 3;
  gettimeofday(&tv_stop, NULL);
  delta[0] = usecs_of_timeval(&tv_stop) - usecs_of_timeval(&tv_start);

  gettimeofday(&tv_start, NULL);
  for (i = 0; i <= 10000000; i++) steps = steps + 3;
  gettimeofday(&tv_stop, NULL);
  delta[1] = usecs_of_timeval(&tv_stop) - usecs_of_timeval(&tv_start);

  gettimeofday(&tv_start, NULL);
  for (i = 0; i <= 1000000; i++) steps = steps + 3;
  gettimeofday(&tv_stop, NULL);
  delta[2] = usecs_of_timeval(&tv_stop) - usecs_of_timeval(&tv_start);

  gettimeofday(&tv_start, NULL);
  for (i = 0; i <= 100000; i++) steps = steps + 3;
  gettimeofday(&tv_stop, NULL);
  delta[3] = usecs_of_timeval(&tv_stop) - usecs_of_timeval(&tv_start);

  gettimeofday(&tv_start, NULL);
  for (i = 0; i <= 10000; i++) steps = steps + 3;
  gettimeofday(&tv_stop, NULL);
  delta[4] = usecs_of_timeval(&tv_stop) - usecs_of_timeval(&tv_start);

  gettimeofday(&tv_start, NULL);
  for (i = 0; i <= 1000; i++) steps = steps + 3;
  gettimeofday(&tv_stop, NULL);
  delta[5] = usecs_of_timeval(&tv_stop) - usecs_of_timeval(&tv_start);

  for (i = 0; i <= 5; i++) 
    printf("\n Time = %15.9f", delta[i]/(double)(GET_TOD_TICS));

  // Test RDTSC().  Run 6 times, each with a different number of iterations
  // in the delay loop.  Each time, call RDTSC before and after the delay loop.
  // Find the difference in the usecond part of the struct and save.
  // Note:  there is a potential bug here:  If we roll over seconds (high order part
  // of struct) the microsecond count would be wrong.  Has that been fixed?

  printf("\n\n Using RDTSC: \n");

  RDTSC(t_0);
  for (i = 0; i <= 100000000; i++) steps = steps + 3;
  RDTSC(t_1);
  delta_int[0] = t_1.int64-t_0.int64;

  RDTSC(t_0);
  for (i = 0; i <= 10000000; i++) steps = steps + 3;
  RDTSC(t_1);
  delta_int[1] = t_1.int64-t_0.int64;

  RDTSC(t_0);
  for (i = 0; i <= 1000000; i++) steps = steps + 3;
  RDTSC(t_1);
  delta_int[2] = t_1.int64-t_0.int64;

  RDTSC(t_0);
  for (i = 0; i <= 100000; i++) steps = steps + 3;
  RDTSC(t_1);
  delta_int[3] = t_1.int64-t_0.int64;

  RDTSC(t_0);
  for (i = 0; i <= 10000; i++) steps = steps + 3;
  RDTSC(t_1);
  delta_int[4] = t_1.int64-t_0.int64;

  RDTSC(t_0);
  for (i = 0; i <= 1000; i++) steps = steps + 3;
  RDTSC(t_1);
  delta_int[5] = t_1.int64-t_0.int64;

  RDTSC(t_0);
  for (i = 0; i <= 100; i++) steps = steps + 3;
  RDTSC(t_1);
  delta_int[6] = t_1.int64-t_0.int64;

  RDTSC(t_0);
  for (i = 0; i <= 10; i++) steps = steps + 3;
  RDTSC(t_1);
  delta_int[7] = t_1.int64-t_0.int64;

  for (i = 0; i <= 7; i++)
    printf("\nTime = %15.9f (cycles = %lld) ",
	   (float)delta_int[i]/CLK_RATE ,delta_int[i]);

  // Test times() through get_seconds().

  printf("\n\n Using times(): \n");

  sec0 = get_seconds();			
  for (i = 0; i <= 100000000; i++) steps = steps + 2;
  sec = (get_seconds() - sec0);	
  printf("\n Seconds = %14.8f", sec);
  
  sec0 = get_seconds();		
  for (i = 0; i <= 10000000; i++) steps = steps + 2;
  sec = (get_seconds() - sec0);
  printf("\n Seconds = %14.8f", sec);
  
  sec0 = get_seconds();		
  for (i = 0; i <= 1000000; i++) steps = steps + 2;
  sec = (get_seconds() - sec0);	
  printf("\n Seconds = %14.8f", sec);

  sec0 = get_seconds();		
  for (i = 0; i <= 100000; i++) steps = steps + 2;
  sec = (get_seconds() - sec0);
  printf("\n Seconds = %14.8f", sec);

  sec0 = get_seconds();		
  for (i = 0; i <= 10000; i++) steps = steps + 2;
  sec = (get_seconds() - sec0);	
  printf("\n Seconds = %14.8f", sec);

  sec0 = get_seconds();		
  for (i = 0; i <= 1000; i++) steps = steps + 2;
  sec = (get_seconds() - sec0);
  printf("\n Seconds = %14.8f", sec);

  sec0 = get_seconds();		
  for (i = 0; i <= 100; i++) steps = steps + 2;
  sec = (get_seconds() - sec0);
  printf("\n Seconds = %14.8f", sec);

  sec0 = get_seconds();		
  for (i = 0; i <= 10; i++) steps = steps + 2;
  sec = (get_seconds() - sec0);
  printf("\n Seconds = %14.8f", sec);

  sec0 = get_seconds();		
  for (i = 0; i <= 1; i++) steps = steps + 2;
  sec = (get_seconds() - sec0);
  printf("\n Seconds = %14.8f", sec);

  printf("\n done\n");
}








