/*************************************************************************/
// gcc -pthread -o test_param test_param1.c

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define NUM_THREADS 5

/************************************************************************/
void *PrintHello(void *threadid)
{
  printf("the value of threadid before type casting: %d !\n", threadid);
  long unsigned int tid = (long unsigned int) threadid;
  
  printf(" Hello World!  It's me, thread # %lu  \n", tid);

  pthread_exit(NULL);
}

/*************************************************************************/
int main(int argc, char *argv[])
{
  int arg,i,j,k,m,t;   	              /* Local variables. */
  pthread_t threads[NUM_THREADS];
  int rc;
  signed char num= -5;

  printf("\n Hello World!  It's me, MAIN!\n");

  for (t = 0; t < NUM_THREADS; t++) {
    printf("In main:  creating thread %d\n", t);
    rc = pthread_create(&threads[t], NULL, PrintHello, (void*) num);
    if (rc) {
      printf("ERROR; return code from pthread_create() is %d\n", rc);
      exit(-1);
    }
  }

  printf("\n It's me MAIN -- Good Bye World!\n");

  pthread_exit(NULL);

}/* end main */

