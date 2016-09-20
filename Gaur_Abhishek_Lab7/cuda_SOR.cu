#include <cstdlib>
#include <math.h>
#include <time.h>
#include <cstdio>

// Assertion to check for errors
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
        if (code != cudaSuccess)
        {
                fprintf(stderr,"CUDA_SAFE_CALL: %s %s %d\n", cudaGetErrorString(code), file, line);
                if (abort) exit(code);
        }
}

#define GIG                     1000000000
#define CPG                     3.07
#define PRINT_TIME              1
#define SM_ARR_LEN              2048*2048
#define OMEGA                   1.98

void initializeArray1D(float *arr, int len, int seed);

__global__ void SOR_add (float* x)
{

        int row_start = threadIdx.x;
        int col_start = threadIdx.y;
        int i;
        float result = 0;

        int offset = (col_start * 2048) + row_start;
	

        while(offset<SM_ARR_LEN)
        {       if((offset > 2047)  && (offset%2048) && (offset%2047) && (offset < SM_ARR_LEN-2047))
		{
                	for(i=0; i<2000; i++)
                	{
                        	result = x[offset] - 0.25*(x[offset-1] + x[offset+1] + x[offset - blockDim.x] + x[offset + blockDim.x]);
						

                        	__syncthreads();
                        	x[offset] -= result*OMEGA;
                        	__syncthreads();

                	}
		}
                offset += blockDim.x*blockDim.y;
        }
}




int main(int argc, char **argv){
        int arrLen = 0;
        struct timespec diff(struct timespec start, struct timespec end);
        struct timespec time1, time2;
        struct timespec time_stamp;

        // GPU Timing variables
        cudaEvent_t start, stop;
        float elapsed_gpu;

        // Arrays on GPU global memory
        float *d_x;
        // Arrays on the host memory
        float *h_x;

        int i, errCount = 0, zeroCount = 0;

        if (argc > 1) {
                arrLen  = atoi(argv[1]);
        }
        else {
                arrLen = SM_ARR_LEN;
        }

        printf("Length of the array = %d\n", arrLen);
    	// Select GPU
   	 CUDA_SAFE_CALL(cudaSetDevice(1));

        // Allocate GPU memory
        size_t allocSize = arrLen * sizeof(float); printf("\ncp1");
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_x, allocSize)); printf("\ncp2");

        // Allocate arrays on host memory
        h_x = (float *) malloc(allocSize); printf("\ncp4");



        // Initialize the host arrays
        printf("\nInitializing the arrays ...");
        // Arrays are initialized with a known seed for reproducability
        initializeArray1D(h_x, arrLen, 2453);
        //initializeArray1D(h_y, arrLen, 1467);
        printf("\t... done\n\n");


#if PRINT_TIME
        // Create the cuda events
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        // Record event on the default stream
        cudaEventRecord(start, 0);
#endif

        // Transfer the arrays to the GPU memory
        CUDA_SAFE_CALL(cudaMemcpy(d_x, h_x, allocSize, cudaMemcpyHostToDevice));
        dim3 dimBlock(16,16);
        // Launch the kernel
        SOR_add<<<1,dimBlock>>>(d_x);
        // Check for errors during launch
       
 	CUDA_SAFE_CALL(cudaPeekAtLastError());
      
        // Transfer the results back to the host
        CUDA_SAFE_CALL(cudaMemcpy(h_x, d_x, allocSize, cudaMemcpyDeviceToHost));



#if PRINT_TIME
        // Stop and destroy the timer
        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_gpu, start, stop);
        printf("\nGPU time: %f (msec)\n", elapsed_gpu);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
#endif

        long int length = 2048;
        int j,k;
        float change;
        // Compute the results on the host
        printf("\ncalculating results on host\n");
        clock_gettime(CLOCK_REALTIME, &time1);
	for(k=0;k<2000;k++)
	{
        	for (i = 1; i < length-1; i++){
	        for (j = 1; j < length-1; j++) {
        			change = h_x[i*length+j] - .25 * (h_x[(i-1)*length+j] +
                                          h_x[(i+1)*length+j] +
                                          h_x[i*length+j+1] +
                                          h_x[i*length+j-1]);
			        h_x[i*length+j] -= change * OMEGA;
			}
		}
	}
        clock_gettime(CLOCK_REALTIME, &time2);
        time_stamp = diff(time1,time2);
      	printf("%lf\n", ((double) (GIG * time_stamp.tv_sec + time_stamp.tv_nsec)/1000000));


        // Free-up device and host memory
        CUDA_SAFE_CALL(cudaFree(d_x));
        //CUDA_SAFE_CALL(cudaFree(d_y));

        free(h_x);
        //free(h_y);

        return 0;
}

void initializeArray1D(float *arr, int len, int seed) {
        int i;
        float randNum;
        srand(seed);

        for (i = 0; i < len; i++) {
                randNum = (float) rand();
                arr[i] = randNum;
        }
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

