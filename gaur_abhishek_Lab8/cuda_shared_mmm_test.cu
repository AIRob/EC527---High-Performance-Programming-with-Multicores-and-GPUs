#include <cstdio>
#include <cstdlib>
#include <math.h>

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

#define NUM_THREADS_PER_BLOCK 	256
#define NUM_BLOCKS 		16
#define PRINT_TIME 		1
#define SM_ARR_LEN		2048*2048
#define TOL			1e-6
#define GIG                     1000000000
#define CPG                     3.07
#define IMUL(a, b) __mul24(a, b)
#define TILE_WIDTH 		16

void initializeArray1D(float *arr, int len, float seed);

__global__ void MatrixMulShared(float *Md, float *Nd, float *Pd, int Width) 
{
	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];  // Shared memory
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];  //   declarations
	int bx = blockIdx.x;
	int by = blockIdx.y;    // ID thread
	int tx = threadIdx.x; 
	int ty = threadIdx.y;	// Identify the row and column of the Pd element to work on
	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;
	float Pvalue = 0; 
	// REGISTER!
	// Loop over the Md and Nd tiles required to compute the Pd element
	for (int m = 0; m < Width/TILE_WIDTH; ++m) {
		// Collaborative loading of Md and Nd tiles into shared memory
	Mds[ty][tx] = Md[Row*Width + (m*TILE_WIDTH + tx)];
	Nds[ty][tx] = Nd[Col + (m*TILE_WIDTH + ty)*Width];
	__syncthreads();
	
	for (int k = 0; k < TILE_WIDTH; ++k)
		Pvalue += Mds[ty][k] * Nds[k][tx];
	__syncthreads();
	}
	Pd[Row*Width+Col] = Pvalue;
}








__global__ void MMK(int width, float* Md, float* Nd, float* Pd)
{

	int row = blockDim.y * blockIdx.y + threadIdx.y;
        int col = blockDim.x * blockIdx.x + threadIdx.x;
        int k;
        float Pvalue = 0.0f;
        if(row < width || col < width) {
                for(k = 0; k < width; k++){
                        Pvalue += Md[row * width + k] * Nd[k * width + col];
                }
                Pd[row * width + col] = Pvalue;
        }
}

int main(int argc, char **argv){
	int arrLen = 0;
		
	// GPU Timing variables
	cudaEvent_t start, stop, start2, stop2;
	float elapsed_gpu;
	
	// Arrays on GPU global memoryc
	float *Md;
	float *Nd;
	float *Pd;

	// Arrays on the host memory
	float *Md_h;
	float *Pd_h;
	float *Nd_h;
	float *Pd_h_gold;
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
	size_t allocSize = arrLen * sizeof(float);
	CUDA_SAFE_CALL(cudaMalloc((void **)&Md, allocSize));
	CUDA_SAFE_CALL(cudaMalloc((void **)&Pd, allocSize));
	CUDA_SAFE_CALL(cudaMalloc((void **)&Nd, allocSize));
		
	// Allocate arrays on host memory
	Pd_h		           = (float *) malloc(allocSize);
	Pd_h_gold		   = (float *) malloc(allocSize);
	Md_h		           = (float *) malloc(allocSize);
	Nd_h		           = (float *) malloc(allocSize);

	
	// Initialize the host arrays
	printf("\nInitializing the arrays ...");
	// Arrays are initialized with a known seed for reproducability
	initializeArray1D(Md_h, arrLen, 24.53);
	initializeArray1D(Nd_h, arrLen, 30.53);
	printf("\t... done\n\n");
	
	
#if PRINT_TIME
	// Create the cuda events
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Record event on the default stream
	cudaEventRecord(start, 0);
#endif
	
	// Transfer the arrays to the GPU memory
	CUDA_SAFE_CALL(cudaMemcpy(Md, Md_h, allocSize, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(Nd, Nd_h, allocSize, cudaMemcpyHostToDevice));
	cudaEventCreate(&start2);
	cudaEventCreate(&stop2);
	cudaEventRecord(start2, 0);
	dim3 dimGrid(128,128);
	dim3 dimBlock(16,16); 
	// Launch the kernel
	MatrixMulShared<<<dimGrid, dimBlock>>>(Md, Nd, Pd, 2048);

	cudaEventRecord(stop2,0);
	cudaEventSynchronize(stop2);
	cudaEventElapsedTime(&elapsed_gpu, start2, stop2);
	printf("\nGPU time: %f (msec)\n", elapsed_gpu);
	cudaEventDestroy(start2);
	cudaEventDestroy(stop2);

	// Check for errors during launch
	CUDA_SAFE_CALL(cudaPeekAtLastError());
	
	// Transfer the results back to the host
	CUDA_SAFE_CALL(cudaMemcpy(Pd_h,Pd, allocSize, cudaMemcpyDeviceToHost));
	
#if PRINT_TIME
	// Stop and destroy the timer
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_gpu, start, stop);
	printf("\nGPU time: %f (msec)\n", elapsed_gpu);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
#endif
	
	// Compute the results on the host
        struct timespec diff(struct timespec start, struct timespec end);
        struct timespec time1, time2;
        struct timespec time_stamp;
	
	printf("Calculating Results on Host: \n");
	int Width = 2048;
	clock_gettime(CLOCK_REALTIME, &time1);
	for (int i = 0; i < Width; ++i){
		for (int j = 0; j < Width; ++j) {
			float sum = 0;
			for (int k = 0; k < Width; ++k) {
				float a = Md_h[i*Width + k];
				float b = Nd_h[k*Width + j];
				sum += a * b;
			}
			Pd_h_gold[i * Width + j] = sum;
		}
	}
	clock_gettime(CLOCK_REALTIME, &time2);
        time_stamp = diff(time1,time2);

        printf("%lf\n", ((double) (GIG * time_stamp.tv_sec + time_stamp.tv_nsec)/1000000));

	// Compare the results
	for(i = 0; i < arrLen; i++) {
		if (abs(Pd_h_gold[i] - Pd_h[i]) > TOL) {
			errCount++;
		}
		if (Pd_h[i] == 0) {
			zeroCount++;
		}
	}
	
	/*
	for(i = 0; i < 50; i++) {
		printf("%d:\t%.8f\t%.8f\n", i, h_result_gold[i], h_result[i]);
	}
	*/
	
	if (errCount > 0) {
		printf("\n@ERROR: TEST FAILED: %d results did not matched\n", errCount);
	}
	else if (zeroCount > 0){
		printf("\n@ERROR: TEST FAILED: %d results (from GPU) are zero\n", zeroCount);
	}
	else {
		printf("\nTEST PASSED: All results matched\n");
	}
	
	// Free-up device and host memory
	CUDA_SAFE_CALL(cudaFree(Pd));
	CUDA_SAFE_CALL(cudaFree(Md));
	CUDA_SAFE_CALL(cudaFree(Nd));
		   
	free(Pd_h);
	free(Md_h);
	free(Nd_h);
		
	return 0;
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


void initializeArray1D(float *arr, int len, float seed) {
	int i;
	float randNum;
	srand(seed);

	for (i = 0; i < len; i++) {
		randNum = (float) (rand()%10000);
		arr[i] = randNum;
	}
}

