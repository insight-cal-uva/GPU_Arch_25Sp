#include<stdio.h>
#include<time.h>
//vector length
//#define N 10000 

//histogram calculation on CPU
void serial_histogram(int *A, int N, int* result) {
    for (int i=0; i<N; i++) {
        result[A[i]]++ ; 
    }
}

// Histogram on GPU
__global__ void naive_parallel_histogram(int *A, int N, int *result) {
    int id = blockDim.x * blockIdx.x + threadIdx.x ;
    
    if(id < N)  {
        //update partial histogram in global memory
        atomicAdd(&result[A[id]], 1);
    }
    
}

// Histogram on GPU using shared memory
__global__ void parallel_histogram(int *A, int N, int *result) {
    int id = blockDim.x * blockIdx.x + threadIdx.x ;
    int tid = threadIdx.x;
    __shared__ int tileSh[256];

    if(tid<256) {       
        //initialize shared memory with zero 
        tileSh[tid] = 0;
    }
    
    if(id < N)  {
        //update partial histogram in shared memory
        atomicAdd(&tileSh[A[id]], 1);
    }
    
    if(tid<256) {
        //update global memory with partial results in shared memory
        atomicAdd(&result[tid], tileSh[tid]);
    }
}

void compare_results(int * result, int* out) {
    for(int i=0; i<256; i++) {
        if(result[i] != out[i]) {
            fprintf(stderr, "incorrect logic on GPU!\n");
            return;   
        }
    }
    printf("Correctnes check done!\n");
}
int main(int argc, char* argv[]) {
    //vector length
    int N = 10000;  //default

    //take input from user
    if(argc == 2 ) {
		N = atoi(argv[1]);
	}
    printf("Vector length : %d\n", N);
    //size of array in bytes
    size_t bytes = sizeof(int) * N ;
    //Allocate memory on host
    int *A = (int *) malloc (bytes);
    int *result = (int *) malloc (256*sizeof(int));   //stores CPU calculated result
    int *out = (int *) malloc (256*sizeof(int));  // stores copied output from GPU

    //initialize inputs
    for (int i=0; i<N; i++) {
        A[i] = rand() % 256 ; 
    }

    //initialize output array
    for(int i=0; i<256; i++) {
        result[i] = 0;
    }

    struct timespec begin, end; 
    //calculate vector addition on CPU
    clock_gettime(CLOCK_REALTIME, &begin);
    serial_histogram(A, N, result);
    clock_gettime(CLOCK_REALTIME, &end);
    double elapsed_msec = (end.tv_sec - begin.tv_sec)*1e3  
                            + (end.tv_nsec - begin.tv_nsec)*1e-6;
    
    printf("Elapsed time on CPU: %.6f ms.\n", elapsed_msec);
   
    //Allocate memory on device
    int *A_d, *out_d;
    cudaMalloc(&A_d, bytes);
    //cudaMalloc(&B_d, bytes);
    cudaMalloc(&out_d, 256*sizeof(int));


    //define block and grid dimensions
    const int blk_sz = 32*32;
    dim3 blk_dim(blk_sz); 
    dim3 grid_dim((N + blk_sz - 1) / blk_sz); 

    //create timing events on device
    cudaEvent_t begin_d, end_d;
    cudaEventCreate(&begin_d);
    cudaEventCreate(&end_d);
    //record begin timestamp
    cudaEventRecord(begin_d, 0);
    //copy input arrays on device
    cudaMemcpy(A_d, A, bytes, cudaMemcpyHostToDevice);
    cudaMemset(out_d, 0, 256*sizeof(int));
    //cudaMemcpy(B_d, B, bytes, cudaMemcpyHostToDevice);
    //launch cuda kernel vector addition
    naive_parallel_histogram<<<grid_dim, blk_dim>>>(A_d, N, out_d);
    //wait for all threads to finish execution on device
    //cudaDeviceSynchronize();
    //copy results back to host
    cudaMemcpy(out, out_d, 256*sizeof(int), cudaMemcpyDeviceToHost);

    //record stop event 
    cudaEventRecord(end_d,0);
    cudaEventSynchronize(end_d);
    float elapsed_gpu_ms;
    //calculate time elapsed between the two events
    cudaEventElapsedTime(&elapsed_gpu_ms, begin_d, end_d);
    printf("Elapsed time on GPU with naive implementation: %.6f ms\n", elapsed_gpu_ms);

    //Histogram using shared memory
    //check correctness of the result
    compare_results(result, out);

    //record begin timestamp
    cudaEventRecord(begin_d, 0);

    cudaMemset(out_d, 0, 256*sizeof(int));
    //cudaMemcpy(B_d, B, bytes, cudaMemcpyHostToDevice);
    //launch cuda kernel vector addition
    parallel_histogram<<<grid_dim, blk_dim>>>(A_d, N, out_d);
    //wait for all threads to finish execution on device
    //cudaDeviceSynchronize();
    //copy results back to host
    cudaMemcpy(out, out_d, 256*sizeof(int), cudaMemcpyDeviceToHost);

    //record stop event 
    cudaEventRecord(end_d,0);
    cudaEventSynchronize(end_d);
    //calculate time elapsed between the two events
    cudaEventElapsedTime(&elapsed_gpu_ms, begin_d, end_d);
    printf("Elapsed time on GPU with SHM: %.6f ms\n", elapsed_gpu_ms);
    //check correctness of the result
    compare_results(result, out);
    
    //free the memory allocated on GPU
    cudaFree(A_d);
    //cudaFree(B_d);
    cudaFree(out_d);

    //free the memory allocated on CPU
    free(A);
    //free(B);
    free(result);
    free(out);
}
