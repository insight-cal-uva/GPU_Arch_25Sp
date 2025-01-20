#include<stdio.h>
#include<time.h>
//vector length
#define N 10000 

//vector addition on CPU
void serial_vector_add(int *A, int *B, int* result) {
    for (int i=0; i<N; i++) {
        result[i] = A[i] + B[i];
    }
}

// vector addition on GPU
__global__ void parallel_vector_add(int *A, int *B, int *result) {
    int id = blockDim.x * blockIdx.x + threadIdx.x ;
    if(id < N)  {
        result[id] = A[id] + B[id];
    }
}

void compare_results(int * result, int* out) {
    for(int i=0; i<N; i++) {
        if(result[i] != out[i]) {
            fprintf(stderr, "incorrect logic on GPU!\n");
            return;   
        }
    }
    printf("Correctnes check done!\n");
}
int main() {
    //size of array in bytes
    size_t bytes = sizeof(int) * N ;
    //Allocate memory on host
    int *A = (int *) malloc (bytes);
    int *B = (int *) malloc (bytes);
    int *result = (int *) malloc (bytes);   //stores CPU calculated result
    int *out = (int *) malloc (bytes);  // stores copied output from GPU

    //initialize inputs
    for (int i=0; i<N; i++) {
        A[i] = rand() % N ; 
        B[i] = rand() % N ;
    }

    struct timespec begin, end; 
    //calculate vector addition on CPU
    clock_gettime(CLOCK_REALTIME, &begin);
    serial_vector_add(A, B, result);
    clock_gettime(CLOCK_REALTIME, &end);
    double elapsed_msec = (end.tv_sec - begin.tv_sec)*1e3  
                            + (end.tv_nsec - begin.tv_nsec)*1e-6;
    
    printf("Vector size: %d\nElapsed time on CPU: %.6f ms.\n", N, elapsed_msec);
    /*for (int i =0; i<N; i++) {
        printf("%d %d %d\n", A[i], B[i], result[i]);
    }*/

    //Allocate memory on device
    int *A_d, *B_d, *out_d;
    cudaMalloc(&A_d, bytes);
    cudaMalloc(&B_d, bytes);
    cudaMalloc(&out_d, bytes);


    //define block and grid dimensions
    const int blk_sz = 32;
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
    cudaMemcpy(B_d, B, bytes, cudaMemcpyHostToDevice);
    //launch cuda kernel vector addition
    parallel_vector_add<<<grid_dim, blk_dim>>>(A_d, B_d, out_d);
    //wait for all threads to finish execution on device
    //cudaDeviceSynchronize();
    //copy results back to host
    cudaMemcpy(out, out_d, bytes, cudaMemcpyDeviceToHost);

    //record stop event 
    cudaEventRecord(end_d,0);
    cudaEventSynchronize(end_d);
    float elapsed_gpu_ms;
    //calculate time elapsed between the two events
    cudaEventElapsedTime(&elapsed_gpu_ms, begin_d, end_d);
    printf("Time taken by shared memory KERNEL to execute is: %.6f ms\n", elapsed_gpu_ms);

    //check correctness of the result
    compare_results(result, out);
    
    //free the memory allocated on GPU
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(out_d);

    //free the memory allocated on CPU
    free(A);
    free(B);
    free(result);
    free(out);
}
