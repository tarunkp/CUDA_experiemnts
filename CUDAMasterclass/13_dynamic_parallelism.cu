#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_common.cuh"

__global__ void dynamic_parallelism_check(int const size, int depth)
{
	int tid = threadIdx.x;
	int nthread = size >> 1;

	printf("Recursion No : %d | thread id : %d \n",depth,tid);

	if (size == 1)
		return;
	if (tid == 0) 
	{
		//dynamic_parallelism_check <<<1, nthread >> > (nthread,++depth);
	}
}

//int main(int argc, char** argv)
//{
//	printf("\n-----------------------DYNAMIC PARALLELISM EXAMPLE------------------------ \n\n");
//
//	int size = 1 << 5;
//	int depth = 0;
//
//	dynamic_parallelism_check << <1,32 >> > (size,depth);
//	cudaDeviceSynchronize();
//	cudaDeviceReset();
//
//	return 0;
//}