#include "scan.cuh"

#define BLOCK_SIZE 1024

void scan_inclusive_cpu(float *input, float *output, int size)
{
	output[0] = input[0];

	for (int i = 1; i < size; i++)
	{
		output[i] = output[i - 1] + input[i];
	}
}

//Work inefficient scan implementation using shared memory
//This code has race condition inside for loop. So the __syncthreads synchronization is unrelaible
//How ever for small input sizes the program will possible yield the correct resutls
__global__ void scan_inclusive_gpu_ineffcient(float *input, float *output, int size)
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	__shared__ float tile[BLOCK_SIZE];

	if (gid < size)
	{
		tile[tid] = input[gid];
	}
	__syncthreads();

	for (int stride = 1; stride <= tid; stride *= 2)
	{
		tile[tid] += tile[tid - stride];
		__syncthreads();
	}

	output[gid] = tile[tid];
}

__global__ void work_efficient_scan_kernel(float *X, float *Y, float *aux, int InputSize, bool storeInAux)
{
	__shared__ float XY[BLOCK_SIZE];

	int gid = blockIdx.x*blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	int bid = blockIdx.x;

	if (gid < InputSize)
	{
		XY[tid] = X[gid];
	}

	for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
	{
		__syncthreads();

		int index = (tid + 1) * 2 * stride - 1;

		if (index < blockDim.x)
		{
			XY[index] += XY[index - stride];
		}
	}

	for (int stride = BLOCK_SIZE / 4; stride > 0; stride /= 2)
	{
		__syncthreads();

		int index = (tid + 1)*stride * 2 - 1;

		if (index + stride < BLOCK_SIZE)
		{
			XY[index + stride] += XY[index];
		}
	}
	__syncthreads();

	Y[gid] = XY[tid];

	if (tid == (BLOCK_SIZE - 1) && storeInAux)
	{
		if(bid + 1 < gridDim.x)
			aux[bid+1] = Y[gid];
		//printf("bid - %d | gid - %d | tid = %d | value Y - %f | aux - %f\n", bid, gid, tid, Y[gid], aux[bid]);
	}
}

__global__ void scan_block_accumulation(float *output, float* aux_block_results, int size)
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	int bid = blockIdx.x;
	output[gid] += aux_block_results[bid];
}

//int main(int argc, char**argv)
//{
//	int input_size = 1 << 20;
//	
//	if (argc > 1)
//	{
//		input_size = 1 << atoi(argv[1]);
//	}
//	
//	const int byte_size = sizeof(float) * input_size;
//
//	float * h_input = (float*)malloc(byte_size);
//	float * h_output = (float*)malloc(byte_size);
//	float * h_ref = (float*)malloc(byte_size);
//
//	float * h_aux1 = (float*)malloc(byte_size);
//	float * h_aux2 = (float*)malloc(byte_size);
//
//	clock_t cpu_start, cpu_end, gpu_start, gpu_end;
//	
//	cpu_start = clock();
//	initialize(h_input, input_size, INIT_ONE);
//	cpu_end = clock();
//
//	scan_inclusive_cpu(h_input, h_output, input_size);
//
//	float *d_input, *d_output, *d_aux_block_results, *aux_out, *aux_temp;
//
//	dim3 block(BLOCK_SIZE);
//	dim3 grid(input_size / block.x);
//
//	int aux_array_bytes = sizeof(int)* grid.x;
//
//	gpu_start = clock();
//	
//	gpuErrchk(cudaMalloc((void**)&d_input, byte_size));
//	gpuErrchk(cudaMalloc((void**)&d_output, byte_size));
//	gpuErrchk(cudaMalloc((void**)&d_aux_block_results, aux_array_bytes));
//	gpuErrchk(cudaMalloc((void**)&aux_out, aux_array_bytes));
//	gpuErrchk(cudaMalloc((void**)&aux_temp, sizeof(float)));
//
//	gpuErrchk(cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice));
//
//	work_efficient_scan_kernel << <grid, block >> > (d_input, d_output, d_aux_block_results, input_size, true);
//	gpuErrchk(cudaDeviceSynchronize());
//	gpuErrchk(cudaMemcpy(h_aux1, d_aux_block_results, aux_array_bytes, cudaMemcpyDeviceToHost));
//	
//	work_efficient_scan_kernel << <1, 1024 >> > (d_aux_block_results, aux_out, aux_temp, grid.x, false);
//	gpuErrchk(cudaDeviceSynchronize());
//	gpuErrchk(cudaMemcpy(h_aux2, aux_out, aux_array_bytes, cudaMemcpyDeviceToHost));
//
//	print_arrays_toafile_side_by_side(h_aux1, h_aux2, 1024, "aux_array.txt");
//	
//	scan_block_accumulation << <grid, block >> > (d_output, aux_out, input_size);
//	gpuErrchk(cudaDeviceSynchronize());
//	
//	gpuErrchk(cudaMemcpy(h_ref, d_output, byte_size, cudaMemcpyDeviceToHost));
//	print_arrays_toafile_side_by_side(h_ref, h_output, input_size, "input_array.txt");
//
//
//	gpu_end = clock();
//
//	compare_arrays(h_ref, h_output, input_size);
//	 
//	//time calculation
//	double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC;
//	double gpu_time = ((double)(gpu_end - gpu_start)) / CLOCKS_PER_SEC;
//
//	printf("scan CPU execution time : %f \n", cpu_time);
//	printf("scan GPU execution time : %f \n", gpu_time);
//	printf("Speed up : %f \n", cpu_time / gpu_time);
//
//	gpuErrchk(cudaFree(d_output));
//	gpuErrchk(cudaFree(d_aux_block_results));
//	gpuErrchk(cudaFree(aux_out));
//	gpuErrchk(cudaFree(aux_temp));
//	gpuErrchk(cudaFree(d_input));
//
//	free(h_ref);
//	free(h_output);
//	free(h_input);
//
//	gpuErrchk(cudaDeviceReset());
//	return 0;
//}