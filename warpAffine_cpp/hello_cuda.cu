#include "xt_cuda.h"
#include<math.h>
#include "device_launch_parameters.h"

#define num_threads   256

static __global__ void compute_kernel(int* input_output, int length)
{
	int position = blockDim.x * blockIdx.x + threadIdx.x;
	if (position > length) return;

	input_output[position] *= 5;
	// 拿到当前线程的id

	// gridDim   表示grid layout的大小，维度，shape，在我们的例子里面是3, 3,  gridDim.x = 3, gridDim.y = 3
	// blockDim  表示block layout的大小，维度，shape，在我们的例子里面是2, 2,  blockDim.x = 2, blockDim.y = 2
	// blockIdx  表示在grid中取一块，记录该块的索引，位置。 在我们的例子中，他的值是：blockIdx.x = 1, blockIdx.y = 2
	// threadIdx 表示在block中取一个，记录该线程的索引，位置，不可再分的单元。与thread对齐。表达的是一个线程。。在我们的例子中值是：threadIdx.x = 0， threadIdx.y = 0
	// flatten_index = position
	// position 表示当前线程对应连续内存空间中的索引位置
	// position = 
	// Dim                Index
	// gridDim.z          blockIdx.z
	// gridDim.y          blockIdx.y
	// gridDim.x          blockIdx.x
	// blockDim.z         threadIdx.z
	// blockDim.y         threadIdx.y
	// blockDim.x         threadIdx.x

	// position = ((((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) * blockDim.z + threadIdx.z) * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x
	// blocks = 361, threads = 256
	// Dim                    Index
	// gridDim.z = 1          blockIdx.z = 0
	// gridDim.y = 1          blockIdx.y = 0
	// gridDim.x = 361        blockIdx.x = bix
	// blockDim.z = 1         threadIdx.z = 0
	// blockDim.y = 1         threadIdx.y = 0
	// blockDim.x = 256       threadIdx.x = tix

	// position = bix * blockDim.x + tix

}

void hello_cuda()
{
	int array_size = 1000;
	int bytes = array_size * sizeof(int);
	int* ihost = new int[array_size];
	int* ohost = new int[array_size];
	int* idevice = nullptr;

	//
	for (int i = 0; i < array_size; i++)
	{
		ihost[i] = i + 1;
	}
	checkRuntime(cudaMalloc(&idevice, bytes));
	checkRuntime(cudaMemcpy(idevice, ihost, bytes, cudaMemcpyHostToDevice));

	dim3 threads(array_size < num_threads ? array_size : num_threads) ;
	dim3 blocks(ceil(array_size / (float)threads.x));

	checkKernel(compute_kernel << <blocks, threads, 0, nullptr >> > (idevice, array_size));

	cudaMemcpy(ihost, idevice, bytes, cudaMemcpyDeviceToHost);

	printf("ihost = ");
	for (int i = array_size - 100; i < array_size; ++i) {
		printf("%d,", ihost[i]);

		if ((i + 1) % 20 == 0) {
			printf("\n");
		}
	}
	printf("\n");
}

void test_stream()
{
	int array_size = 1000;
	int bytes = array_size * sizeof(int);
	int* ihost = new int[array_size];
	int* ohost = new int[array_size];
	int* idevice = nullptr;
	cudaStream_t stream = nullptr;
	cudaEvent_t record_start = nullptr, record_stop = nullptr;

	// 创建stream
	// 1. stream理解为任务队列。任务队列内的所有任务都是串行的
	// 2. stream之间，任务是并行的
	// 3. 如果指定stream=nullptr，则使用默认流。也就是说，每个context都会有一个默认流
	// 一个context可以有n个stream
	// 一个device可以有n个context

	cudaSetDevice(0);
	checkRuntime(cudaStreamCreate(&stream));
	checkRuntime(cudaEventCreate(&record_start));
	checkRuntime(cudaEventCreate(&record_stop));

	for (int i = 0; i < array_size; ++i)
		ihost[i] = i + 1;

	checkRuntime(cudaMalloc(&idevice, bytes));
	// 第一，用的是默认流
	// 第二，执行完后进行cudaDeviceSynchronize
	// checkRuntime(cudaMemcpy(idevice, ihost, bytes, cudaMemcpyHostToDevice));
	//这里是异步的方式
	checkRuntime(cudaMemcpyAsync(idevice, ihost, bytes, cudaMemcpyHostToDevice, stream));

	dim3 threads(array_size < num_threads ? array_size : num_threads);    // 1024 1024 64
	dim3 blocks(ceil(array_size / (float)threads.x));

	checkRuntime(cudaEventRecord(record_start, stream));
	checkKernel(compute_kernel << <blocks, threads, 0, nullptr >> > (idevice, array_size)); //shared_memory --> sizeof(t)
	checkRuntime(cudaEventRecord(record_stop, stream));

	checkRuntime(cudaMemcpyAsync(ihost, idevice, bytes, cudaMemcpyDeviceToHost));
	checkRuntime(cudaDeviceSynchronize()); //等待执行完毕

	float time = 0;
	checkRuntime(cudaEventElapsedTime(&time, record_start, record_stop));
	printf("kernel func elapse %d ms\n", time);

	// 如果要调试是否正确。一般取开头几个，结尾几个。错误大部分发生在边界上。边界控制很容易出错
	printf("ihost::\n");
	for (int i = array_size - 100; i < array_size; ++i) {
		printf("%d,", ihost[i]);

		if ((i + 1) % 20 == 0) {
			printf("\n");
		}
	}
	printf("\n");
}
