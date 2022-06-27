#include "xt_cuda.h"
#include<math.h>
#include "device_launch_parameters.h"

#define num_threads   256

static __global__ void compute_kernel(int* input_output, int length)
{
	int position = blockDim.x * blockIdx.x + threadIdx.x;
	if (position > length) return;

	input_output[position] *= 5;
	// �õ���ǰ�̵߳�id

	// gridDim   ��ʾgrid layout�Ĵ�С��ά�ȣ�shape�������ǵ�����������3, 3,  gridDim.x = 3, gridDim.y = 3
	// blockDim  ��ʾblock layout�Ĵ�С��ά�ȣ�shape�������ǵ�����������2, 2,  blockDim.x = 2, blockDim.y = 2
	// blockIdx  ��ʾ��grid��ȡһ�飬��¼�ÿ��������λ�á� �����ǵ������У�����ֵ�ǣ�blockIdx.x = 1, blockIdx.y = 2
	// threadIdx ��ʾ��block��ȡһ������¼���̵߳�������λ�ã������ٷֵĵ�Ԫ����thread���롣������һ���̡߳��������ǵ�������ֵ�ǣ�threadIdx.x = 0�� threadIdx.y = 0
	// flatten_index = position
	// position ��ʾ��ǰ�̶߳�Ӧ�����ڴ�ռ��е�����λ��
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

	// ����stream
	// 1. stream���Ϊ������С���������ڵ����������Ǵ��е�
	// 2. stream֮�䣬�����ǲ��е�
	// 3. ���ָ��stream=nullptr����ʹ��Ĭ������Ҳ����˵��ÿ��context������һ��Ĭ����
	// һ��context������n��stream
	// һ��device������n��context

	cudaSetDevice(0);
	checkRuntime(cudaStreamCreate(&stream));
	checkRuntime(cudaEventCreate(&record_start));
	checkRuntime(cudaEventCreate(&record_stop));

	for (int i = 0; i < array_size; ++i)
		ihost[i] = i + 1;

	checkRuntime(cudaMalloc(&idevice, bytes));
	// ��һ���õ���Ĭ����
	// �ڶ���ִ��������cudaDeviceSynchronize
	// checkRuntime(cudaMemcpy(idevice, ihost, bytes, cudaMemcpyHostToDevice));
	//�������첽�ķ�ʽ
	checkRuntime(cudaMemcpyAsync(idevice, ihost, bytes, cudaMemcpyHostToDevice, stream));

	dim3 threads(array_size < num_threads ? array_size : num_threads);    // 1024 1024 64
	dim3 blocks(ceil(array_size / (float)threads.x));

	checkRuntime(cudaEventRecord(record_start, stream));
	checkKernel(compute_kernel << <blocks, threads, 0, nullptr >> > (idevice, array_size)); //shared_memory --> sizeof(t)
	checkRuntime(cudaEventRecord(record_stop, stream));

	checkRuntime(cudaMemcpyAsync(ihost, idevice, bytes, cudaMemcpyDeviceToHost));
	checkRuntime(cudaDeviceSynchronize()); //�ȴ�ִ�����

	float time = 0;
	checkRuntime(cudaEventElapsedTime(&time, record_start, record_stop));
	printf("kernel func elapse %d ms\n", time);

	// ���Ҫ�����Ƿ���ȷ��һ��ȡ��ͷ��������β����������󲿷ַ����ڱ߽��ϡ��߽���ƺ����׳���
	printf("ihost::\n");
	for (int i = array_size - 100; i < array_size; ++i) {
		printf("%d,", ihost[i]);

		if ((i + 1) % 20 == 0) {
			printf("\n");
		}
	}
	printf("\n");
}
