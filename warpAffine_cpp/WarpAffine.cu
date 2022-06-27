#include "xt_cuda.h"
#include <math.h>
#include <iostream>
#include <time.h>

#include <opencv2/opencv.hpp>
#include "device_launch_parameters.h"

using namespace cv;
using namespace std;

#define num_threads   512

struct AffineMatrix {
	float value[6]; //2x3
};

static __device__ float desigmoid(float x)
{
	return -log(1.0f / x - 1.0f);
}

static __device__ float sigmoid(float x)
{
	return 1.0f / (1.0f + exp(-x));
}

__device__ void affine_project(float* matrix, int x, int y, float* proj_x, float* proj_y)
{
	// m0, m1, m2
	// m3, m4, m5
	*proj_x = matrix[0] * x + matrix[1] * y + matrix[2] + 0.5f;
	*proj_y = matrix[3] * x + matrix[4] * y + matrix[5] + 0.5f;
}
//cv::Mat

__global__ void warpaffine_kernel(
	uint8_t* input_image, int input_image_width, int input_image_height,
	uint8_t* output_image, int output_image_width, int output_image_height,
	AffineMatrix m, uint8_t const_value, int edge)
{
	//��input_point�����в�ֵ���õ�����ֵ������ֵ��output_image[output_point]
	int position = blockDim.x * blockIdx.x + threadIdx.x;
	if (position >= edge) return;

	int output_x = position % output_image_width;
	int output_y = position / output_image_width;
	
	float input_x = 0, input_y = 0;
	affine_project(m.value, output_x, output_y, &input_x, &input_y);

	//��input_point���в�ֵ�����Ҹ�ֵ��output_image[output_point]
	// ���input_point��Ӧ��ԭͼʱԽ���ˣ�������㲻��Ҫ����
	if (input_x < 0 || input_x >= input_image_width || input_y < 0 || input_y >= input_image_height)
	{
		uint8_t* output_pointer = output_image + (output_y*output_image_width + output_x) * 3;
		//bgrbgrbgr
		//bgrbgrbgr
		//(1*3+1)*3
		*output_pointer++ = const_value;
		*output_pointer++ = const_value;
		*output_pointer++ = const_value;
		return;
	}

	int y_low = floor(input_y);
	int x_low = floor(input_x);
	int y_high = y_low + 1;
	int x_high = x_low + 1;

	float left = input_x - x_low;
	float top = input_y - y_low;
	float right = x_high - input_x;
	float bottom = y_high - input_y;

	float w0 = right * bottom;
	float w1 = left * bottom;
	float w2 = left * top;
	float w3 = top * right;

	uint8_t const_value_array[] = {
		const_value,const_value,const_value
	};
	uint8_t* output_pointer = output_image + (output_y * output_image_width + output_x) * 3;
	uint8_t* v0 = const_value_array;
	uint8_t* v1 = const_value_array;
	uint8_t* v2 = const_value_array;
	uint8_t* v3 = const_value_array;

	//   v0(x_low, y_low)         v1(x_high, y_low)
	//    
	//
	//   v3(x_low, y_high)        v2(x_high, y_high)
	if (x_low >= 0 && x_low < input_image_width && y_low >= 0 && y_low < input_image_height) {
		v0 = input_image + (y_low * input_image_width + x_low) * 3;
	}
	if (x_high >= 0 && x_high < input_image_width && y_low >= 0 && y_low < input_image_height) {
		v1 = input_image + (y_low * input_image_width + x_high) * 3;
	}
	if (x_high >= 0 && x_high < input_image_width && y_high >= 0 && y_high < input_image_height) {
		v2 = input_image + (y_high * input_image_width + x_high) * 3;
	}
	if (x_low >= 0 && x_low < input_image_width && y_high >= 0 && y_high < input_image_height) {
		v3 = input_image + (y_high * input_image_width + x_low) * 3;
	}

	for (int i = 0; i < 3; ++i)
		output_pointer[i] = v0[i] * w0 + v1[i] * w1 + v2[i] * w2 + v3[i] * w3;


}

cv::Mat get_affine_matrix(const Size& input, const Size& output)
{
	//�õ�����任����
	//������ͼ��ŵ����ͼ���ϣ�����Ҫ������������

	float scale_factor = std::max(output.width, output.height) / (float)std::max(input.width, input.height);
	Mat scale_matrix = (Mat_<float>(3, 3) << //
		scale_factor, 0, 0,
		0, scale_factor, 0,
		0, 0, 1);
	//ƽ�Ʊ任����
	Mat translation_matrix = (Mat_<float>(3, 3) <<
		1, 0, -input.width * 0.5 * scale_factor + output.width * 0.5,
		0, 1, -input.height * 0.5 * scale_factor + output.height * 0.5,
		0, 0, 1
		);
	Mat affine_matrix = translation_matrix * scale_matrix; //@
	affine_matrix = affine_matrix(Rect(0, 0, 3, 2)); //2x3
	return affine_matrix;
}

AffineMatrix get_gpu_affine_matrix(const Size& input, const Size& output)
{
	auto affine_matrix=get_affine_matrix(input, output);
	Mat invert_affine_matrix;
	cv::invertAffineTransform(affine_matrix, invert_affine_matrix);

	//�õ���任����
	AffineMatrix am;
	memcpy(am.value, invert_affine_matrix.ptr<float>(0),sizeof(am.value));
	return am;
}

void warpAffine()
{
	//cv
	// cv::Mat image = cv::imread("./cat1.png");
	cv::Mat image = cv::imread("/home/lai/xianjia/resources/images/fish.jpeg");
	size_t image_bytes = image.cols * image.rows * 3;
	uint8_t* image_device = nullptr;

	cv::Mat affine(640, 640, CV_8UC3);
	size_t affine_bytes = affine.rows * affine.cols * 3;
	uint8_t* affine_device = nullptr;

	clock_t t0 = clock();
	cv::Mat output;
	cv::Mat M = get_affine_matrix(image.size(),affine.size());

	cv::warpAffine(image, output, M, affine.size(), 1, 0, Scalar::all(114)); //�߽���ɫ
	clock_t t1 = clock();
	std::cout << "Cpu time: " << (t1 - t0)*1000.0/CLOCKS_PER_SEC << "ms" << std::endl;
	imwrite("output-cpu.jpg", output);//

	//GPU
	cudaStream_t stream = nullptr;
	checkRuntime(cudaStreamCreate(&stream));
	t0 = clock();
	checkRuntime(cudaMalloc(&image_device, image_bytes));
	checkRuntime(cudaMalloc(&affine_device, affine_bytes));
	checkRuntime(cudaMemcpyAsync(image_device, image.data, image_bytes, cudaMemcpyHostToDevice));
	t1 = clock();
	std::cout << "Copy time: " << (t1 - t0)*1000.0/CLOCKS_PER_SEC << "ms" << std::endl;
	t0 = clock();
	auto gpu_M = get_gpu_affine_matrix(image.size(), affine.size()); //	����� �����
	auto jobs = affine.size().area();
	int threads = jobs > num_threads ? num_threads : jobs;
	int blocks = ceil(jobs / (float)threads);
	checkKernel(warpaffine_kernel << <blocks, threads, 0, stream >> > (image_device, image.cols, image.rows,
		affine_device, affine.cols, affine.rows,
		gpu_M, 114, jobs));
	checkRuntime(cudaMemcpyAsync(affine.ptr<float>(0), affine_device, affine_bytes,cudaMemcpyDeviceToHost,stream));
	checkRuntime(cudaStreamSynchronize(stream));
	t1 = clock();
	std::cout << "Gpu time: " << (t1 - t0)*1000.0/CLOCKS_PER_SEC << "ms" << std::endl;
	cv::imwrite("output-gpu.jpg", affine);

}




