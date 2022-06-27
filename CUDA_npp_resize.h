#pragma once
#include <opencv2/opencv.hpp>
#include <nppi.h>

void make_input(const cv::Mat& img, void* gpu_data_planes, int count ,int INPUT_W,int INPUT_H)
{
	void *gpu_img_buf, *gpu_img_resize_buf, *gpu_data_buf;
	int w, h, x, y;
	float r_w = INPUT_W / (img.cols*1.0);
	float r_h = INPUT_H / (img.rows*1.0);
	if (r_h > r_w) {
		w = INPUT_W;
		h = r_w * img.rows;
		x = 0;
		y = (INPUT_H - h) / 2;
	}
	else {
		w = r_h * img.cols;
		h = INPUT_H;
		x = (INPUT_W - w) / 2;
		y = 0;
	}

	int width_in = img.cols;
	int height_in = img.rows;

	uchar* img_data = img.data;

	cudaMalloc(&gpu_img_buf, width_in * height_in * 3 * sizeof(uchar));
	cudaMalloc(&gpu_img_resize_buf, INPUT_W * INPUT_H * 3 * sizeof(uchar));
	cudaMalloc(&gpu_data_buf, INPUT_W * INPUT_H * 3 * sizeof(float));

	Npp32f m_scale[3] = { 0.00392157, 0.00392157, 0.00392157 };
	//Npp32f a_scale[3] = {-1, -1, -1};
	Npp32f* r_plane = (Npp32f*)((Npp32f*)gpu_data_planes + count * 3 * INPUT_H * INPUT_W * sizeof(float));
	Npp32f* g_plane = (Npp32f*)((Npp32f*)gpu_data_planes + INPUT_W * INPUT_H * sizeof(float) + count * 3 * INPUT_H * INPUT_W * sizeof(float));
	Npp32f* b_plane = (Npp32f*)((Npp32f*)gpu_data_planes + INPUT_W * INPUT_H * 2 * sizeof(float) + count * 3 * INPUT_H * INPUT_W * sizeof(float));
	Npp32f* dst_planes[3] = { r_plane, g_plane, b_plane };
	int aDstOrder[3] = { 2, 1, 0 };


	NppiSize srcSize = { width_in, height_in };
	NppiRect srcROI = { 0, 0, width_in, height_in };
	NppiSize dstSize = { INPUT_W, INPUT_H };
	NppiRect dstROI = { x, y, w, h };

	cudaMemcpy(gpu_img_buf, img_data, width_in*height_in * 3, cudaMemcpyHostToDevice);
	nppiResize_8u_C3R((Npp8u*)gpu_img_buf, width_in * 3, srcSize, srcROI,
		(Npp8u*)gpu_img_resize_buf, INPUT_W * 3, dstSize, dstROI,
		NPPI_INTER_LINEAR);      //resize

	nppiSwapChannels_8u_C3IR((Npp8u*)gpu_img_resize_buf, INPUT_W * 3, dstSize, aDstOrder);   //rbg2bgr
	nppiConvert_8u32f_C3R((Npp8u*)gpu_img_resize_buf, INPUT_W * 3, (Npp32f*)gpu_data_buf, INPUT_W * 3 * sizeof(float), dstSize);  //ת���ɸ�����
	nppiMulC_32f_C3IR(m_scale, (Npp32f*)gpu_data_buf, INPUT_W * 3 * sizeof(float), dstSize);    //����
	//nppiAddC_32f_C3IR(a_scale, (Npp32f*)this->gpu_data_buf, this->INPUT_W*3*sizeof(float), dstSize);
	nppiCopy_32f_C3P3R((Npp32f*)gpu_data_buf, INPUT_W * 3 * sizeof(float), dst_planes, INPUT_W * sizeof(float), dstSize); //	��ͨ��32λ��������ƽ��ͼ�񸱱�

	cudaFree(gpu_img_buf);
	cudaFree(gpu_img_resize_buf);
	cudaFree(gpu_data_buf);
}



