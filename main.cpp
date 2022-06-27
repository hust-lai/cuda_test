#pragma once
#include <iostream>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nppi.h>

#include "CUDA_npp_resize.h"
#include "warpAffine_cpp/xt_cuda.h"
#include "Common/UtilNPP/ImageAllocatorsNPP.h"
#include "Common/helper_cuda.h"
#include "Common/UtilNPP/ImageIO.h"
#include <npp.h>

//cuda核函数是需要声明的=_=
void hello_cuda();
void test_stream();
void warpAffine();

static void copyYUVCPU2GPU(Npp8u*pDst, uint8_t*pSrcY, uint8_t*pSrcU, uint8_t*pSrcV, int width, int height)
{
	if (pDst == nullptr || pSrcY == nullptr || pSrcU == nullptr || pSrcV == nullptr) {
		return;
	}

	uint8_t*pTemp = new uint8_t[width*height * 3];
	memcpy(pTemp, pSrcY, width*height);

	uint8_t *pTempDst = pTemp + width * height;
	uint8_t *pTempSrc = pSrcU;
	for (int i = 0; i < height / 2; i++) {
		memcpy(pTempDst, pTempSrc, width / 2);
		pTempDst += width;
		pTempSrc += width / 2;
	}

	pTempDst = pTemp + width * height * 2;
	pTempSrc = pSrcV;
	for (int i = 0; i < height / 2; i++) {
		memcpy(pTempDst, pTempSrc, width / 2);
		pTempDst += width;
		pTempSrc += width / 2;
	}

	cudaMemcpy(pDst, pTemp, width*height * 3, cudaMemcpyHostToDevice);

	delete[] pTemp;
}

int test()
{
	const char* file_yuv = "out240x128.yuv";
	int width = 240;
	int height = 128;

	size_t srcSize = width * height * 3 / 2;
	uint8_t* pInData = new uint8_t[srcSize];
	Npp8u* pYUV_dev[3]; //uchar
	// Npp8u* pYUV_dev;
	Npp8u* pRGB_dev;
	cudaMalloc((void**)&pYUV_dev, width * height * 3 * sizeof(Npp8u));
	cudaMalloc((void**)&pRGB_dev, width * height * 3 * sizeof(Npp8u));

	FILE* fp = fopen(file_yuv, "rb");
	if (!fp)
	{
		printf("open %s error!!\n", file_yuv);
		return 0;
	}

	int i = 0;
	while (fread(pInData, 1, srcSize, fp) == srcSize) //YUV 422
	{
		uint8_t* pY = pInData;
		uint8_t* pU = pY + width * height;
		uint8_t* pV = pU + width * height / 4;

		copyYUVCPU2GPU(pYUV_dev[0], pY, pU, pV, width, height);
		NppiSize nppSize = { width,height };
		printf("[%s:%d],nppSize(%d,%d)\n", __FILE__, __LINE__, nppSize.width, nppSize.height);

		/*YUV格式有两大类：planar和packed。
		对于planar的YUV格式，先连续存储所有像素点的Y，紧接着存储所有像素点的U，随后是所有像素点的V。
		对于packed的YUV格式，每个像素点的Y, U, V是连续交*存储的。
		YUV420P，Y，U，V三个分量都是平面格式，分为I420和YV12。I420格式和YV12格式的不同处在U平面和V平面的位置不同。
		在I420格式中，U平面紧跟在Y平面之后，然后才是V平面（即：YUV）；但YV12则是相反（即：YVU）
		YUV420SP, Y分量平面格式，UV打包格式, 即NV12。 NV12与NV21类似，U 和 V 交错排列,不同在于UV顺序。
		*/

		int width3 = width * 3;
		// auto ret = nppiYUVToRGB_8u_P3R((const Npp8u * const*)pYUV_dev, width * 3, &pRGB_dev,width*3, nppSize);
		auto ret = nppiYUV420ToRGB_8u_P3R(pYUV_dev, &width3, &pRGB_dev, width * 3, nppSize);
		if (ret != 0)
		{
			printf("nppiYUVToRGB_8u_C3R error:%d\n", ret);
			return 0;
		}

		cv::Mat img(height, width, CV_8UC3);
		cudaMemcpy(img.data, pRGB_dev, width*height * 3, cudaMemcpyDeviceToHost);
		std::string str1 = std::to_string(i + 1) + ".jpg";
		cv::imwrite(str1.c_str(), img);
		//cv::waitKey(1);
		i++;
		if (i > 5)
		{
			break;
		}
	}
	delete[] pInData;
	cudaFree(pYUV_dev);
	cudaFree(pRGB_dev);
	fclose(fp);
}

bool printfNPPinfo(int argc, char *argv[]) {
  const NppLibraryVersion *libVer = nppGetLibVersion();

  printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor,
         libVer->build);

  int driverVersion, runtimeVersion;
  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);

  printf("  CUDA Driver  Version: %d.%d\n", driverVersion / 1000,
         (driverVersion % 100) / 10);
  printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000,
         (runtimeVersion % 100) / 10);

  // Min spec is SM 1.0 devices
  bool bVal = checkCudaCapabilities(1, 0);
  return bVal;
}

int nppi_boxfilter_test(int argc, char **argv)
{
	try {
		std::string sFilename;
		char *filePath;

		findCudaDevice(argc, (const char **)argv);

		if (printfNPPinfo(argc, argv) == false) {
			exit(EXIT_SUCCESS);
		}

		if (checkCmdLineFlag(argc, (const char **)argv, "input")) {
			getCmdLineArgumentString(argc, (const char **)argv, "input", &filePath);
		} else {
			filePath = sdkFindFilePath("teapot512.pgm", argv[0]);
		}

		if (filePath) {
			sFilename = filePath;
		} else {
			sFilename = "teapot512.pgm";
		}

		// if we specify the filename at the command line, then we only test
		// sFilename[0].
		int file_errors = 0;
		std::ifstream infile(sFilename.data(), std::ifstream::in);

		if (infile.good()) {
			std::cout << "boxFilterNPP opened: <" << sFilename.data()
					<< "> successfully!" << std::endl;
			file_errors = 0;
			infile.close();
		} else {
			std::cout << "boxFilterNPP unable to open: <" << sFilename.data() << ">"
					<< std::endl;
			file_errors++;
			infile.close();
		}

		if (file_errors > 0) {
			exit(EXIT_FAILURE);
		}

		std::string sResultFilename = sFilename;

		std::string::size_type dot = sResultFilename.rfind('.');

		if (dot != std::string::npos) {
			sResultFilename = sResultFilename.substr(0, dot);
		}

		sResultFilename += "_boxFilter.pgm";

		if (checkCmdLineFlag(argc, (const char **)argv, "output")) {
			char *outputFilePath;
			getCmdLineArgumentString(argc, (const char **)argv, "output",
								&outputFilePath);
			sResultFilename = outputFilePath;
		}

		// declare a host image object for an 8-bit grayscale image
		npp::ImageCPU_8u_C1 oHostSrc;
		// load gray-scale image from disk
		npp::loadImage(sFilename, oHostSrc);
		// declare a device image and copy construct from the host image,
		// i.e. upload host to device
		npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);

		// create struct with box-filter mask size
		NppiSize oMaskSize = {5, 5};

		NppiSize oSrcSize = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
		NppiPoint oSrcOffset = {0, 0};

		// create struct with ROI size
		NppiSize oSizeROI = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
		// allocate device image of appropriately reduced size
		npp::ImageNPP_8u_C1 oDeviceDst(oSizeROI.width, oSizeROI.height);
		// set anchor point inside the mask to (oMaskSize.width / 2,
		// oMaskSize.height / 2) It should round down when odd
		NppiPoint oAnchor = {oMaskSize.width / 2, oMaskSize.height / 2};

		// run box filter
		NPP_CHECK_NPP(nppiFilterBoxBorder_8u_C1R(
			oDeviceSrc.data(), oDeviceSrc.pitch(), oSrcSize, oSrcOffset,
			oDeviceDst.data(), oDeviceDst.pitch(), oSizeROI, oMaskSize, oAnchor,
			NPP_BORDER_REPLICATE));

		// declare a host image for the result
		npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
		// and copy the device result data into it
		oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());

		saveImage(sResultFilename, oHostDst);
		std::cout << "Saved image: " << sResultFilename << std::endl;

		nppiFree(oDeviceSrc.data());
		nppiFree(oDeviceDst.data());

		exit(EXIT_SUCCESS);
	} catch (npp::Exception &rException) {
		std::cerr << "Program error! The following exception occurred: \n";
		std::cerr << rException << std::endl;
		std::cerr << "Aborting." << std::endl;

		exit(EXIT_FAILURE);
	} catch (...) {
		std::cerr << "Program error! An unknow type of exception occurred. \n";
		std::cerr << "Aborting." << std::endl;

		exit(EXIT_FAILURE);
		return -1;
	}
}


int main()
{
	//printf("1");
	//cv::Mat image = cv::imread("cat1.png");
	//printf("image size = %d x %d\n", image.cols, image.rows);
	//cv::imshow("cat", image);
	//cv::waitKey(3000);

	//void* ptr = nullptr;
	//checkRuntime(cudaMalloc(&ptr, 32));
	//printf("ptr = %d\n", ptr); // 77594624
	//cudaFree(&ptr);

	//hello_cuda();
	//test_stream();
	// warpAffine();

	test();

	return 0;
}

