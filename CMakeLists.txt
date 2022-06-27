cmake_minimum_required(VERSION 3.0)
project(CudaTest LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_NVCC_FLAGS "CMAKE_NVCC_FLAGS -std=c++11")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmakes")

find_package(PkgConfig)
find_package(CUDA REQUIRED)
find_package(OpenCV 4.5.5 REQUIRED)
# find_package(FreeImage REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${FreeImage_INCLUDE_DIRS})
include_directories(warpAffine_cpp/)
include_directories(Common/)

ADD_EXECUTABLE(cuda_test
            main.cpp
            warpAffine_cpp/xt_cuda.cpp
            kernel.cu
            warpAffine_cpp/hello_cuda.cu
            warpAffine_cpp/WarpAffine.cu
        )

target_link_libraries(cuda_test
    # /usr/local/cuda-11.6/targets/x86_64-linux/lib/libcudart.so
    # /usr/local/cuda-11.6/targets/x86_64-linux/lib/libnppicc.so
    # -L/usr/local/cuda-11.6/lib64 -lnppig
    ${CUDA_cudart_LIBRARY}
    ${CUDA_cuda_LIBRARY}
    ${CUDA_npp_LIBRARY}
    ${OpenCV_LIBS}
    /usr/lib/x86_64-linux-gnu/libfreeimage.so
    # ${FREEIMAGE_LIBRARIES}
)