#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include "cublas_v2.h"
#include "device_launch_parameters.h"
#include "cufft.hpp"

extern __global__ void scaleUniformArrayFloat(cuFFT<float>::R_t* data_d, cuFFT<float>::R_t factor, int n);
extern __global__ void scaleUniformArrayDouble(cuFFT<double>::R_t* data_d, cuFFT<double>::R_t factor, int n);

template<typename T> 
struct Random_Tests { 
   static decltype(curandGenerateUniform)* generateUniform;
   static decltype(cublasScasum)* cublasSum;
   static decltype(scaleUniformArrayFloat)* scaleUniformArray;
};

template<typename T> decltype(curandGenerateUniform)* Random_Tests<T>::generateUniform = curandGenerateUniform;
template<typename T> decltype(cublasScasum)* Random_Tests<T>::cublasSum = cublasScasum;
template<typename T> decltype(scaleUniformArrayFloat)* Random_Tests<T>::scaleUniformArray = scaleUniformArrayFloat;

template<> struct Random_Tests<double> { 
   static decltype(curandGenerateUniformDouble)* generateUniform;
   static decltype(cublasDzasum)* cublasSum;
   static decltype(scaleUniformArrayDouble)* scaleUniformArray;
};