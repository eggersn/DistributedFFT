#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include "cublas_v2.h"
#include "device_launch_parameters.h"
#include "cufft.hpp"

extern __global__ void scaleUniformArrayFloat(cuFFT<float>::R_t* data_d, cuFFT<float>::R_t factor, int n);
extern __global__ void scaleUniformArrayDouble(cuFFT<double>::R_t* data_d, cuFFT<double>::R_t factor, int n);
extern __global__ void scaleUniformComplexArrayFloat(cuFFT<float>::C_t* data_d, float factor, int n);
extern __global__ void scaleUniformComplexArrayDouble(cuFFT<double>::C_t* data_d, double factor, int n);

template<typename T> 
struct Random_Tests { 
   static decltype(curandGenerateUniform)* generateUniform;
   static decltype(cublasScasum)* cublasSum;
   static decltype(cublasSasum)* cublasSumInv;
   static decltype(cublasIsamax)* cublasMaxIndex;
   static decltype(scaleUniformArrayFloat)* scaleUniformArray;
   static decltype(scaleUniformComplexArrayFloat)* scaleUniformComplexArray;
};

template<typename T> decltype(curandGenerateUniform)* Random_Tests<T>::generateUniform = curandGenerateUniform;
template<typename T> decltype(cublasScasum)* Random_Tests<T>::cublasSum = cublasScasum;
template<typename T> decltype(cublasSasum)* Random_Tests<T>::cublasSumInv = cublasSasum;
template<typename T> decltype(cublasIsamax)* Random_Tests<T>::cublasMaxIndex = cublasIsamax;
template<typename T> decltype(scaleUniformArrayFloat)* Random_Tests<T>::scaleUniformArray = scaleUniformArrayFloat;
template<typename T> decltype(scaleUniformComplexArrayFloat)* Random_Tests<T>::scaleUniformComplexArray = scaleUniformComplexArrayFloat;

template<> struct Random_Tests<double> { 
   static decltype(curandGenerateUniformDouble)* generateUniform;
   static decltype(cublasDzasum)* cublasSum;
   static decltype(cublasDasum)* cublasSumInv;
   static decltype(cublasIdamax)* cublasMaxIndex;
   static decltype(scaleUniformArrayDouble)* scaleUniformArray;
   static decltype(scaleUniformComplexArrayDouble)* scaleUniformComplexArray;
};