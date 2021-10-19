/* 
* Copyright (C) 2021 Simon Egger
* 
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
* 
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
* 
* You should have received a copy of the GNU General Public License
* along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

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
   static decltype(cublasSasum)* cublasSumInv;
   static decltype(cublasIsamax)* cublasMaxIndex;
   static decltype(scaleUniformArrayFloat)* scaleUniformArray;
};

template<typename T> decltype(curandGenerateUniform)* Random_Tests<T>::generateUniform = curandGenerateUniform;
template<typename T> decltype(cublasScasum)* Random_Tests<T>::cublasSum = cublasScasum;
template<typename T> decltype(cublasSasum)* Random_Tests<T>::cublasSumInv = cublasSasum;
template<typename T> decltype(cublasIsamax)* Random_Tests<T>::cublasMaxIndex = cublasIsamax;
template<typename T> decltype(scaleUniformArrayFloat)* Random_Tests<T>::scaleUniformArray = scaleUniformArrayFloat;

template<> struct Random_Tests<double> { 
   static decltype(curandGenerateUniformDouble)* generateUniform;
   static decltype(cublasDzasum)* cublasSum;
   static decltype(cublasDasum)* cublasSumInv;
   static decltype(cublasIdamax)* cublasMaxIndex;
   static decltype(scaleUniformArrayDouble)* scaleUniformArray;
};