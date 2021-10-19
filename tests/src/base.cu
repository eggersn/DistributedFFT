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

#include "tests_base.hpp"

__global__ void scaleUniformArrayFloat(cuFFT<float>::R_t* data_d, cuFFT<float>::R_t factor, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {
        data_d[i] *= factor;
    }
}

__global__ void scaleUniformArrayDouble(cuFFT<double>::R_t* data_d, cuFFT<double>::R_t factor, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {
        data_d[i] *= factor;
    }
}

decltype(curandGenerateUniformDouble)* Random_Tests<double>::generateUniform = curandGenerateUniformDouble;
decltype(cublasDzasum)* Random_Tests<double>::cublasSum = cublasDzasum;
decltype(cublasDasum)* Random_Tests<double>::cublasSumInv = cublasDasum;
decltype(cublasIdamax)* Random_Tests<double>::cublasMaxIndex = cublasIdamax;
decltype(scaleUniformArrayDouble)* Random_Tests<double>::scaleUniformArray = scaleUniformArrayDouble;