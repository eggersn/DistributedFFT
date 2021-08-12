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