#pragma once

#include <cufft.h>
#include <cuda.h>

template<typename T> struct cuFFT {
  using C_t = cufftComplex;
  using R_t = cufftReal;
  
  static const cufftType R2Ctype = CUFFT_R2C;
  static const cufftType C2Rtype = CUFFT_C2R;
  static const cufftType C2Ctype = CUFFT_C2C;
  
  static decltype(cufftExecR2C)* execR2C;
  static decltype(cufftExecC2R)* execC2R;
  static decltype(cufftExecC2C)* execC2C;
  
  static inline C_t* complex(void *ptr) { return static_cast<C_t*>(ptr); };
  static inline C_t* complex(const void *ptr) { return static_cast<C_t*>(const_cast<void*>(ptr)); };
  static inline R_t* real(void *ptr) { return static_cast<R_t*>(ptr); };
  static inline R_t* real(const void *ptr) { return static_cast<R_t*>(const_cast<void*>(ptr)); };
};

template<typename T> decltype(cufftExecR2C)* cuFFT<T>::execR2C = cufftExecR2C;
template<typename T> decltype(cufftExecC2R)* cuFFT<T>::execC2R = cufftExecC2R;
template<typename T> decltype(cufftExecC2C)* cuFFT<T>::execC2C = cufftExecC2C;

template<> struct cuFFT<double> {
  using C_t = cufftDoubleComplex;
  using R_t = cufftDoubleReal;
  
  static const cufftType R2Ctype = CUFFT_D2Z;
  static const cufftType C2Rtype = CUFFT_Z2D;
  static const cufftType C2Ctype = CUFFT_Z2Z;
  
  static decltype(cufftExecD2Z)* execR2C;
  static decltype(cufftExecZ2D)* execC2R;
  static decltype(cufftExecZ2Z)* execC2C;
  
  static inline C_t* complex(void *ptr) { return static_cast<C_t*>(ptr); };
  static inline C_t* complex(const void *ptr) { return static_cast<C_t*>(const_cast<void*>(ptr)); };
  static inline R_t* real(void *ptr) { return static_cast<R_t*>(ptr); };
  static inline R_t* real(const void *ptr) { return static_cast<R_t*>(const_cast<void*>(ptr)); };
};

// Initialization in mpicufft.cpp
// decltype(cufftExecD2Z)* cuFFT<double>::execR2C = cufftExecD2Z;
// decltype(cufftExecZ2D)* cuFFT<double>::execC2R = cufftExecZ2D;
// decltype(cufftExecZ2Z)* cuFFT<double>::execC2C = cufftExecZ2Z;
