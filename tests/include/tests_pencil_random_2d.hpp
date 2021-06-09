#pragma once

#include "tests_pencil_random.hpp"

template<typename T> 
class Tests_Pencil_Random_2D : public Tests_Pencil_Random<T> {
public:
    int run(const int testcase, const int opt, const int runs);
protected:
    int testcase0(const int opt, const int runs);
    int testcase1(const int opt, const int runs);
    int compute(const int rank, const int world_size, const int opt, const int runs);
    int coordinate(const int world_size, const int runs);

    using Tests_Pencil_Random<T>::Nx;
    using Tests_Pencil_Random<T>::Ny;
    using Tests_Pencil_Random<T>::Nz;
    using Tests_Pencil_Random<T>::P1;
    using Tests_Pencil_Random<T>::P2;
    using Tests_Pencil_Random<T>::cuda_aware;
};