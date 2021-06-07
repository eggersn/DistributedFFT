#pragma once

#include "tests_slab_random.hpp"

template<typename T> 
class Tests_Slab_Random_Y_Then_ZX : public Tests_Slab_Random<T> {
public:
    int run(int testcase, int opt, int runs);
protected:
    int testcase0(int opt, int runs);
    int testcase1(int opt, int runs);
    int compute(int rank, int world_size, int opt, int runs);
    int coordinate(int world_size, int runs);

    using Tests_Slab_Random<T>::Nx;
    using Tests_Slab_Random<T>::Ny;
    using Tests_Slab_Random<T>::Nz;
    using Tests_Slab_Random<T>::cuda_aware;
};