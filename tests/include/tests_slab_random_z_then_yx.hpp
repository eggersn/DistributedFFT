#pragma once

#include "tests_slab_random.hpp"

template<typename T> 
class Tests_Slab_Random_Z_Then_YX : public Tests_Slab_Random<T> {
public: 
    int run(int test_case, int opt);
protected:
    int testcase0(int opt);
    int testcase1(int opt);
    int compute(int rank, int world_size, int opt);
    int coordinate(int world_size);

    using Tests_Slab_Random<T>::Nx;
    using Tests_Slab_Random<T>::Ny;
    using Tests_Slab_Random<T>::Nz;
    using Tests_Slab_Random<T>::cuda_aware;
};