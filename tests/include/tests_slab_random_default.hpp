#pragma once

#include "tests_slab_random.hpp"

template<typename T> 
class Tests_Slab_Random_Default : public Tests_Slab_Random<T> {
public:
    int run(const int testcase, const int opt, const int runs);
protected:
    int testcase0(const int opt, const int runs);
    int testcase1(const int opt, const int runs);
    int testcase2(const int opt, const int runs);
    int compute(const int rank, const int world_size, const int opt, const int runs);
    int coordinate(const int world_size, const int runs);

    using Tests_Slab_Random<T>::Nx;
    using Tests_Slab_Random<T>::Ny;
    using Tests_Slab_Random<T>::Nz;
    using Tests_Slab_Random<T>::config;
};