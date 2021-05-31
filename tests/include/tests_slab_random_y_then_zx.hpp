#pragma once

#include "tests_slab_random.hpp"

template<typename T> 
class Tests_Slab_Random_Y_Then_ZX : public Tests_Slab_Random<T> {
protected:
     int compute(int rank, int world_size, int opt);
     int coordinate(int world_size);

    using Tests_Slab_Random<T>::Nx;
    using Tests_Slab_Random<T>::Ny;
    using Tests_Slab_Random<T>::Nz;
    using Tests_Slab_Random<T>::cuda_aware;
};