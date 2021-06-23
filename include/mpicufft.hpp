#pragma once

#include "params.hpp"
#include <iostream>
#include <cstddef>
#include <vector>
#include <mpi.h>

/*! \mainpage Main Page

This library implements different methods for distributed FFT computation on heterogeneous GPU Systems. \n
In general, we assume that the input data is aligned as [z][y][x] (such that the data is continuous in z-direction). The various implementations can be grouped into:

- \subpage MPIcuFFT_Slab \n
    Here the global input data \f$N_x \times N_y \times N_z\f$ is split in x-direction. Therefore, we assume that each process starts with 
    input data of size \f$\frac{N_x}{P} \times N_y \times N_z\f$. \n
    The default procedure for slab decomposition is the following:
    1. Input: \f$\frac{N_x}{P} \times N_y \times N_z\f$
    2. Compute the 2D-FFT in y- and z-direction: \f$\frac{N_x}{P} \times \hat{N_y} \times \hat{N_z}\f$
    3. Redistribute: Each node has \f$N_x \times \frac{\hat{N_y}}{P} \times \hat{N_z}\f$
    4. Compute the remaining 1D-FFT in x-direction: \f$\hat{N_x} \times \frac{\hat{N_y}}{P} \times \hat{N_z}\f$
- \subpage Pencil_Decomposition \n
    Here the global input data \f$N_x \times N_y \times N_z\f$ is split in x- and y-direction. Therefore, we assume that each process starts with 
    input data of size \f$\frac{N_x}{P1} \times \frac{N_y}{P2} \times N_z\f$. \n
    The default procedure for pencil decomposition is the following:
    1. Input: \f$\frac{N_x}{P1} \times \frac{N_y}{P2} \times N_z\f$
    2. Compute the 1D-FFT in z-direction: \f$\frac{N_x}{P1} \times \frac{N_y}{P2} \times \hat{N_z}\f$
    3. Redistribute: Each node has \f$\frac{N_x}{P1} \times N_y \times \frac{\hat{N_z}}{P2}\f$
    4. Compute the 1D-FFT in y-direction: \f$\frac{N_x}{P1} \times \hat{N_y} \times \frac{\hat{N_z}}{P2}\f$
    5. Redistribute: Each node has \f$N_x \times \frac{\hat{N_y}}{P1} \times \frac{\hat{N_z}}{P2}\f$
    6. Compute the 1D-FFT in x-direction: \f$\hat{N_x} \times \frac{\hat{N_y}}{P1} \times \frac{\hat{N_z}}{P2}\f$

All methods implement \subpage MPIcuFFT.
*/

template<typename T> class MPIcuFFT {
public:
    MPIcuFFT (MPI_Comm comm=MPI_COMM_WORLD, bool mpi_cuda_aware=false, int max_world_size=-1);
    ~MPIcuFFT ();

    virtual void initFFT(GlobalSize *global_size, Partition *partition, bool allocate=true)= 0;
    virtual void setWorkArea(void *device=nullptr, void *host=nullptr) = 0;

    virtual void execR2C(void *out, const void *in) = 0;

    virtual inline void getInSize(size_t *isize) = 0;
    virtual inline void getInStart(size_t *istart) = 0;
    virtual inline void getOutSize(size_t *osize) = 0;
    virtual inline void getOutStart(size_t *ostart) = 0;

    inline size_t getDomainSize() const { return domainsize; };
    inline size_t getWorkSizeDevice() const { return worksize_d; };
    inline size_t getWorkSizeHost() const { return worksize_h; };

    inline void* getWorkAreaDevice() const { return workarea_d; };
    inline void* getWorkAreaHost() const { return workarea_h; };

    inline int getRank() const { return pidx; };
    inline int getWorldSize() const { return pcnt; };

  
protected:  
    enum commMode_e {Peer, All2All} comm_mode;
    MPI_Comm comm;

    int pidx, pcnt;

    std::vector<int> comm_order;

    size_t domainsize;
    size_t fft_worksize;
    
    size_t worksize_d;
    size_t worksize_h;
    
    void* workarea_d;
    void* workarea_h;

    std::vector<void*> mem_d;
    std::vector<void*> mem_h;

    bool allocated_d, allocated_h;
    bool cuda_aware;
    bool initialized;
    bool fft3d;
};
