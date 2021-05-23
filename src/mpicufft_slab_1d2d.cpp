#include "mpicufft_slab_1d2d.hpp"
#include "cufft.hpp"
#include <cuda_runtime_api.h>


#define CUDA_CALL(x) do { if((x)!=cudaSuccess) {        \
    printf("Error %d at %s:%d\n",x,__FILE__,__LINE__);  \
    exit(EXIT_FAILURE); }} while(0)

#define CUFFT_CALL(x) do { if((x)!=CUFFT_SUCCESS) {     \
    printf("Error %d at %s:%d\n",x,__FILE__,__LINE__);  \
    exit(EXIT_FAILURE); }} while(0)

#define DEBUG 1
#define debug(d, v) {                                                 \
  if (DEBUG == 1) {                                                   \
    printf("DEBUG (%d): %s: %s in %s:%d\n",pidx,d,v,__FILE__,__LINE__); \
  }                                                                   \
}

#define debug_h(v) {                                                  \
  if (DEBUG == 1) {                                                   \
    printf("%s",v);                \
  }                                                                   \
}

#define debug_int(d, v) {                                             \
  if (DEBUG == 1) {                                                   \
    printf("DEBUG (%d): %s: %d in %s:%d\n",pidx,d,v,__FILE__,__LINE__); \
  }                                                                   \
}

#define debug_p(d, v) {                                                  \
  if (DEBUG == 1) {                                                   \
    printf("DEBUG (%d): %s: %p in %s:%d\n",pidx,d,v,__FILE__,__LINE__); \
  }                                                                   \
}

template<typename T> 
MPIcuFFT_Slab_1D2D<T>::MPIcuFFT_Slab_1D2D(MPI_Comm comm, bool mpi_cuda_aware, int max_world_size) : MPIcuFFT<T>(comm, mpi_cuda_aware, max_world_size) {
    input_sizes_x.resize(pcnt, 0);
    istartx.resize(pcnt, 0);
    output_sizes_y.resize(pcnt, 0);
    ostarty.resize(pcnt, 0);
    output_sizes_z.resize(pcnt, 0);
    ostartz.resize(pcnt, 0);

    send_req.resize(pcnt, MPI_REQUEST_NULL);
    recv_req.resize(pcnt, MPI_REQUEST_NULL);

    input_size_z = 0;
    output_size_z = 0;

    planC2C = 0;

    for (int i = 1; i < pcnt; i++)
        comm_order.push_back((pidx + i) % pcnt);

    for (int i = 0; i < pcnt; i++){
        cudaStream_t stream;
        CUDA_CALL(cudaStreamCreate(&stream));
        streams.push_back(stream);
    }
}

template<typename T> 
MPIcuFFT_Slab_1D2D<T>::~MPIcuFFT_Slab_1D2D() {
    if (!planR2C.empty()) 
        for (auto plan : planR2C)
            CUFFT_CALL(cufftDestroy(plan));
    if (planC2C) 
        CUFFT_CALL(cufftDestroy(planC2C));
}

template<typename T> 
void MPIcuFFT_Slab_1D2D<T>::initFFT(GlobalSize *global_size, bool allocate, sequence_e _sequence) {
    using R_t = typename cuFFT<T>::R_t;
    using C_t = typename cuFFT<T>::C_t;

    sequence = _sequence;

    // input_sizes_x stores how the input 3d array is distributed among the mpi processes
    size_t N1    = global_size->Nx / pcnt;
    size_t N1mod = global_size->Nx % pcnt;
    for (int p = 0; p < pcnt; ++p) {
        input_sizes_x[p]  = N1 + ((static_cast<size_t>(p) < N1mod) ? 1 : 0);
        istartx[p] = ((p==0) ? 0 : istartx[p-1]+input_sizes_x[p-1]);
    }

    // We only divide across the x-axis
    input_size_y = global_size->Ny; input_size_z = global_size->Nz;

    if (sequence == Z_Then_YX) {
        // After transposing the array, it is divided across the z-axis
        size_t N2 = (global_size->Nz/2+1) / pcnt;
        size_t N2mod = (global_size->Nz/2+1) % pcnt;

        for (int p = 0; p < pcnt; p++){
            output_sizes_z[p] = N2 + ((static_cast<size_t>(p) < N2mod) ? 1 : 0);
            ostartz[p] = ((p==0) ? 0 : ostartz[p-1]+output_sizes_z[p-1]);
        }
        // For real input values, the second half (of the z-axis) is symmetric (complex conjugate) to the first half
        output_size_x = global_size->Nx; output_size_y = global_size->Ny;

        domainsize = sizeof(C_t) * std::max(input_sizes_x[pidx]*input_size_y*(input_size_z/2 + 1), output_size_x*output_size_y*output_sizes_z[pidx]);
    } else {
        // After transposing the array, it is divided across the y-axis
        size_t N2    = (global_size->Ny/2+1) / pcnt;
        size_t N2mod = (global_size->Ny/2+1) % pcnt;
        for (int p = 0; p < pcnt; ++p) {
            output_sizes_y[p]  = N2 + ((static_cast<size_t>(p) < N2mod) ? 1 : 0);
            ostarty[p] = ((p==0) ? 0 : ostarty[p-1]+output_sizes_y[p-1]);
        }
        // For real input values, the second half (of the y-axis) is symmetric (complex conjugate) to the first half
        output_size_x = global_size->Nx; output_size_z = global_size->Nz;

        domainsize = sizeof(C_t) * std::max(input_sizes_x[pidx]*input_size_y*((input_size_z/2) + 1), output_size_x*output_sizes_y[pidx]*output_size_z);
    }

    // Sizes of the different workspaces
    size_t ws_c2c;
        
    if (fft3d) { // Combined 3d fft, in case only one mpi process is used
        cufftHandle handle;

        CUFFT_CALL(cufftCreate(&handle));
        CUFFT_CALL(cufftSetAutoAllocation(handle, 0));
        planR2C.push_back(handle);

        CUFFT_CALL(cufftMakePlan3d(handle, global_size->Nx, global_size->Ny, global_size->Nz, cuFFT<T>::R2Ctype, &ws_r2c));

        fft_worksize = ws_r2c;
    } else if (sequence == Z_Then_YX) { 
        num_of_streams = 1;
        size_t batch = input_size_y * input_sizes_x[pidx];

        cufftHandle handle;

        CUFFT_CALL(cufftCreate(&handle));
        CUFFT_CALL(cufftSetAutoAllocation(handle, 0));
        planR2C.push_back(handle);

        CUFFT_CALL(cufftCreate(&planC2C));
        CUFFT_CALL(cufftSetAutoAllocation(planC2C, 0));

        long long n[3] = {static_cast<long long>(output_size_x), static_cast<long long>(output_size_y), static_cast<long long>(input_size_z)};

        // For the 1D R2C FFT, the default data layer can be used (in case sequence = Z_Then_YX)
        CUFFT_CALL(cufftMakePlanMany64(planR2C[0], 1, &n[2], 0, 0, 0, 0, 0, 0, cuFFT<T>::R2Ctype, batch, &ws_r2c));

        batch = output_sizes_z[pidx];
        long long nembed[1] = {1, static_cast<long long>(output_size_y)};

        CUFFT_CALL(cufftMakePlanMany64(planC2C, 2, &n[0], nembed, output_sizes_z[pidx], 1, nembed, 
            output_sizes_z[pidx], 1, cuFFT<T>::R2Ctype, batch, &ws_c2c));

        fft_worksize = std::max(ws_r2c, ws_c2c);
    } else { // sequence == Y_Then_ZX
        CUFFT_CALL(cufftCreate(&planC2C));
        CUFFT_CALL(cufftSetAutoAllocation(planC2C, 0));

        num_of_streams = std::min(input_sizes_x[pidx], input_size_z);
        size_t batch = std::max(input_sizes_x[pidx], input_size_z);

        long long n[3] = {static_cast<long long>(output_size_y), static_cast<long long>(output_size_x), static_cast<long long>(input_size_z)};  
        long long inembed[3] = {input_sizes_x[pidx] <= input_size_z ? 1 : static_cast<long long>(input_size_z * input_size_y),  
            input_size_z, input_size_z * output_sizes_y[pidx]};
        long long onembed[3] = {input_sizes_x[pidx] <= input_size_z ? 1 : static_cast<long long>(input_size_z * (input_size_y/2+1)),
            input_size_z, input_size_z * output_sizes_y[pidx]};

        for (int s = 0; s < num_of_streams; s++){
            if (s >= pcnt) {
                cufftHandle handle;
                cudaStream_t stream;
                CUDA_CALL(cudaStreamCreate(&stream));
                streams.push_back(stream);
            }

            CUFFT_CALL(cufftCreate(&handle));
            CUFFT_CALL(cufftSetAutoAllocation(handle, 0));

            CUFFT_CALL(cufftMakePlanMany64(handle, 1, &n[0], &inembed[0], input_size_z, inembed[0], &onembed[0], input_size_z, onembed[0], cuFFT<T>::R2Ctype, batch, &ws_r2c));
            CUFFT_CALL(cufftSetStream(handle, streams[s]));

            planR2C.push_back(handle);
        }

        batch = output_sizes_y[pidx];
        CUFFT_CALL(cufftMakePlanMany64(planC2C, 2, &n[1], &inembed[1], 1, inembed[1], &onembed[1], 1, onembed[1], cuFFT<T>::C2Ctype, batch, &ws_c2c));

        fft_worksize = std::max(num_of_streams*ws_r2c, ws_c2c);
    }

    if (fft_worksize < domainsize)
        fft_worksize = domainsize;

    // worksize_d is split into 3 parts:
    // 1. space for received data, 2. space for transmitted data, 3. actual workspace (see "mem_d")
    worksize_d = fft_worksize + (fft3d ? 0 : 2*domainsize);
    // analogously for the host worksize, if mpi is not cuda-aware
    worksize_h = (cuda_aware || fft3d ? 0 : 2*domainsize);

    if (allocate) 
        this->setWorkArea();
    
    CUDA_CALL(cudaDeviceSynchronize());
}

//default parameters device=nullptr, host=nullptr
template<typename T> 
void MPIcuFFT_Slab_1D2D<T>::setWorkArea(void *device, void *host) {
    if (!domainsize) 
        return;

    if (device && allocated_d) {
        CUDA_CALL(cudaFree(workarea_d));
        allocated_d = false;
        workarea_d = device;
    } else if (!allocated_d && device) {
        workarea_d = device;
    } else if (!allocated_d && !device) {
        CUDA_CALL(cudaMalloc(&(workarea_d), worksize_d));
        allocated_d = true;
    }

    mem_d.clear();
    for (size_t i=0; i< 1 + (fft3d ? 0 : (cuda_aware ? 2 : 1)); ++i) 
        mem_d.push_back(&static_cast<char*>(workarea_d)[i*domainsize]);
    
    if (fft3d) {
        CUFFT_CALL(cufftSetWorkArea(planR2C[0], mem_d[0]));
    } else if (sequence == Z_Then_YX) {
        CUFFT_CALL(cufftSetWorkArea(planR2C[0], mem_d[cuda_aware ? 2 : 1]));
        CUFFT_CALL(cufftSetWorkArea(planC2C, mem_d[cuda_aware ? 2 : 1]));
    } else {
        // The 1D FFT (y-direction) is split into multiple streams,
        // therefore each one has its own workspace
        size_t offset = cuda_aware ? 2 : 1;
        for (size_t s = 1; s < num_of_streams; s++)
            mem_d.push_back(&static_cast<char*>(workarea_d)[offset*domainsize+s*ws_r2c]);

        for (int i = 0; i < planR2C.size(); i++) 
            CUFFT_CALL(cufftSetWorkArea(planR2C[i], mem_d[offset+i]));
        CUFFT_CALL(cufftSetWorkArea(planC2C, mem_d[cuda_aware ? 2 : 1]));
    }
        
    if (host && allocated_h) {
        CUDA_CALL(cudaFreeHost(workarea_h));
        allocated_h = false;
        workarea_h = host;
    } else if (!allocated_h && host) {
        workarea_h = host;
    } else if (!allocated_h && !host && worksize_h) {
        CUDA_CALL(cudaMallocHost(&(workarea_h), worksize_h));
        allocated_h = true;
    }

    mem_h.clear();
    for (size_t i=0; i<((fft3d || cuda_aware) ? 0 : 2); ++i)
        mem_h.push_back(&static_cast<char*>(workarea_h)[i*domainsize]);

    initialized = true;
}

template<typename T>
void MPIcuFFT_Slab_1D2D<T>::execR2C(void *out, const void *in) {
    if (!initialized) 
        return;

    using R_t = typename cuFFT<T>::R_t;
    using C_t = typename cuFFT<T>::C_t;
    R_t *real    = cuFFT<T>::real(in);
    C_t *complex = cuFFT<T>::complex(out);
    if (fft3d) {
        CUFFT_CALL(cuFFT<T>::execR2C(planR2C, real, complex));
        CUDA_CALL(cudaDeviceSynchronize());
    } else {
        size_t input_batch_offset = 1;
        size_t output_batch_offset = 1;
        if (input_size_z < input_sizes_x[pidx]) {
            input_batch_offset = input_size_z * input_size_y;
            output_batch_offset = input_size_z * (input_size_y/2+1);
        }

        // In case sequence == Z_Then_YX, then num_of_streams == 1.
        for (int s = 0; s < num_of_streams; s++) {
            // compute 2d FFT 
            CUFFT_CALL(cuFFT<T>::execR2C(planR2C[s], real[s*input_batch_offset], complex[s*output_batch_offset]));
        }

        /* ***********************************************************************************************************************
        *                                                       Global Transpose
        *  *********************************************************************************************************************** */

        C_t *recv_ptr, *send_ptr, *temp_ptr;
        temp_ptr = cuFFT<T>::complex(mem_d[0]);
        if (cuda_aware) {
            recv_ptr = cuFFT<T>::complex(mem_d[0]); // = temp_ptr!
            send_ptr = cuFFT<T>::complex(mem_d[1]);
        } else {
            recv_ptr = cuFFT<T>::complex(mem_h[0]);
            send_ptr = cuFFT<T>::complex(mem_h[1]);
        }
        recv_req[pidx] = MPI_REQUEST_NULL;
        send_req[pidx] = MPI_REQUEST_NULL;

        CUDA_CALL(cudaDeviceSynchronize());

        if (sequence == Z_Then_YX){
            for (auto p : comm_order) {
                MPI_Irecv(&recv_ptr[output_sizes_z[pidx]*output_size_y*istart[p]], 
                    sizeof(C_t)*output_sizes_z[pidx]*output_size_y*input_sizes_x[p], MPI_BYTE, p, p, comm, &rev_req[p]);

                size_t oslice = ostartz[p]*input_size_y*input_sizes_x[pidx];

                CUDA_CALL(cudaMemcpy2DAsync(&send_ptr[oslice], sizeof(C_t)*output_sizes_z[p],
                    &complex[ostartz[p]], sizeof(C_t)*(input_size_z/2+1), sizeof(C_t)*output_sizes_z[p], input_size_y*input_sizes_x[pidx],
                    cuda_aware?cudaMemcpyDeviceToDevice:cudaMemcpyDeviceToHost, streams[p]));

                CUDA_CALL(cudaDeviceSynchronize());

                MPI_Isend(&send_ptr[oslice], sizeof(C_t)*output_sizes_z[p]*input_size_y*input_sizes_x[pidx], 
                    MPI_BYTE, p, pidx, comm, &send_req[p]);
            }
        } else {
            for (auto p : comm_order) { 
                // start non-blocking receive for rank p
                MPI_Irecv(&recv_ptr[output_size_z*output_sizes_y[pidx]*istartx[p]],
                    sizeof(C_t)*output_size_z*output_sizes_y[pidx]*input_sizes_x[p], MPI_BYTE,
                    p, p, comm, &recv_req[p]);

                size_t oslice = output_size_z*ostarty[p]*input_sizes_x[pidx];
            
                CUDA_CALL(cudaMemcpy2DAsync(&send_ptr[oslice], sizeof(C_t)*output_size_z*output_sizes_y[p],
                    &complex[output_size_z*ostarty[p]], sizeof(C_t)*output_size_z*input_size_y, sizeof(C_t)*output_size_z*output_sizes_y[p], input_sizes_x[pidx], 
                    cuda_aware?cudaMemcpyDeviceToDevice:cudaMemcpyDeviceToHost, streams[p]));

                CUDA_CALL(cudaDeviceSynchronize());

                MPI_Isend(&send_ptr[oslice], 
                    sizeof(C_t)*output_size_z*output_sizes_y[p]*input_sizes_x[pidx], 
                    MPI_BYTE, p, pidx, comm, &send_req[p]);
            }
        }

        // Next up: Local Transpose
    }
}