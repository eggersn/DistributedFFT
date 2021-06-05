#include "mpicufft_slab_z_then_yx_opt1.hpp"
#include "cufft.hpp"
#include <cuda_runtime_api.h>


#define CUDA_CALL(x) do { if((x)!=cudaSuccess) {        \
    printf("Error %d at %s:%d\n",x,__FILE__,__LINE__);  \
    exit(EXIT_FAILURE); }} while(0)

#define CUFFT_CALL(x) do { if((x)!=CUFFT_SUCCESS) {     \
    printf("Error %d at %s:%d\n",x,__FILE__,__LINE__);  \
    exit(EXIT_FAILURE); }} while(0)


template<typename T> 
void MPIcuFFT_Slab_Z_Then_YX_Opt1<T>::initFFT(GlobalSize *global_size, bool allocate) {
    timer->start();
    using R_t = typename cuFFT<T>::R_t;
    using C_t = typename cuFFT<T>::C_t;

    // input_sizes_x stores how the input 3d array is distributed among the mpi processes
    size_t N1    = global_size->Nx / pcnt;
    size_t N1mod = global_size->Nx % pcnt;
    for (int p = 0; p < pcnt; ++p) {
        input_sizes_x[p]  = N1 + ((static_cast<size_t>(p) < N1mod) ? 1 : 0);
        input_start_x[p] = ((p==0) ? 0 : input_start_x[p-1]+input_sizes_x[p-1]);
    }

    // We only divide across the x-axis
    input_size_y = global_size->Ny; input_size_z = global_size->Nz;

    // After transposing the array, it is divided across the z-axis
    size_t N2 = (global_size->Nz/2+1) / pcnt;
    size_t N2mod = (global_size->Nz/2+1) % pcnt;

    for (int p = 0; p < pcnt; p++){
        output_sizes_z[p] = N2 + ((static_cast<size_t>(p) < N2mod) ? 1 : 0);
        output_start_z[p] = ((p==0) ? 0 : output_start_z[p-1]+output_sizes_z[p-1]);
    }
    // For real input values, the second half (of the z-axis) is symmetric (complex conjugate) to the first half
    output_size_x = global_size->Nx; output_size_y = global_size->Ny;

    domainsize = sizeof(C_t) * std::max(input_sizes_x[pidx]*input_size_y*(input_size_z/2 + 1), 
        output_size_x*output_size_y*output_sizes_z[pidx]);

    // Sizes of the different workspaces
    size_t ws_r2c, ws_c2c;
        
    if (fft3d) { // Combined 3d fft, in case only one mpi process is used
        CUFFT_CALL(cufftCreate(&planR2C));
        CUFFT_CALL(cufftSetAutoAllocation(planR2C, 0));

        CUFFT_CALL(cufftMakePlan3d(planR2C, global_size->Nx, global_size->Ny, global_size->Nz, cuFFT<T>::R2Ctype, &ws_r2c));

        fft_worksize = ws_r2c;
    } else {
        size_t batch = input_size_y * input_sizes_x[pidx];
        CUFFT_CALL(cufftCreate(&planR2C));
        CUFFT_CALL(cufftSetAutoAllocation(planR2C, 0));

        CUFFT_CALL(cufftCreate(&planC2C));
        CUFFT_CALL(cufftSetAutoAllocation(planC2C, 0));

        long long n[3] = {static_cast<long long>(output_size_x), static_cast<long long>(output_size_y), 
            static_cast<long long>(input_size_z)};
        long long inembed[3] = {static_cast<long long>(input_size_z), static_cast<long long>(output_size_y*output_size_x), static_cast<long long>(output_size_y)};
        long long onembed[3] = {1, 1, static_cast<long long>(output_size_y)};

        // For the 1D R2C FFT, the default data layer can be used (in case sequence = Z_Then_YX)
        CUFFT_CALL(cufftMakePlanMany64(planR2C, 1, &n[2], 
            &inembed[0], 1, inembed[0], // istride = 1, idist = Nz
            &onembed[0], input_size_y*input_sizes_x[pidx], 1, // ostride = Ny*Nxp, odist = 1
            cuFFT<T>::R2Ctype, batch, &ws_r2c)); // batch = Ny*Nxp

        batch = output_sizes_z[pidx];

        CUFFT_CALL(cufftMakePlanMany64(planC2C, 2, &n[0], 
            &inembed[1], 1, inembed[1], // istride = 1, idist = Ny*Nx, nembed[1] = Ny
            &onembed[1], output_sizes_z[pidx], 1, // ostride = Nzp, odist = 1, nembed[1] = Ny
            cuFFT<T>::C2Ctype, batch, &ws_c2c)); // batch = Nzp

        fft_worksize = std::max(ws_r2c, ws_c2c);
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
    timer->stop_store("init");
}

template<typename T> 
void MPIcuFFT_Slab_Z_Then_YX_Opt1<T>::MPIsend_Callback(void *data) {
  struct Callback_Params *params = (Callback_Params *)data;
  struct Callback_Params_Base *base_params = params->base_params;
  {
    std::lock_guard<std::mutex> lk(base_params->mutex);
    base_params->comm_ready.push_back(params->p);
  }
  base_params->cv.notify_one();
}

template<typename T>
void MPIcuFFT_Slab_Z_Then_YX_Opt1<T>::MPIsend_Thread(Callback_Params_Base &base_params, void *ptr) {
  using C_t = typename cuFFT<T>::C_t;
  C_t *send_ptr = (C_t *) ptr;

  for (int i = 0; i < comm_order.size(); i++){
    std::unique_lock<std::mutex> lk(base_params.mutex);
    base_params.cv.wait(lk, [&base_params]{return !base_params.comm_ready.empty();});

    int p = base_params.comm_ready.back();
    base_params.comm_ready.pop_back();

    if (i == 0)
      timer->stop_store("Transpose (First Send)");

    size_t oslice = output_start_z[p]*input_size_y*input_sizes_x[pidx];

    MPI_Isend(&send_ptr[oslice], sizeof(C_t)*output_sizes_z[p]*input_size_y*input_sizes_x[pidx], 
        MPI_BYTE, p, pidx, comm, &send_req[p]);

    lk.unlock();
  }
  timer->stop_store("Transpose (Packing)");
}

template<typename T>
void MPIcuFFT_Slab_Z_Then_YX_Opt1<T>::execR2C(void *out, const void *in) {
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
        timer->start();
        CUFFT_CALL(cuFFT<T>::execR2C(planR2C, real, complex));

        /* ***********************************************************************************************************************
        *                                                       Global Transpose
        *  *********************************************************************************************************************** */

        C_t *recv_ptr, *send_ptr, *temp_ptr;
        temp_ptr = cuFFT<T>::complex(mem_d[0]);
        if (cuda_aware) {
            recv_ptr = cuFFT<T>::complex(mem_d[1]);
        } else {
            recv_ptr = cuFFT<T>::complex(mem_h[0]);
            send_ptr = cuFFT<T>::complex(mem_h[1]);
        }
        recv_req[pidx] = MPI_REQUEST_NULL;
        send_req[pidx] = MPI_REQUEST_NULL;


        /* We are interested in sending the block via MPI as soon as possible.
        *  If MPI is compiled with the cuda_aware flag, then we are able to send the block directly from device memory.
        *  Otherwise, we need to copy the data to host memory first. Directly after the stream is done copying, a second
        *  thread is notified (via cudaLauchHostFunc) which calls MPI_Isendv.
        */
        Callback_Params_Base base_params;
        std::vector<Callback_Params> params_array;
        if (!cuda_aware) {
            for (int i = 0; i < pcnt; i++){
                Callback_Params params = {&base_params, i};
                params_array.push_back(params);
            }
        }
        CUDA_CALL(cudaDeviceSynchronize());
        timer->stop_store("1D FFT Z-Direction");

        for (auto p : comm_order) {
            MPI_Irecv(&recv_ptr[output_size_y*input_start_x[p]*output_sizes_z[pidx]], 
                sizeof(C_t)*output_size_y*input_sizes_x[p]*output_sizes_z[pidx], MPI_BYTE, p, p, comm, &recv_req[p]);

            size_t oslice = input_size_y*input_sizes_x[pidx]*output_start_z[p];

            if (!cuda_aware) {
                // data is aligned correctly, but it has to be copied to host memory first
                CUDA_CALL(cudaMemcpyAsync(&send_ptr[oslice], &complex[oslice], 
                    sizeof(C_t)*input_size_y*input_sizes_x[pidx]*output_sizes_z[p], 
                    cudaMemcpyDeviceToHost, streams[p]));
                
                // Callback function for the specific stream
                CUDA_CALL(cudaLaunchHostFunc(streams[p], this->MPIsend_Callback, (void *)&params_array[p]));
            } else {
                if (p == comm_order[0])
                    timer->stop_store("Transpose (First Send)");

                MPI_Isend(&complex[oslice], 
                    sizeof(C_t)*input_size_y*input_sizes_x[pidx]*output_sizes_z[p], 
                    MPI_BYTE, p, pidx, comm, &send_req[p]);
            }
        }
        std::thread mpisend_thread;
        if (!cuda_aware)
            mpisend_thread = std::thread(&MPIcuFFT_Slab_Z_Then_YX_Opt1<T>::MPIsend_Thread, this, std::ref(base_params), send_ptr);
        timer->stop_store("Transpose (Start Local Transpose)");
        { 
            // transpose local block
            size_t oslice = output_size_y*input_sizes_x[pidx]*output_start_z[pidx];

            CUDA_CALL(cudaMemcpy2DAsync(&temp_ptr[output_size_y*input_start_x[pidx]], sizeof(C_t)*output_size_y*output_size_x,
                &complex[oslice], sizeof(C_t)*input_size_y*input_sizes_x[pidx],
                sizeof(C_t)*input_size_y*input_sizes_x[pidx], output_sizes_z[pidx],
                cudaMemcpyDeviceToDevice, streams[pidx]));
        }
        timer->stop_store("Transpose (Start Receive)");
        int p;
        do {
            MPI_Waitany(pcnt, recv_req.data(), &p, MPI_STATUSES_IGNORE);
            if (p == MPI_UNDEFINED) 
                break;

            size_t oslice = output_size_y*input_start_x[p]*output_sizes_z[pidx];   

            CUDA_CALL(cudaMemcpy2DAsync(&temp_ptr[output_size_y*input_start_x[p]], sizeof(C_t)*output_size_y*output_size_x,
                &recv_ptr[oslice], sizeof(C_t)*output_size_y*input_sizes_x[p],
                sizeof(C_t)*output_size_y*input_sizes_x[p], output_sizes_z[pidx],
                cuda_aware?cudaMemcpyDeviceToDevice:cudaMemcpyHostToDevice, streams[p]));
        } while(p != MPI_UNDEFINED);
        CUDA_CALL(cudaDeviceSynchronize());
        timer->stop_store("Transpose (Finished Receive)");
        
        // avoid modification of complex, while MPI_Isendv is not done yet
        if (cuda_aware)
            MPI_Waitall(pcnt, send_req.data(), MPI_STATUSES_IGNORE);

        // compute remaining 1d FFT, for cuda-aware recv and temp buffer are identical
        CUFFT_CALL(cuFFT<T>::execC2C(planC2C, temp_ptr, complex, CUFFT_FORWARD));
        CUDA_CALL(cudaDeviceSynchronize());
        timer->stop_store("2D FFT Y-X-Direction");
        if (!cuda_aware) {
            mpisend_thread.join();
            MPI_Waitall(pcnt, send_req.data(), MPI_STATUSES_IGNORE);
        }
        timer->stop_store("Run complete");
    }
    timer->gather();
}

template class MPIcuFFT_Slab_Z_Then_YX_Opt1<float>;
template class MPIcuFFT_Slab_Z_Then_YX_Opt1<double>;