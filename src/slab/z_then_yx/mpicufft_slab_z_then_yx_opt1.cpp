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
MPIcuFFT_Slab_Z_Then_YX_Opt1<T>::~MPIcuFFT_Slab_Z_Then_YX_Opt1() {
    if (planC2C_inv)
        CUFFT_CALL(cufftDestroy(planC2C_inv));
}

template<typename T> 
void MPIcuFFT_Slab_Z_Then_YX_Opt1<T>::initFFT(GlobalSize *global_size, bool allocate) {
    mkdir((config.benchmark_dir +  "/slab_z_then_yx").c_str(), 0777);
    std::stringstream ss;
    ss << config.benchmark_dir <<  "/slab_z_then_yx/test_1_" << config.comm_method << "_" << config.send_method << "_" << global_size->Nx << "_" << global_size->Ny << "_" << global_size->Nz;
    ss << "_" << cuda_aware << "_" << pcnt << ".csv";
    std::string filename = ss.str();
    
    timer = new Timer(comm, 0, pcnt, pidx, section_descriptions, filename);
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
    size_t ws_r2c, ws_c2c, ws_c2c_inv, ws_c2r;
    CUFFT_CALL(cufftCreate(&planR2C));
    CUFFT_CALL(cufftSetAutoAllocation(planR2C, 0));

    CUFFT_CALL(cufftCreate(&planC2R));
    CUFFT_CALL(cufftSetAutoAllocation(planC2R, 0));
        
    if (fft3d) { // Combined 3d fft, in case only one mpi process is used
        CUFFT_CALL(cufftMakePlan3d(planR2C, global_size->Nx, global_size->Ny, global_size->Nz, cuFFT<T>::R2Ctype, &ws_r2c));
        CUFFT_CALL(cufftMakePlan3d(planC2R, global_size->Nx, global_size->Ny, global_size->Nz, cuFFT<T>::C2Rtype, &ws_c2r));

        fft_worksize = std::max(ws_r2c, ws_c2r);
    } else {
        size_t batch = input_size_y * input_sizes_x[pidx];

        CUFFT_CALL(cufftCreate(&planC2C));
        CUFFT_CALL(cufftSetAutoAllocation(planC2C, 0));

        CUFFT_CALL(cufftCreate(&planC2C_inv));
        CUFFT_CALL(cufftSetAutoAllocation(planC2C_inv, 0));

        long long n[3] = {static_cast<long long>(output_size_x), static_cast<long long>(output_size_y), 
            static_cast<long long>(input_size_z)};
        long long inembed[3] = {static_cast<long long>(input_size_z), static_cast<long long>(output_size_y*output_size_x), static_cast<long long>(output_size_y)};
        long long onembed[3] = {1, 1, static_cast<long long>(output_size_y)};

        // For the 1D R2C FFT, the default data layer can be used (in case sequence = Z_Then_YX)
        CUFFT_CALL(cufftMakePlanMany64(planR2C, 1, &n[2], 
            &inembed[0], 1, inembed[0], // istride = 1, idist = Nz
            &onembed[0], input_size_y*input_sizes_x[pidx], 1, // ostride = Ny*Nxp, odist = 1
            cuFFT<T>::R2Ctype, batch, &ws_r2c)); // batch = Ny*
            
        CUFFT_CALL(cufftMakePlanMany64(planC2R, 1, &n[2], 
            &onembed[0], input_size_y*input_sizes_x[pidx], 1, // istride = Ny*Nxp, idist = 1
            &inembed[0], 1, inembed[0], // ostride = 1, odist = Nz
            cuFFT<T>::C2Rtype, batch, &ws_c2r)); // batch = Ny*Nxp

        batch = output_sizes_z[pidx];

        CUFFT_CALL(cufftMakePlanMany64(planC2C, 2, &n[0], 
            &inembed[1], 1, inembed[1], // istride = 1, idist = Ny*Nx, nembed[1] = Ny
            &onembed[1], output_sizes_z[pidx], 1, // ostride = Nzp, odist = 1, nembed[1] = Ny
            cuFFT<T>::C2Ctype, batch, &ws_c2c)); // batch = Nzp

        CUFFT_CALL(cufftMakePlanMany64(planC2C_inv, 2, &n[0], 
            &onembed[1], output_sizes_z[pidx], 1, // ostride = Nzp, odist = 1, nembed[1] = Ny
            &inembed[1], 1, inembed[1], // istride = 1, idist = Ny*Nx, nembed[1] = Ny
            cuFFT<T>::C2Ctype, batch, &ws_c2c_inv)); // batch = Nzp

        fft_worksize = std::max(std::max(ws_r2c, ws_c2c), std::max(ws_c2c_inv, ws_c2r));
    } 

    if (fft_worksize < domainsize)
        fft_worksize = domainsize;

    // worksize_d is split into 3 parts:
    // 1. space for received data, 2. space for transmitted data, 3. actual workspace (see "mem_d")
    worksize_d = fft_worksize + (fft3d ? 0 : (config.send_method == MPI_Type || !cuda_aware ? domainsize : 2*domainsize));
    // analogously for the host worksize, if mpi is not cuda-aware
    worksize_h = (cuda_aware || fft3d ? 0 : 2*domainsize);

    if (allocate) {
        this->setWorkArea();
        if (!fft3d) 
            CUFFT_CALL(cufftSetWorkArea(planC2C_inv, mem_d[!cuda_aware || config.send_method == MPI_Type ? 1 : 2]));
    }

    if (config.comm_method == Peer2Peer) {
        if (config.send_method == Streams) { 
            for (int i = 0; i < pcnt; i++){
                Callback_Params params = {&base_params, i};
                params_array.push_back(params);
            }
        } else if (config.send_method == MPI_Type) {
            MPI_PENCILS = std::vector<MPI_Datatype>(pcnt);
            for (int i = 0; i < pcnt; i++) {
                MPI_Type_vector(output_sizes_z[pidx], output_size_y*input_sizes_x[i]*sizeof(C_t), output_size_y*output_size_x*sizeof(C_t), MPI_BYTE, &MPI_PENCILS[i]);
                MPI_Type_commit(&MPI_PENCILS[i]);
            }
        }
    } else {
        if (config.send_method == MPI_Type) {
            MPI_PENCILS = std::vector<MPI_Datatype>(pcnt);
            MPI_SND = std::vector<MPI_Datatype>(pcnt);

            sendcounts = std::vector<int>(pcnt, 0);
            sdispls = std::vector<int>(pcnt, 0);
            recvcounts = std::vector<int>(pcnt, 1);
            rdispls = std::vector<int>(pcnt, 0);
            for (int p = 0; p < pcnt; p++) {
                sendcounts[p] = output_sizes_z[p]*output_size_y*input_sizes_x[pidx]*sizeof(C_t);
                sdispls[p] = output_start_z[p]*output_size_y*input_sizes_x[pidx]*sizeof(C_t);
                rdispls[p] = output_size_y*input_start_x[p]*sizeof(C_t);
                MPI_Type_vector(output_sizes_z[pidx], output_size_y*input_sizes_x[p]*sizeof(C_t), output_size_y*output_size_x*sizeof(C_t), MPI_BYTE, &MPI_PENCILS[p]);
                MPI_Type_commit(&MPI_PENCILS[p]);
                MPI_SND[p] = MPI_BYTE;
            }
        } else {
            sendcounts = std::vector<int>(pcnt, 0);
            sdispls = std::vector<int>(pcnt, 0);
            recvcounts = std::vector<int>(pcnt, 0);
            rdispls = std::vector<int>(pcnt, 0);
            for (int p = 0; p < pcnt; p++) {
                sendcounts[p] = output_sizes_z[p]*output_size_y*input_sizes_x[pidx]*sizeof(C_t);
                sdispls[p] = output_start_z[p]*output_size_y*input_sizes_x[pidx]*sizeof(C_t);
                recvcounts[p] = output_sizes_z[pidx]*output_size_y*input_sizes_x[p]*sizeof(C_t);
                rdispls[p] = output_sizes_z[pidx]*output_size_y*input_start_x[p]*sizeof(C_t);
            }
        }
    }
    
    CUDA_CALL(cudaDeviceSynchronize());
    timer->stop_store("init");
}

template<typename T>
void MPIcuFFT_Slab_Z_Then_YX_Opt1<T>::Peer2Peer_Sync(void *complex_, void *recv_ptr_, bool forward) {
    using C_t = typename cuFFT<T>::C_t;
    C_t *complex = cuFFT<T>::complex(complex_);
    C_t *recv_ptr = cuFFT<T>::complex(recv_ptr_);
    C_t *send_ptr;

    if (forward) {
        if (cuda_aware) {
            send_ptr = complex;
        } else {
            send_ptr = cuFFT<T>::complex(mem_h[0]);
            CUDA_CALL(cudaMemcpyAsync(send_ptr, complex, sizeof(C_t)*input_sizes_x[pidx]*input_size_y*(input_size_z/2+1), cudaMemcpyDeviceToHost));
            CUDA_CALL(cudaDeviceSynchronize());
        }

        for (auto p : comm_order) { 
                // start non-blocking receive for rank p
                MPI_Irecv(&recv_ptr[output_size_y*input_start_x[p]*output_sizes_z[pidx]], 
                    sizeof(C_t)*output_size_y*input_sizes_x[p]*output_sizes_z[pidx], MPI_BYTE, p, p, comm, &recv_req[p]);

                size_t oslice = output_start_z[p]*input_size_y*input_sizes_x[pidx];

                if (p == comm_order[0])
                    timer->stop_store("Transpose (First Send)");

                // complex can be used directly as send buffer
                MPI_Isend(&send_ptr[oslice], sizeof(C_t)*output_sizes_z[p]*input_size_y*input_sizes_x[pidx], 
                    MPI_BYTE, p, pidx, comm, &send_req[p]);
        }
        timer->stop_store("Transpose (Packing)");
    } else {
        C_t *temp_ptr = cuFFT<T>::complex(mem_d[0]);
        if (cuda_aware)
            send_ptr = cuFFT<T>::complex(mem_d[1]);
        else
            send_ptr = cuFFT<T>::complex(mem_h[0]);

        for (auto p : comm_order) {
            MPI_Irecv(&recv_ptr[output_start_z[p]*input_size_y*input_sizes_x[pidx]], 
                sizeof(C_t)*output_sizes_z[p]*input_size_y*input_sizes_x[pidx], MPI_BYTE, p, p, comm, &recv_req[p]);

            size_t oslice = output_size_y*input_start_x[p]*output_sizes_z[pidx];   

            CUDA_CALL(cudaMemcpy2DAsync(&send_ptr[oslice], sizeof(C_t)*output_size_y*input_sizes_x[p],
                &temp_ptr[output_size_y*input_start_x[p]], sizeof(C_t)*output_size_y*output_size_x,
                sizeof(C_t)*output_size_y*input_sizes_x[p], output_sizes_z[pidx],
                cuda_aware?cudaMemcpyDeviceToDevice:cudaMemcpyDeviceToHost));

            CUDA_CALL(cudaDeviceSynchronize());

            if (p == comm_order[0])
                    timer->stop_store("Transpose (First Send)");

            MPI_Isend(&send_ptr[oslice], sizeof(C_t)*output_size_y*input_sizes_x[p]*output_sizes_z[pidx], 
                MPI_BYTE, p, pidx, comm, &send_req[p]);
        }
        timer->stop_store("Transpose (Packing)");
    }
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

    if (forward) {
        size_t oslice = output_start_z[p]*input_size_y*input_sizes_x[pidx];
        MPI_Isend(&send_ptr[oslice], sizeof(C_t)*output_sizes_z[p]*input_size_y*input_sizes_x[pidx], 
            MPI_BYTE, p, pidx, comm, &send_req[p]);
    } else {
        size_t oslice = output_size_y*input_start_x[p]*output_sizes_z[pidx];   
        MPI_Isend(&send_ptr[oslice], sizeof(C_t)*output_size_y*input_sizes_x[p]*output_sizes_z[pidx], 
            MPI_BYTE, p, pidx, comm, &send_req[p]);
    }

    lk.unlock();
  }
  timer->stop_store("Transpose (Packing)");
}

template<typename T>
void MPIcuFFT_Slab_Z_Then_YX_Opt1<T>::Peer2Peer_Streams(void *complex_, void *recv_ptr_, bool forward) {
    using C_t = typename cuFFT<T>::C_t;
    C_t *complex = cuFFT<T>::complex(complex_);
    C_t *recv_ptr = cuFFT<T>::complex(recv_ptr_);
    C_t *send_ptr;

    if (forward) {
        if (!cuda_aware) 
            send_ptr = cuFFT<T>::complex(mem_h[0]);

        if (!cuda_aware)
            mpisend_thread = std::thread(&MPIcuFFT_Slab_Z_Then_YX_Opt1<T>::MPIsend_Thread, this, std::ref(base_params), send_ptr);
        else 
            timer->stop_store("Transpose (Packing)");  
            
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
    } else {
        C_t *temp_ptr = cuFFT<T>::complex(mem_d[0]);
        if (cuda_aware)
            send_ptr = cuFFT<T>::complex(mem_d[1]);
        else
            send_ptr = cuFFT<T>::complex(mem_h[0]);

        mpisend_thread = std::thread(&MPIcuFFT_Slab_Z_Then_YX_Opt1<T>::MPIsend_Thread, this, std::ref(base_params), send_ptr);
        for (auto p : comm_order) {
            MPI_Irecv(&recv_ptr[output_start_z[p]*input_size_y*input_sizes_x[pidx]], 
                sizeof(C_t)*output_sizes_z[p]*input_size_y*input_sizes_x[pidx], MPI_BYTE, p, p, comm, &recv_req[p]);

            size_t oslice = output_size_y*input_start_x[p]*output_sizes_z[pidx];   

            CUDA_CALL(cudaMemcpy2DAsync(&send_ptr[oslice], sizeof(C_t)*output_size_y*input_sizes_x[p],
                &temp_ptr[output_size_y*input_start_x[p]], sizeof(C_t)*output_size_y*output_size_x,
                sizeof(C_t)*output_size_y*input_sizes_x[p], output_sizes_z[pidx],
                cuda_aware?cudaMemcpyDeviceToDevice:cudaMemcpyDeviceToHost, streams[p]));

            CUDA_CALL(cudaLaunchHostFunc(streams[p], this->MPIsend_Callback, (void *)&params_array[p]));
        }
    }
}

template<typename T>
void MPIcuFFT_Slab_Z_Then_YX_Opt1<T>::Peer2Peer_MPIType(void *complex_, void *recv_ptr_, bool forward) {
    using C_t = typename cuFFT<T>::C_t;
    C_t *complex = cuFFT<T>::complex(complex_);
    C_t *recv_ptr = cuFFT<T>::complex(recv_ptr_);
    C_t *send_ptr;

    if (forward) {
        if (cuda_aware) {
            send_ptr = complex;
        } else {
            send_ptr = cuFFT<T>::complex(mem_h[0]);
            CUDA_CALL(cudaMemcpyAsync(send_ptr, complex, sizeof(C_t)*input_sizes_x[pidx]*input_size_y*(input_size_z/2+1), cudaMemcpyDeviceToHost));
            CUDA_CALL(cudaDeviceSynchronize());
        }
        timer->stop_store("Transpose (Packing)");

        for (auto p : comm_order) { 
            // start non-blocking receive for rank p
            MPI_Irecv(&recv_ptr[input_size_y*input_start_x[p]], 1, MPI_PENCILS[p], p, p, comm, &recv_req[p]);

            size_t oslice = output_start_z[p]*input_size_y*input_sizes_x[pidx];

            if (p == comm_order[0])
                timer->stop_store("Transpose (First Send)");

            MPI_Isend(&send_ptr[oslice], sizeof(C_t)*output_sizes_z[p]*input_size_y*input_sizes_x[pidx], 
                MPI_BYTE, p, pidx, comm, &send_req[p]);
        }
    } else {
        C_t *temp_ptr = cuFFT<T>::complex(mem_d[0]);
        if (cuda_aware) {
            send_ptr = temp_ptr;
        } else {
            send_ptr = cuFFT<T>::complex(mem_h[0]);
            CUDA_CALL(cudaMemcpyAsync(send_ptr, temp_ptr, sizeof(C_t)*output_sizes_z[pidx]*output_size_y*output_size_x, cudaMemcpyDeviceToHost));
            CUDA_CALL(cudaDeviceSynchronize());
        }
        timer->stop_store("Transpose (Packing)");

        for (auto p : comm_order) {
            MPI_Irecv(&recv_ptr[output_start_z[p]*input_size_y*input_sizes_x[pidx]], 
                sizeof(C_t)*output_sizes_z[p]*input_size_y*input_sizes_x[pidx], MPI_BYTE, p, p, comm, &recv_req[p]);

            if (p == comm_order[0])
                    timer->stop_store("Transpose (First Send)");

            MPI_Isend(&send_ptr[input_size_y*input_start_x[p]], 1, MPI_PENCILS[p], p, pidx, comm, &send_req[p]);
        }
    }
}

template<typename T>
void MPIcuFFT_Slab_Z_Then_YX_Opt1<T>::Peer2Peer_Communication(void *complex_, bool forward) {
    using C_t = typename cuFFT<T>::C_t;
    C_t *complex = cuFFT<T>::complex(complex_);
    C_t *recv_ptr, *temp_ptr;
    temp_ptr = cuFFT<T>::complex(mem_d[0]);

    if (forward) {
        if (config.send_method == MPI_Type) {
            if (cuda_aware)
                recv_ptr = temp_ptr;
            else 
                recv_ptr = cuFFT<T>::complex(mem_h[1]);

            this->Peer2Peer_MPIType(complex_, (void *) recv_ptr);


            // transpose local block
            timer->stop_store("Transpose (Start Local Transpose)");
            {
                size_t oslice = output_size_y*input_sizes_x[pidx]*output_start_z[pidx];

                CUDA_CALL(cudaMemcpy2DAsync(&recv_ptr[output_size_y*input_start_x[pidx]], sizeof(C_t)*output_size_y*output_size_x,
                    &complex[oslice], sizeof(C_t)*input_size_y*input_sizes_x[pidx],
                    sizeof(C_t)*input_size_y*input_sizes_x[pidx], output_sizes_z[pidx],
                    cuda_aware?cudaMemcpyDeviceToDevice:cudaMemcpyDeviceToHost, streams[pidx]));
            }
            CUDA_CALL(cudaDeviceSynchronize());

            timer->stop_store("Transpose (Start Receive)");
            MPI_Waitall(pcnt, recv_req.data(), MPI_STATUSES_IGNORE);
            timer->stop_store("Transpose (First Receive)");
            timer->stop_store("Transpose (Finished Receive)");        
            if (!cuda_aware) {
                CUDA_CALL(cudaMemcpyAsync(temp_ptr, recv_ptr, sizeof(C_t)*output_sizes_z[pidx]*output_size_y*output_size_x, cudaMemcpyDeviceToHost));
                CUDA_CALL(cudaDeviceSynchronize());
            }
            timer->stop_store("Transpose (Unpacking)");
        } else {
            if (cuda_aware)
                recv_ptr = cuFFT<T>::complex(mem_d[1]);
            else 
                recv_ptr = cuFFT<T>::complex(mem_h[1]);

            if (config.send_method == Sync) 
                this->Peer2Peer_Sync(complex_, (void *) recv_ptr);
            else if (config.send_method == Streams)
                this->Peer2Peer_Streams(complex_, (void *) recv_ptr);

            // transpose local block
            timer->stop_store("Transpose (Start Local Transpose)");
            { 
                size_t oslice = output_size_y*input_sizes_x[pidx]*output_start_z[pidx];

                CUDA_CALL(cudaMemcpy2DAsync(&temp_ptr[output_size_y*input_start_x[pidx]], sizeof(C_t)*output_size_y*output_size_x,
                    &complex[oslice], sizeof(C_t)*input_size_y*input_sizes_x[pidx],
                    sizeof(C_t)*input_size_y*input_sizes_x[pidx], output_sizes_z[pidx],
                    cudaMemcpyDeviceToDevice, streams[pidx]));
            }
            timer->stop_store("Transpose (Start Receive)");
            int p,count=0;
            do {
                MPI_Waitany(pcnt, recv_req.data(), &p, MPI_STATUSES_IGNORE);
                if (p == MPI_UNDEFINED) 
                    break;
                if (count == 0)
                    timer->stop_store("Transpose (First Receive)");
                if (count == pcnt-2)
                    timer->stop_store("Transpose (Finished Receive)");            
                count++;

                size_t oslice = output_size_y*input_start_x[p]*output_sizes_z[pidx];   

                CUDA_CALL(cudaMemcpy2DAsync(&temp_ptr[output_size_y*input_start_x[p]], sizeof(C_t)*output_size_y*output_size_x,
                    &recv_ptr[oslice], sizeof(C_t)*output_size_y*input_sizes_x[p],
                    sizeof(C_t)*output_size_y*input_sizes_x[p], output_sizes_z[pidx],
                    cuda_aware?cudaMemcpyDeviceToDevice:cudaMemcpyHostToDevice, streams[p]));
            } while(p != MPI_UNDEFINED);
            CUDA_CALL(cudaDeviceSynchronize());
            timer->stop_store("Transpose (Unpacking)");
        }
    } else {
        C_t *copy_ptr = complex;
        if (cuda_aware)
            recv_ptr = copy_ptr;
        else
            recv_ptr = cuFFT<T>::complex(mem_h[1]);
        
        if (config.send_method == Sync) 
            this->Peer2Peer_Sync(complex_, (void *) recv_ptr, false);
        else if (config.send_method == Streams)
            this->Peer2Peer_Streams(complex_, (void *) recv_ptr, false);
        else if (config.send_method == MPI_Type)
            this->Peer2Peer_MPIType(complex_, (void *) recv_ptr, false);

        // transpose local block
        timer->stop_store("Transpose (Start Local Transpose)");
        { 
            size_t oslice = output_size_y*input_sizes_x[pidx]*output_start_z[pidx];

            CUDA_CALL(cudaMemcpy2DAsync(&copy_ptr[oslice], sizeof(C_t)*input_size_y*input_sizes_x[pidx],
                &temp_ptr[output_size_y*input_start_x[pidx]], sizeof(C_t)*output_size_y*output_size_x,
                sizeof(C_t)*input_size_y*input_sizes_x[pidx], output_sizes_z[pidx],
                cudaMemcpyDeviceToDevice, streams[pidx]));
        }

        timer->stop_store("Transpose (Start Receive)");
        if (!cuda_aware) { // copy received blocks to device
            int p,count=0;
            do {
                MPI_Waitany(pcnt, recv_req.data(), &p, MPI_STATUSES_IGNORE);
                if (p == MPI_UNDEFINED)
                   break;
                if (count == 0)
                    timer->stop_store("Transpose (First Receive)");
                if (count == pcnt-2)
                    timer->stop_store("Transpose (Finished Receive)");
                count++;

                CUDA_CALL(cudaMemcpyAsync(&copy_ptr[output_start_z[p]*output_size_y*input_sizes_x[pidx]],
                                        &recv_ptr[output_start_z[p]*output_size_y*input_sizes_x[pidx]],
                                        output_sizes_z[p]*output_size_y*input_sizes_x[pidx]*sizeof(C_t), cudaMemcpyHostToDevice, streams[p]));
            } while(p != MPI_UNDEFINED);
        } else { // just wait for all receives
            MPI_Waitall(pcnt, recv_req.data(), MPI_STATUSES_IGNORE);
            timer->stop_store("Transpose (First Receive)");
            timer->stop_store("Transpose (Finished Receive)");
        }
        CUDA_CALL(cudaDeviceSynchronize());
        timer->stop_store("Transpose (Unpacking)");
    }
}

template<typename T> 
void MPIcuFFT_Slab_Z_Then_YX_Opt1<T>::All2All_Sync(void *complex_, bool forward) {
    using C_t = typename cuFFT<T>::C_t;
    C_t *complex = cuFFT<T>::complex(complex_);
    C_t *recv_ptr, *send_ptr, *temp_ptr;

    if (forward) { 
        temp_ptr = cuFFT<T>::complex(mem_d[0]);
        if (cuda_aware) {
            send_ptr = complex;
            recv_ptr = cuFFT<T>::complex(mem_d[1]);
        } else {
            send_ptr = cuFFT<T>::complex(mem_h[0]);
            recv_ptr = cuFFT<T>::complex(mem_h[1]);
            CUDA_CALL(cudaMemcpyAsync(send_ptr, complex, sizeof(C_t)*input_sizes_x[pidx]*input_size_y*(input_size_z/2+1), cudaMemcpyDeviceToHost));
            CUDA_CALL(cudaDeviceSynchronize());
        }
        timer->stop_store("Transpose (Packing)");

        timer->stop_store("Transpose (Start All2All)");
        MPI_Alltoallv(send_ptr, sendcounts.data(), sdispls.data(), MPI_BYTE, 
                        recv_ptr, recvcounts.data(), rdispls.data(), MPI_BYTE, comm);
        timer->stop_store("Transpose (Finished All2All)");

        for (int p = 0; p < pcnt; p++) {
            size_t oslice = output_size_y*input_start_x[p]*output_sizes_z[pidx];   

            CUDA_CALL(cudaMemcpy2DAsync(&temp_ptr[output_size_y*input_start_x[p]], sizeof(C_t)*output_size_y*output_size_x,
                &recv_ptr[oslice], sizeof(C_t)*output_size_y*input_sizes_x[p],
                sizeof(C_t)*output_size_y*input_sizes_x[p], output_sizes_z[pidx],
                cuda_aware?cudaMemcpyDeviceToDevice:cudaMemcpyHostToDevice, streams[p]));
        }  
        CUDA_CALL(cudaDeviceSynchronize());
        timer->stop_store("Transpose (Unpacking)");
    } else {
        temp_ptr = cuFFT<T>::complex(mem_d[0]);
        C_t *copy_ptr = complex;
        if (cuda_aware) {
            send_ptr = cuFFT<T>::complex(mem_d[1]);
            recv_ptr = copy_ptr;
        } else {
            send_ptr = cuFFT<T>::complex(mem_h[0]);
            recv_ptr = cuFFT<T>::complex(mem_h[1]);
        }

        for (int p = 0; p < pcnt; p++) {
            size_t oslice = output_size_y*input_start_x[p]*output_sizes_z[pidx]; 
            CUDA_CALL(cudaMemcpy2DAsync(&send_ptr[oslice], sizeof(C_t)*output_size_y*input_sizes_x[p],
                &temp_ptr[output_size_y*input_start_x[p]], sizeof(C_t)*output_size_y*output_size_x,
                sizeof(C_t)*output_size_y*input_sizes_x[p], output_sizes_z[pidx],
                cuda_aware?cudaMemcpyDeviceToDevice:cudaMemcpyDeviceToHost, streams[p]));
        }

        CUDA_CALL(cudaDeviceSynchronize());
        timer->stop_store("Transpose (Packing)");

        timer->stop_store("Transpose (Start All2All)");
        MPI_Alltoallv(send_ptr, recvcounts.data(), rdispls.data(), MPI_BYTE, 
                      recv_ptr, sendcounts.data(), sdispls.data(), MPI_BYTE, comm);
        timer->stop_store("Transpose (Finished All2All)");

        if (!cuda_aware) {
            CUDA_CALL(cudaMemcpyAsync(copy_ptr, recv_ptr, (input_size_z/2+1)*input_size_y*input_sizes_x[pidx]*sizeof(C_t), cudaMemcpyHostToDevice));
            CUDA_CALL(cudaDeviceSynchronize());
        }
        timer->stop_store("Transpose (Unpacking)");
    }
}

template<typename T> 
void MPIcuFFT_Slab_Z_Then_YX_Opt1<T>::All2All_MPIType(void *complex_, bool forward) {
    using C_t = typename cuFFT<T>::C_t;
    C_t *complex = cuFFT<T>::complex(complex_);
    C_t *recv_ptr, *send_ptr, *temp_ptr;

    temp_ptr = cuFFT<T>::complex(mem_d[0]);

    if (forward) {
        if (cuda_aware) {
            send_ptr = complex;
            recv_ptr = temp_ptr;
        } else {
            send_ptr = cuFFT<T>::complex(mem_h[0]);
            recv_ptr = cuFFT<T>::complex(mem_h[1]);
            CUDA_CALL(cudaMemcpyAsync(send_ptr, complex, sizeof(C_t)*input_sizes_x[pidx]*input_size_y*(input_size_z/2+1), cudaMemcpyDeviceToHost));
            CUDA_CALL(cudaDeviceSynchronize());
        }
        timer->stop_store("Transpose (Packing)");

        timer->stop_store("Transpose (Start All2All)");
        MPI_Alltoallw(send_ptr, sendcounts.data(), sdispls.data(), MPI_SND.data(), 
                        recv_ptr, recvcounts.data(), rdispls.data(), MPI_PENCILS.data(), comm);
        timer->stop_store("Transpose (Finished All2All)");

        if (!cuda_aware) {
            CUDA_CALL(cudaMemcpyAsync(temp_ptr, recv_ptr, output_sizes_z[pidx]*output_size_y*output_size_x*sizeof(C_t), cudaMemcpyHostToDevice));
            CUDA_CALL(cudaDeviceSynchronize());
        }
        timer->stop_store("Transpose (Unpacking)");
    } else {
        C_t *copy_ptr = complex;
        if (cuda_aware) {
            send_ptr = temp_ptr;
            recv_ptr = copy_ptr;
        } else {
            send_ptr = cuFFT<T>::complex(mem_h[0]);
            recv_ptr = cuFFT<T>::complex(mem_h[1]);
            CUDA_CALL(cudaMemcpyAsync(send_ptr, temp_ptr, output_sizes_z[pidx]*output_size_y*output_size_x*sizeof(C_t), cudaMemcpyDeviceToHost));
            CUDA_CALL(cudaDeviceSynchronize());
        }
        timer->stop_store("Transpose (Packing)");

        timer->stop_store("Transpose (Start All2All)");
        MPI_Alltoallw(send_ptr, recvcounts.data(), rdispls.data(), MPI_PENCILS.data(), 
                      recv_ptr, sendcounts.data(), sdispls.data(), MPI_SND.data(), comm);
        timer->stop_store("Transpose (Finished All2All)");

        if (!cuda_aware) {
            CUDA_CALL(cudaMemcpyAsync(copy_ptr, recv_ptr, sizeof(C_t)*input_sizes_x[pidx]*input_size_y*(input_size_z/2+1), cudaMemcpyHostToDevice));
            CUDA_CALL(cudaDeviceSynchronize());
        }
        timer->stop_store("Transpose (Unpacking)");
    }
}

template<typename T> 
void MPIcuFFT_Slab_Z_Then_YX_Opt1<T>::All2All_Communication(void *complex_, bool forward) {
  if (config.send_method == Sync)
    this->All2All_Sync(complex_, forward);
  else 
    this->All2All_MPIType(complex_, forward);
}

template<typename T>
void MPIcuFFT_Slab_Z_Then_YX_Opt1<T>::execR2C(void *out, const void *in) {
    if (!initialized) 
        return;

    forward = true;

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
        CUDA_CALL(cudaDeviceSynchronize());
        timer->stop_store("1D FFT Z-Direction");

        C_t* temp_ptr = cuFFT<T>::complex(mem_d[0]);

        if (config.comm_method == Peer2Peer) 
            this->Peer2Peer_Communication((void *)complex);
        else 
            this->All2All_Communication((void *)complex);        
        
        // avoid modification of complex, while MPI_Isend is not done yet
        if (config.comm_method == Peer2Peer && cuda_aware)
            MPI_Waitall(pcnt, send_req.data(), MPI_STATUSES_IGNORE);

        // compute remaining 1d FFT, for cuda-aware recv and temp buffer are identical
        CUFFT_CALL(cuFFT<T>::execC2C(planC2C, temp_ptr, complex, CUFFT_FORWARD));
        CUDA_CALL(cudaDeviceSynchronize());
        timer->stop_store("2D FFT Y-X-Direction");
        if (config.comm_method == Peer2Peer && !cuda_aware) {
            if (config.send_method == Streams)
                mpisend_thread.join();
            MPI_Waitall(pcnt, send_req.data(), MPI_STATUSES_IGNORE);
        }
        timer->stop_store("Run complete");
    }
    if (config.warmup_rounds == 0) 
        timer->gather();
    else 
        config.warmup_rounds--;
}

template<typename T> 
void MPIcuFFT_Slab_Z_Then_YX_Opt1<T>::execC2R(void *out, const void *in) {
  if (!initialized) 
    return;

  forward = false;

  using R_t = typename cuFFT<T>::R_t;
  using C_t = typename cuFFT<T>::C_t;

  C_t *complex = cuFFT<T>::complex(in);
  R_t *real    = cuFFT<T>::real(out);

  timer->start();
  if (fft3d) {
    CUFFT_CALL(cuFFT<T>::execC2R(planC2R, complex, real));
    CUDA_CALL(cudaDeviceSynchronize());
  } else {
    C_t *temp_ptr, *copy_ptr;
    temp_ptr = cuFFT<T>::complex(mem_d[0]);
    copy_ptr = complex;

    // compute 1d complex to complex FFT in x direction
    CUFFT_CALL(cuFFT<T>::execC2C(planC2C_inv, complex, temp_ptr, CUFFT_INVERSE));
    CUDA_CALL(cudaDeviceSynchronize());
    timer->stop_store("2D FFT Y-X-Direction");

    if (config.comm_method == Peer2Peer) 
      this->Peer2Peer_Communication((void *)complex, false);
    else 
      this->All2All_Communication((void *)complex, false);

    CUFFT_CALL(cuFFT<T>::execC2R(planC2R, copy_ptr, real));
    CUDA_CALL(cudaDeviceSynchronize());
    timer->stop_store("1D FFT Z-Direction");

    if (config.comm_method == Peer2Peer) {
      if (config.send_method == Streams)
        mpisend_thread.join();
      MPI_Waitall(pcnt, send_req.data(), MPI_STATUSES_IGNORE);
    }
    timer->stop_store("Run complete");
    if (config.warmup_rounds == 0) 
        timer->gather();
    else 
        config.warmup_rounds--;
  }
}

template class MPIcuFFT_Slab_Z_Then_YX_Opt1<float>;
template class MPIcuFFT_Slab_Z_Then_YX_Opt1<double>;