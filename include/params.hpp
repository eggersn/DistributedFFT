/* 
* Copyright (C) 2021 Simon Egger
* 
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
* 
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
* 
* You should have received a copy of the GNU General Public License
* along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#pragma once

#include <vector>
#include <cstddef>
#include <iostream>

struct GlobalSize{
    GlobalSize(size_t Nx_, size_t Ny_, size_t Nz_){
        Nx = Nx_;
        Ny = Ny_;
        Nz = Nz_;

        Nz_out = Nz / 2 + 1;
    }

    size_t Nx;
    size_t Ny;
    size_t Nz;
    size_t Nz_out;
};

struct Partition{
    size_t P1;
    size_t P2;
};

struct Slab_Partition : public Partition {
    Slab_Partition(size_t P1_) {
        P1 = P1_;
        P2 = 1;
    }
};

struct Pencil_Partition : public Partition {
    Pencil_Partition(size_t P1_, size_t P2_) {
        P1 = P1_;
        P2 = P2_;
    }
};

struct Partition_Dimensions{
    void computeOffsets() {
        computeStart(&size_x, &start_x);
        computeStart(&size_y, &start_y);
        computeStart(&size_z, &start_z);
    }

    std::vector<size_t> size_x;
    std::vector<size_t> size_y;
    std::vector<size_t> size_z;

    std::vector<size_t> start_x;
    std::vector<size_t> start_y;
    std::vector<size_t> start_z;

private:
    void computeStart(std::vector<size_t> *size, std::vector<size_t> *start) {
        size_t offset = 0;
        for (size_t i = 0; i < size->size(); i++){
            start->push_back(offset);
            offset += (*size)[i];
        }
    }
};

enum CommunicationMethod {Peer2Peer, All2All};
enum SendMethod {Sync, Streams, MPI_Type};
struct Configurations {
    bool cuda_aware;
    int warmup_rounds;
    CommunicationMethod comm_method;
    SendMethod send_method;
    std::string benchmark_dir;
    CommunicationMethod comm_method2;
    SendMethod send_method2;
};