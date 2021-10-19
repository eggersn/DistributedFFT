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

#include "timer.hpp"
#include <algorithm>
#include <fstream>
#include <sys/stat.h>

Timer::Timer(MPI_Comm comm, int p_gather, int pcnt, int pidx, std::vector<std::string> descs, std::string filename) : 
    comm(comm), p_gather(p_gather), pcnt(pcnt), pidx(pidx), descs(descs), filename(filename) {
    durations.resize(descs.size(), 0);
    tstop_points.resize(descs.size(), 0);
}

void Timer::start() {
    // std::lock_guard<std::mutex> lk(mutex); 
    tstart = MPI_Wtime();
}

void Timer::stop(std::string desc) {
    // std::lock_guard<std::mutex> lk(mutex); 
    auto it = std::find(descs.begin(), descs.end(), desc);
    int index = std::distance(descs.begin(), it);        
    tstop_points[index] = MPI_Wtime();
}

double Timer::getDuration(std::string desc) {
    auto it = std::find(descs.begin(), descs.end(), desc);
    int index = std::distance(descs.begin(), it);  
    return (tstop_points[index] - tstart) * 1000;
}

double Timer::getDuration(int index) {
    return (tstop_points[index] - tstart) * 1000;
}

void Timer::store(std::string desc) {
    // std::lock_guard<std::mutex> lk(mutex); 
    auto it = std::find(descs.begin(), descs.end(), desc);
    int index = std::distance(descs.begin(), it);
    durations[index] = getDuration(index);
}

void Timer::gather() {
    std::vector<double> other_durations;
    int send_size = durations.size();

    // Some tests might only use a subset of workers for the actual computation.
    // Then, world_size > pcnt which is why we need to introduce the actual world_size for MPI_Gatherv
    int world_size;
    MPI_Comm_size(comm, &world_size);
    // Assumption: Each worker process has the same number of duration values
    std::vector<int> recv_count(pcnt, send_size);
    recv_count.resize(world_size, 0);

    std::vector<int> recv_displ(world_size, 0);
    for (int i = 1; i < world_size; i++)
        recv_displ[i] = recv_displ[i-1] + recv_count[i-1];

    int recv_size = durations.size() * pcnt;
    if (pidx == p_gather) {
        other_durations.resize(recv_size, 0);
    } 

    MPI_Gatherv(&durations[0], send_size, MPI_DOUBLE, 
        &other_durations[0], recv_count.data(), recv_displ.data(), MPI_DOUBLE, p_gather, comm);

    if (pidx == p_gather){
        std::ofstream myfile;
        struct stat buffer; 
        if (!stat (filename.c_str(), &buffer) == 0) {
            myfile.open(filename);
            myfile << ",";
            for (int i = 0; i < pcnt; i++)
                myfile << i << ",";
        } else {
            myfile.open(filename, std::ios_base::app);
        }
        myfile << "\n";
        for (int i = 0; i < durations.size(); i++){
            myfile << descs[i] << ",";
            for (int j = 0; j < pcnt; j++)
                myfile << other_durations[j * durations.size() + i] << ",";
            myfile << "\n";
        }
        myfile.close();
    }
}