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
#include <iostream>
#include <vector>
#include <thread> 
#include <mutex>
#include <mpi.h>

class Timer {
public:
    Timer(MPI_Comm comm, int p_gather, int pcnt, int pidx, std::vector<std::string> descs, std::string filename);
    void start();
    void stop(std::string desc);
    void store(std::string desc);
    void stop_store(std::string desc) {stop(desc); store(desc);}
    void setFileName(std::string filename_) {filename=filename_;}
    void gather();

protected:
    double getDuration(int index);
    double getDuration(std::string desc);

    MPI_Comm comm;
    int p_gather; 

    int pcnt, pidx;

    std::vector<double> durations;
    std::vector<double> tstop_points;
    double tstart;
    std::vector<std::string> descs;
    std::string filename;

    std::mutex mutex;
};