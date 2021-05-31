#pragma once
#include <iostream>
#include <vector>
#include <thread> 
#include <mutex>
#include <mpi.h>

class Timer {
public:
    Timer(MPI_Comm comm, int p_gather, int pcnt, int pidx, std::vector<std::string> &descs, std::string filename);
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
    std::vector<std::string> &descs;
    std::string filename;

    std::mutex mutex;
};