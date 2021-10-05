#!/bin/bash

echo "start building"
cd $HOME/DistributedFFT/
rm -rf build_pcsgs
mkdir build_pcsgs
cd build_pcsgs

cmake ..
cmake --build .
echo "finished building"

sleep 5
cd ..

echo "start python script"
echo "-----------------------------------------------------------------------------"
echo "Pencil Default"
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json bwunicluster/pencil/validation.json --build_dir "build_pcsgs" --global_params "-p1 2 -p2 2 -b ../benchmarks/pcsgs/forward"
echo "Pencil Opt1"
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json bwunicluster/pencil/validation.json --build_dir "build_pcsgs" --global_params "-p1 2 -p2 2 -b ../benchmarks/pcsgs/forward --opt 1"
echo "Pencil Default Inverse"
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json --build_dir "build_pcsgs" --global_params "-t 2 -p1 2 -p2 2 -b ../benchmarks/pcsgs/inverse"
echo "Pencil Opt1 Inverse"
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json --build_dir "build_pcsgs" --global_params "-t 2 -p1 2 -p2 2 -b ../benchmarks/pcsgs/inverse --opt 1"

echo "all done"