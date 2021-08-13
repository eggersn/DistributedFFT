#!/bin/bash

echo "start building"
cd /home/eggersn/DistributedFFT/
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
echo "Slab 2D->1D default"
python launch.py --jobs pcsgs/slab/benchmarks_base.json pcsgs/slab/validation.json --build_dir "build_pcsgs" --global_params "-p 4 -b ../benchmarks/pcsgs/forward"
echo "Slab 2D->1D opt1"
python launch.py --jobs pcsgs/slab/benchmarks_base.json pcsgs/slab/validation.json --build_dir "build_pcsgs" --global_params "-p 4 -b ../benchmarks/pcsgs/forward --opt 1"
echo "Slab 1D->2D default"
python launch.py --jobs pcsgs/slab/benchmarks_base.json pcsgs/slab/validation.json --build_dir "build_pcsgs" --global_params "-p 4 -b ../benchmarks/pcsgs/forward -s Z_Then_YX"
echo "Slab 1D->2D opt1"
python launch.py --jobs pcsgs/slab/benchmarks_base.json pcsgs/slab/validation.json --build_dir "build_pcsgs" --global_params "-p 4 -b ../benchmarks/pcsgs/forward -s Z_Then_YX --opt 1"
echo "Slab 2D->1D default (inverse)"
python launch.py --jobs pcsgs/slab/benchmarks_base.json --build_dir "build_pcsgs" --global_params "-t 2 -p 4 -b ../benchmarks/pcsgs/inverse"
echo "Slab 2D->1D opt1 (inverse)"
python launch.py --jobs pcsgs/slab/benchmarks_base.json --build_dir "build_pcsgs" --global_params "-t 2 -p 4 -b ../benchmarks/pcsgs/inverse --opt 1"
echo "Slab 1D->2D default (inverse)"
python launch.py --jobs pcsgs/slab/benchmarks_base.json --build_dir "build_pcsgs" --global_params "-t 2 -p 4 -b ../benchmarks/pcsgs/inverse -s Z_Then_YX"
echo "Slab 1D->2D opt1 (inverse)"
python launch.py --jobs pcsgs/slab/benchmarks_base.json --build_dir "build_pcsgs" --global_params "-t 2 -p 24 -b ../benchmarks/pcsgs/inverse -s Z_Then_YX --opt 1"

echo "all done"