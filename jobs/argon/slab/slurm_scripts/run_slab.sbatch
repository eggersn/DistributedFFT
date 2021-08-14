#!/bin/bash
#SBATCH -p all
#SBATCH --nodelist=argon-tesla1, argon-tesla2
#SBATCH --exclusive
#SBATCH --ntasks=4
#SBATCH --time=24:00:00
#SBATCH --job-name=slab
#SBATCH --output=slab.%j.out
#SBATCH --account=st

# load modules
module load mpi/u2004/openmpi-4.1.1-cuda
echo "Modules loaded"

# build
echo "start building"
cd /home/eggersn/DistributedFFT/
rm -rf build_argon
mkdir build_argon
cd build_argon

cmake ..
cmake --build .
echo "finished building"

sleep 5
cd ..

echo "start python script"
echo "-----------------------------------------------------------------------------"
echo "Slab 2D->1D default"
python launch.py --jobs argon/slab/benchmarks_base.json argon/slab/validation.json --build_dir "build_argon" --global_params "-p 4 -b ../benchmarks/argon/forward"
echo "Slab 2D->1D opt1"
python launch.py --jobs argon/slab/benchmarks_base.json argon/slab/validation.json --build_dir "build_argon" --global_params "-p 4 -b ../benchmarks/argon/forward --opt 1"
echo "Slab 1D->2D default"
python launch.py --jobs argon/slab/benchmarks_base.json argon/slab/validation.json --build_dir "build_argon" --global_params "-p 4 -b ../benchmarks/argon/forward -s Z_Then_YX"
echo "Slab 1D->2D opt1"
python launch.py --jobs argon/slab/benchmarks_base.json argon/slab/validation.json --build_dir "build_argon" --global_params "-p 4 -b ../benchmarks/argon/forward -s Z_Then_YX --opt 1"
echo "Slab 2D->1D default (inverse)"
python launch.py --jobs argon/slab/benchmarks_base.json --build_dir "build_argon" --global_params "-t 2 -p 4 -b ../benchmarks/argon/inverse"
echo "Slab 2D->1D opt1 (inverse)"
python launch.py --jobs argon/slab/benchmarks_base.json --build_dir "build_argon" --global_params "-t 2 -p 4 -b ../benchmarks/argon/inverse --opt 1"
echo "Slab 1D->2D default (inverse)"
python launch.py --jobs argon/slab/benchmarks_base.json --build_dir "build_argon" --global_params "-t 2 -p 4 -b ../benchmarks/argon/inverse -s Z_Then_YX"
echo "Slab 1D->2D opt1 (inverse)"
python launch.py --jobs argon/slab/benchmarks_base.json --build_dir "build_argon" --global_params "-t 2 -p 4 -b ../benchmarks/argon/inverse -s Z_Then_YX --opt 1"

echo "all done"