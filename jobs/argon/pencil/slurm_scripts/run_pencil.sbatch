#!/bin/bash
#SBATCH -p all
#SBATCH --nodelist=argon-tesla1, argon-tesla2
#SBATCH --exclusive
#SBATCH --ntasks=4
#SBATCH --time=24:00:00
#SBATCH --job-name=pencil
#SBATCH --output=pencil.%j.out
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
echo "Pencil Default"
python launch.py --jobs argon/pencil/benchmarks_base.json argon/pencil/validation.json --build_dir "build_argon" --global_params "-p1 2 -p2 2 -b ../benchmarks/argon/forward"
echo "Pencil Opt1"
python launch.py --jobs argon/pencil/benchmarks_base.json argon/pencil/validation.json --build_dir "build_argon" --global_params "-p1 2 -p2 2 -b ../benchmarks/argon/forward --opt 1"
echo "Pencil Default Inverse"
python launch.py --jobs argon/pencil/benchmarks_base.json --build_dir "build_argon" --global_params "-t 2 -p1 2 -p2 2 -b ../benchmarks/argon/inverse"
echo "Pencil Opt1 Inverse"
python launch.py --jobs argon/pencil/benchmarks_base.json --build_dir "build_argon" --global_params "-t 2 -p1 2 -p2 2 -b ../benchmarks/argon/inverse --opt 1"

echo "all done"