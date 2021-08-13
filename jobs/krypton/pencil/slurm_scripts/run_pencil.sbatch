#!/bin/bash
#SBATCH -p sgs-only
#SBATCH --exclusive
#SBATCH --ntasks=4
#SBATCH --time=24:00:00
#SBATCH --job-name=pencil
#SBATCH --output=pencil.%j.out
#SBATCH --account=st

# load modules
module load gcc/7.4.0
module load cuda/10.2
module load krypton/cmake/3.14.3

export PATH="$PATH:/home/eggersn/opt/krypton/openmpi-4.1.1/bin"
echo "Modules loaded"

# build
echo "start building"
cd /home/eggersn/DistributedFFT/
rm -rf build_krypton
mkdir build_krypton
cd build_krypton

cmake ..
cmake --build .
echo "finished building"

sleep 5
cd ..

echo "start python script"
echo "-----------------------------------------------------------------------------"
echo "Pencil Default"
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json bwunicluster/pencil/validation.json --build_dir "build_krypton" --global_params "-p1 2 -p2 2 -b ../benchmarks/krypton/forward"
echo "Pencil Opt1"
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json bwunicluster/pencil/validation.json --build_dir "build_krypton" --global_params "-p1 2 -p2 2 -b ../benchmarks/krypton/forward --opt 1"
echo "Pencil Default Inverse"
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json --build_dir "build_krypton" --global_params "-t 2 -p1 2 -p2 2 -b ../benchmarks/krypton/inverse"
echo "Pencil Opt1 Inverse"
python launch.py --jobs bwunicluster/pencil/benchmarks_base.json --build_dir "build_krypton" --global_params "-t 2 -p1 2 -p2 2 -b ../benchmarks/krypton/inverse --opt 1"

echo "all done"