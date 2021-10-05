#!/bin/bash
#SBATCH -p sgs-only
#SBATCH --exclusive
#SBATCH --ntasks=4
#SBATCH --time=24:00:00
#SBATCH --job-name=slab
#SBATCH --output=slab.%j.out
#SBATCH --account=st

# load modules
module load gcc/7.4.0
module load cuda/10.2
module load krypton/cmake/3.14.3

export PATH="$PATH:/home/eggersn/opt/krypton/openmpi-4.1.1/bin"
echo "Modules loaded"

# build
echo "start building"
cd $HOME/DistributedFFT/
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
echo "Slab 2D->1D default"
python launch.py --jobs krypton/slab/benchmarks_base.json krypton/slab/validation.json --build_dir "build_krypton" --global_params "-p 4 -b ../benchmarks/krypton/forward"
echo "Slab 2D->1D opt1"
python launch.py --jobs krypton/slab/benchmarks_base.json krypton/slab/validation.json --build_dir "build_krypton" --global_params "-p 4 -b ../benchmarks/krypton/forward --opt 1"
echo "Slab 1D->2D default"
python launch.py --jobs krypton/slab/benchmarks_base.json krypton/slab/validation.json --build_dir "build_krypton" --global_params "-p 4 -b ../benchmarks/krypton/forward -s Z_Then_YX"
echo "Slab 1D->2D opt1"
python launch.py --jobs krypton/slab/benchmarks_base.json krypton/slab/validation.json --build_dir "build_krypton" --global_params "-p 4 -b ../benchmarks/krypton/forward -s Z_Then_YX --opt 1"
echo "Slab 2D->1D default (inverse)"
python launch.py --jobs krypton/slab/benchmarks_base.json --build_dir "build_krypton" --global_params "-t 2 -p 4 -b ../benchmarks/krypton/inverse"
echo "Slab 2D->1D opt1 (inverse)"
python launch.py --jobs krypton/slab/benchmarks_base.json --build_dir "build_krypton" --global_params "-t 2 -p 4 -b ../benchmarks/krypton/inverse --opt 1"
echo "Slab 1D->2D default (inverse)"
python launch.py --jobs krypton/slab/benchmarks_base.json --build_dir "build_krypton" --global_params "-t 2 -p 4 -b ../benchmarks/krypton/inverse -s Z_Then_YX"
echo "Slab 1D->2D opt1 (inverse)"
python launch.py --jobs krypton/slab/benchmarks_base.json --build_dir "build_krypton" --global_params "-t 2 -p 4 -b ../benchmarks/krypton/inverse -s Z_Then_YX --opt 1"

echo "all done"