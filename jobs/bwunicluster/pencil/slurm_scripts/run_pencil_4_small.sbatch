#!/bin/bash
#SBATCH -p gpu_4
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --nodes=6
#SBATCH --ntasks=24
#SBATCH --time=18:00:00
#SBATCH --job-name=gpu4_pencil_small
#SBATCH --output=gpu4_pencil_small.%j.out
#SBATCH --account=st

# load modules
module load compiler/gnu/8.3.1
module load devel/cuda/11.0
module load devel/cmake/3.18
module load mpi/openmpi/4.1
echo "Modules loaded"

# determine hosts
HOSTS="$(mpirun | sort -n | sed -r 's/\.localdomain//'"
echo "$HOSTS"
HOST24="$(echo "$HOSTS" | tr '\n' ',' | sed 's/,$//' | sed 's/,/ /g'"
echo "24: $HOST24"

# build
echo "start building"
cd /home/st/st_us-051200/st_st160727/DistributedFFT/
rm -rf build_gpu4
mkdir build_gpu4
cd build_gpu4

cmake ..
cmake --build .
echo "finished building"

sleep 5
cd ..

echo "*****************************************************************************"
echo "Starting on HOST24"
echo "*****************************************************************************"
echo "-----------------------------------------------------------------------------"
echo "Partition 3x8 / 6x4"
echo "-----------------------------------------------------------------------------"
echo "Pencil Default"
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json bwunicluster/pencil/validation.json --hosts $HOST24 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-p1 3 -p2 8 -b ../benchmarks/bwunicluster/gpu4/forward"
echo "Pencil Opt1"
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json bwunicluster/pencil/validation.json --hosts $HOST24 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-p1 3 -p2 8 -b ../benchmarks/bwunicluster/gpu4/forward --opt 1"
echo "Pencil Default Inverse"
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json --hosts $HOST24 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-t 2 -p1 3 -p2 8 -b ../benchmarks/bwunicluster/gpu4/inverse"
echo "Pencil Opt1 Inverse"
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json --hosts $HOST24 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-t 2 -p1 3 -p2 8 -b ../benchmarks/bwunicluster/gpu4/inverse --opt 1"

echo "-----------------------------------------------------------------------------"
echo "Partition 6x4"
echo "-----------------------------------------------------------------------------"
echo "Pencil Default"
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json bwunicluster/pencil/validation.json --hosts $HOST24 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-p1 6 -p2 4 -b ../benchmarks/bwunicluster/gpu4/forward"
echo "Pencil Opt1"
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json bwunicluster/pencil/validation.json --hosts $HOST24 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-p1 6 -p2 4 -b ../benchmarks/bwunicluster/gpu4/forward --opt 1"
echo "Pencil Default Inverse"
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json --hosts $HOST24 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-t 2 -p1 6 -p2 4 -b ../benchmarks/bwunicluster/gpu4/inverse"
echo "Pencil Opt1 Inverse"
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json --hosts $HOST24 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-t 2 -p1 6 -p2 4 -b ../benchmarks/bwunicluster/gpu4/inverse --opt 1"

echo "-----------------------------------------------------------------------------"
echo "Partition 12x2"
echo "-----------------------------------------------------------------------------"
echo "Pencil Default"
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json --hosts $HOST24 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-p1 12 -p2 2 -b ../benchmarks/bwunicluster/gpu4/forward" --mpi_params "--mca btl_openib_warn_default_gid_prefix 0"
echo "Pencil Opt1"
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json --hosts $HOST24 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-t 2 -p1 12 -p2 2 -b ../benchmarks/bwunicluster/gpu4/inverse" --mpi_params "--mca btl_openib_warn_default_gid_prefix 0"
echo "Pencil Default Inverse"
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json --hosts $HOST24 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-p1 12 -p2 2 -b ../benchmarks/bwunicluster/gpu4/forward --opt 1" --mpi_params "--mca btl_openib_warn_default_gid_prefix 0"
echo "Pencil Opt1 Inverse"
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json --hosts $HOST24 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-t 2 -p1 12 -p2 2 -b ../benchmarks/bwunicluster/gpu4/inverse --opt 1" --mpi_params "--mca btl_openib_warn_default_gid_prefix 0"


echo "all done"