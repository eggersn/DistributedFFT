#!/bin/bash
#SBATCH -p gpu_4
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --time=15:00:00
#SBATCH --job-name=gpu4_pencil_small_cuda
#SBATCH --output=gpu4_pencil_small_cuda.%j.out
#SBATCH --account=st

# load modules
module load compiler/gnu/8.3.1
module load devel/cuda/11.0
module load devel/cmake/3.18
module load mpi/openmpi/4.1
echo "Modules loaded"

# determine hosts
HOSTS="$(mpirun hostname | sort -n | sed -r 's/\.localdomain//')"
echo "$HOSTS"
HOST16="$(echo "$HOSTS" | tr '\n' ',' | sed 's/,$//' | sed 's/,/ /g')"
echo "16: $HOST16"

# build
echo "start building"
cd $HOME/DistributedFFT/
rm -rf build_gpu4
mkdir build_gpu4
cd build_gpu4

cmake ..
cmake --build .
echo "finished building"

sleep 5
cd ..


echo "*****************************************************************************"
echo "Starting on HOST16"
echo "*****************************************************************************"
echo "-----------------------------------------------------------------------------"
echo "Partition 2x8 / 4x4"
echo "-----------------------------------------------------------------------------"
echo "Pencil Default"
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json bwunicluster/pencil/validation.json --hosts $HOST16 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -p1 2 -p2 8 -b ../benchmarks/bwunicluster/gpu4/forward"
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json bwunicluster/pencil/validation.json --hosts $HOST16 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -p1 4 -p2 4 -b ../benchmarks/bwunicluster/gpu4/forward"
echo "Pencil Opt1"
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json bwunicluster/pencil/validation.json --hosts $HOST16 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -p1 2 -p2 8 -b ../benchmarks/bwunicluster/gpu4/forward --opt 1"
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json bwunicluster/pencil/validation.json --hosts $HOST16 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -p1 4 -p2 4 -b ../benchmarks/bwunicluster/gpu4/forward --opt 1"
echo "Pencil Default Inverse"
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json --hosts $HOST16 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -t 2 -p1 2 -p2 8 -b ../benchmarks/bwunicluster/gpu4/inverse"
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json --hosts $HOST16 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -t 2 -p1 4 -p2 4 -b ../benchmarks/bwunicluster/gpu4/inverse"
echo "Pencil Opt1 Inverse"
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json --hosts $HOST16 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -t 2 -p1 2 -p2 8 -b ../benchmarks/bwunicluster/gpu4/inverse --opt 1"
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json --hosts $HOST16 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -t 2 -p1 4 -p2 4 -b ../benchmarks/bwunicluster/gpu4/inverse --opt 1"
echo "-----------------------------------------------------------------------------"
echo "Partition 8x2"
echo "-----------------------------------------------------------------------------"
echo "Pencil Default"
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json bwunicluster/pencil/validation.json --hosts $HOST16 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -p1 8 -p2 2 -b ../benchmarks/bwunicluster/gpu4/forward" --mpi_params "--mca btl_openib_warn_default_gid_prefix 0" 
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json bwunicluster/pencil/validation.json --hosts $HOST16 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -p1 8 -p2 2 -b ../benchmarks/bwunicluster/gpu4/forward --opt 1" --mpi_params "--mca btl_openib_warn_default_gid_prefix 0" 
echo "Pencil Default Inverse"
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json --hosts $HOST16 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -t 2 -p1 8 -p2 2 -b ../benchmarks/bwunicluster/gpu4/inverse" --mpi_params "--mca btl_openib_warn_default_gid_prefix 0" 
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json --hosts $HOST16 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -t 2 -p1 8 -p2 2 -b ../benchmarks/bwunicluster/gpu4/inverse --opt 1" --mpi_params "--mca btl_openib_warn_default_gid_prefix 0" 

echo "all done"