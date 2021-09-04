#!/bin/bash
#SBATCH -p gpu_4
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --nodes=8
#SBATCH --ntasks=32
#SBATCH --time=40:00:00
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
HOST32="$(echo "$HOSTS" | tr '\n' ',' | sed 's/,$//' | sed 's/,/ /g')"
echo "32: $HOST32"
HOST24="$(echo "$HOSTS" | head -n 24 | tr '\n' ',' | sed 's/,$//' | sed 's/,/ /g')"
echo "24: $HOST24"
HOST16x0="$(echo "$HOSTS" | head -n 16 | tr '\n' ',' | sed 's/,$//' | sed 's/,/ /g')"
echo "16x0: $HOST16x0"
HOST16x1="$(echo "$HOSTS" | tail -n 16 | tr '\n' ',' | sed 's/,$//' | sed 's/,/ /g')"
echo "16x1: $HOST16x1"
HOST8="$(echo "$HOSTS" | tail -n 8 | tr '\n' ',' | sed 's/,$//' | sed 's/,/ /g')"
echo "8: $HOST8"
HOST4="$(echo "$HOSTS" | tail -n 4 | tr '\n' ',' | sed 's/,$//' | sed 's/,/ /g')"
echo "4: $HOST4"

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
echo "Starting on HOST32"
echo "*****************************************************************************"
echo "-----------------------------------------------------------------------------"
echo "Partition 4x8"
echo "-----------------------------------------------------------------------------"
echo "Pencil Default"
python launch.py --jobs bwunicluster/pencil/benchmarks_base1.json bwunicluster/pencil/validation.json --hosts $HOST32 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -p1 4 -p2 8 -b ../benchmarks/bwunicluster/gpu4/forward"
echo "Pencil Opt1"
python launch.py --jobs bwunicluster/pencil/benchmarks_base2.json bwunicluster/pencil/validation.json --hosts $HOST32 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -p1 4 -p2 8 -b ../benchmarks/bwunicluster/gpu4/forward --opt 1"
echo "Pencil Default Inverse"
python launch.py --jobs bwunicluster/pencil/benchmarks_base2.json --hosts $HOST32 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -t 2 -p1 4 -p2 8 -b ../benchmarks/bwunicluster/gpu4/inverse"
echo "Pencil Opt1 Inverse"
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json --hosts $HOST32 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -t 2 -p1 4 -p2 8 -b ../benchmarks/bwunicluster/gpu4/inverse --opt 1"
echo "-----------------------------------------------------------------------------"
echo "Partition 8x4"
echo "-----------------------------------------------------------------------------"
echo "Pencil Default"
python launch.py --jobs bwunicluster/pencil/benchmarks_base1.json bwunicluster/pencil/validation.json --hosts $HOST32 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -p1 8 -p2 4 -b ../benchmarks/bwunicluster/gpu4/forward"
echo "Pencil Opt1"
python launch.py --jobs bwunicluster/pencil/benchmarks_base2.json bwunicluster/pencil/validation.json --hosts $HOST32 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -p1 8 -p2 4 -b ../benchmarks/bwunicluster/gpu4/forward --opt 1"
echo "Pencil Default Inverse"
python launch.py --jobs bwunicluster/pencil/benchmarks_base2.json --hosts $HOST32 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -t 2 -p1 8 -p2 4 -b ../benchmarks/bwunicluster/gpu4/inverse"
echo "Pencil Opt1 Inverse"
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json --hosts $HOST32 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -t 2 -p1 8 -p2 4 -b ../benchmarks/bwunicluster/gpu4/inverse --opt 1"
echo "-----------------------------------------------------------------------------"
echo "Partition 16x2"
echo "-----------------------------------------------------------------------------"
echo "Pencil Default"
python launch.py --jobs bwunicluster/pencil/benchmarks_base1.json bwunicluster/pencil/validation.json --hosts $HOST32 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -p1 16 -p2 2 -b ../benchmarks/bwunicluster/gpu4/forward" --mpi_params "--mca btl_openib_warn_default_gid_prefix 0"
echo "Pencil Opt1"
python launch.py --jobs bwunicluster/pencil/benchmarks_base2.json bwunicluster/pencil/validation.json --hosts $HOST32 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -p1 16 -p2 2 -b ../benchmarks/bwunicluster/gpu4/forward --opt 1" --mpi_params "--mca btl_openib_warn_default_gid_prefix 0"
echo "Pencil Default Inverse"
python launch.py --jobs bwunicluster/pencil/benchmarks_base2.json --hosts $HOST32 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -t 2 -p1 16 -p2 2 -b ../benchmarks/bwunicluster/gpu4/inverse" --mpi_params "--mca btl_openib_warn_default_gid_prefix 0"
echo "Pencil Opt1 Inverse"
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json --hosts $HOST32 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -t 2 -p1 16 -p2 2 -b ../benchmarks/bwunicluster/gpu4/inverse --opt 1" --mpi_params "--mca btl_openib_warn_default_gid_prefix 0"


echo "*****************************************************************************"
echo "Starting on HOST24"
echo "*****************************************************************************"
echo "-----------------------------------------------------------------------------"
echo "Partition 3x8 / 2x4"
echo "-----------------------------------------------------------------------------"
echo "Pencil Default"
python launch.py --jobs bwunicluster/pencil/benchmarks_base1.json bwunicluster/pencil/validation.json --hosts $HOST24 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -p1 3 -p2 8 -b ../benchmarks/bwunicluster/gpu4/forward" --id 1 &
python launch.py --jobs bwunicluster/pencil/benchmarks_base1.json bwunicluster/pencil/validation.json --hosts $HOST8 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -p1 2 -p2 4 -b ../benchmarks/bwunicluster/gpu4/forward" --id 2 &
wait
echo "Pencil Opt1"
python launch.py --jobs bwunicluster/pencil/benchmarks_base2.json bwunicluster/pencil/validation.json --hosts $HOST24 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -p1 3 -p2 8 -b ../benchmarks/bwunicluster/gpu4/forward --opt 1" --id 1 &
python launch.py --jobs bwunicluster/pencil/benchmarks_base2.json bwunicluster/pencil/validation.json --hosts $HOST8 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -p1 2 -p2 4 -b ../benchmarks/bwunicluster/gpu4/forward --opt 1" --id 2 &
wait
echo "Pencil Default Inverse"
python launch.py --jobs bwunicluster/pencil/benchmarks_base2.json --hosts $HOST24 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -t 2 -p1 3 -p2 8 -b ../benchmarks/bwunicluster/gpu4/inverse" --id 1 &
python launch.py --jobs bwunicluster/pencil/benchmarks_base2.json --hosts $HOST8 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -t 2 -p1 2 -p2 4 -b ../benchmarks/bwunicluster/gpu4/inverse" --id 2 &
wait
echo "Pencil Opt1 Inverse"
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json --hosts $HOST24 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -t 2 -p1 3 -p2 8 -b ../benchmarks/bwunicluster/gpu4/inverse --opt 1" --id 1 &
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json --hosts $HOST8 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -t 2 -p1 2 -p2 4 -b ../benchmarks/bwunicluster/gpu4/inverse --opt 1" --id 2 &
wait

echo "-----------------------------------------------------------------------------"
echo "Partition 6x4 / 4x2"
echo "-----------------------------------------------------------------------------"
python launch.py --jobs bwunicluster/pencil/benchmarks_base1.json bwunicluster/pencil/validation.json --hosts $HOST24 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -p1 6 -p2 4 -b ../benchmarks/bwunicluster/gpu4/forward" --id 1 &
python launch.py --jobs bwunicluster/pencil/benchmarks_base1.json bwunicluster/pencil/validation.json --hosts $HOST8 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -p1 4 -p2 2 -b ../benchmarks/bwunicluster/gpu4/forward" --id 2 --mpi_params "--mca btl_openib_warn_default_gid_prefix 0" &
wait
echo "Pencil Opt1"
python launch.py --jobs bwunicluster/pencil/benchmarks_base2.json bwunicluster/pencil/validation.json --hosts $HOST24 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -p1 6 -p2 4 -b ../benchmarks/bwunicluster/gpu4/forward --opt 1" --id 1 &
python launch.py --jobs bwunicluster/pencil/benchmarks_base2.json bwunicluster/pencil/validation.json --hosts $HOST8 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -p1 4 -p2 2 -b ../benchmarks/bwunicluster/gpu4/forward --opt 1" --id 2 --mpi_params "--mca btl_openib_warn_default_gid_prefix 0" &
wait
echo "Pencil Default Inverse"
python launch.py --jobs bwunicluster/pencil/benchmarks_base2.json --hosts $HOST24 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -t 2 -p1 6 -p2 4 -b ../benchmarks/bwunicluster/gpu4/inverse" --id 1 &
python launch.py --jobs bwunicluster/pencil/benchmarks_base2.json --hosts $HOST8 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -t 2 -p1 4 -p2 2 -b ../benchmarks/bwunicluster/gpu4/inverse" --id 2 --mpi_params "--mca btl_openib_warn_default_gid_prefix 0" &
wait
echo "Pencil Opt1 Inverse"
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json --hosts $HOST24 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -t 2 -p1 6 -p2 4 -b ../benchmarks/bwunicluster/gpu4/inverse --opt 1" --id 1 &
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json --hosts $HOST8 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -t 2 -p1 4 -p2 2 -b ../benchmarks/bwunicluster/gpu4/inverse --opt 1" --id 2 --mpi_params "--mca btl_openib_warn_default_gid_prefix 0" &
wait

echo "-----------------------------------------------------------------------------"
echo "Partition 12x2 / 2x2"
echo "-----------------------------------------------------------------------------"
echo "Pencil Default Forward"
python launch.py --jobs bwunicluster/pencil/benchmarks_base1.json bwunicluster/pencil/validation.json --hosts $HOST24 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -p1 12 -p2 2 -b ../benchmarks/bwunicluster/gpu4/forward" --id 1 --mpi_params "--mca btl_openib_warn_default_gid_prefix 0" &
python launch.py --jobs bwunicluster/pencil/benchmarks_base1.json bwunicluster/pencil/validation.json --hosts $HOST4 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -p1 2 -p2 2 -b ../benchmarks/bwunicluster/gpu4/forward" --id 2 --mpi_params "--mca btl_openib_warn_default_gid_prefix 0" &
wait
echo "Pencil Opt1"
python launch.py --jobs bwunicluster/pencil/benchmarks_base2.json bwunicluster/pencil/validation.json --hosts $HOST24 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -p1 12 -p2 2 -b ../benchmarks/bwunicluster/gpu4/forward --opt 1" --id 1 --mpi_params "--mca btl_openib_warn_default_gid_prefix 0" &
python launch.py --jobs bwunicluster/pencil/benchmarks_base2.json bwunicluster/pencil/validation.json --hosts $HOST4 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -p1 2 -p2 2 -b ../benchmarks/bwunicluster/gpu4/forward --opt 1" --id 2 --mpi_params "--mca btl_openib_warn_default_gid_prefix 0" &
wait
echo "Pencil Default Inverse"
python launch.py --jobs bwunicluster/pencil/benchmarks_base2.json --hosts $HOST24 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -t 2 -p1 12 -p2 2 -b ../benchmarks/bwunicluster/gpu4/inverse" --id 1 --mpi_params "--mca btl_openib_warn_default_gid_prefix 0" &
python launch.py --jobs bwunicluster/pencil/benchmarks_base2.json --hosts $HOST4 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -t 2 -p1 2 -p2 2 -b ../benchmarks/bwunicluster/gpu4/inverse" --id 2 --mpi_params "--mca btl_openib_warn_default_gid_prefix 0" &
wait
echo "Pencil Opt1 Inverse"
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json --hosts $HOST24 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -t 2 -p1 12 -p2 2 -b ../benchmarks/bwunicluster/gpu4/inverse --opt 1" --id 1 --mpi_params "--mca btl_openib_warn_default_gid_prefix 0" &
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json --hosts $HOST4 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -t 2 -p1 2 -p2 2 -b ../benchmarks/bwunicluster/gpu4/inverse --opt 1" --id 2 --mpi_params "--mca btl_openib_warn_default_gid_prefix 0" &
wait


echo "*****************************************************************************"
echo "Starting on HOST16"
echo "*****************************************************************************"
echo "-----------------------------------------------------------------------------"
echo "Partition 2x8 / 4x4"
echo "-----------------------------------------------------------------------------"
echo "Pencil Default"
python launch.py --jobs bwunicluster/pencil/benchmarks_base1.json bwunicluster/pencil/validation.json --hosts $HOST16x0 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -p1 2 -p2 8 -b ../benchmarks/bwunicluster/gpu4/forward" --id 1 &
python launch.py --jobs bwunicluster/pencil/benchmarks_base1.json bwunicluster/pencil/validation.json --hosts $HOST16x1 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -p1 4 -p2 4 -b ../benchmarks/bwunicluster/gpu4/forward" --id 2 &
wait
echo "Pencil Opt1"
python launch.py --jobs bwunicluster/pencil/benchmarks_base2.json bwunicluster/pencil/validation.json --hosts $HOST16x0 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -p1 2 -p2 8 -b ../benchmarks/bwunicluster/gpu4/forward --opt 1" --id 1 &
python launch.py --jobs bwunicluster/pencil/benchmarks_base2.json bwunicluster/pencil/validation.json --hosts $HOST16x1 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -p1 4 -p2 4 -b ../benchmarks/bwunicluster/gpu4/forward --opt 1" --id 2 &
wait
echo "Pencil Default Inverse"
python launch.py --jobs bwunicluster/pencil/benchmarks_base2.json --hosts $HOST16x0 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -t 2 -p1 2 -p2 8 -b ../benchmarks/bwunicluster/gpu4/inverse" --id 1 &
python launch.py --jobs bwunicluster/pencil/benchmarks_base2.json --hosts $HOST16x1 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -t 2 -p1 4 -p2 4 -b ../benchmarks/bwunicluster/gpu4/inverse" --id 2 &
wait
echo "Pencil Opt1 Inverse"
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json --hosts $HOST16x0 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -t 2 -p1 2 -p2 8 -b ../benchmarks/bwunicluster/gpu4/inverse --opt 1" --id 1 &
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json --hosts $HOST16x1 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -t 2 -p1 4 -p2 4 -b ../benchmarks/bwunicluster/gpu4/inverse --opt 1" --id 2 &
wait
echo "-----------------------------------------------------------------------------"
echo "Partition 8x2"
echo "-----------------------------------------------------------------------------"
echo "Pencil Default"
python launch.py --jobs bwunicluster/pencil/benchmarks_base1.json bwunicluster/pencil/validation.json --hosts $HOST16x0 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -p1 8 -p2 2 -b ../benchmarks/bwunicluster/gpu4/forward" --id 1 --mpi_params "--mca btl_openib_warn_default_gid_prefix 0" &
python launch.py --jobs bwunicluster/pencil/benchmarks_base2.json bwunicluster/pencil/validation.json --hosts $HOST16x1 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -p1 8 -p2 2 -b ../benchmarks/bwunicluster/gpu4/forward --opt 1" --id 2 --mpi_params "--mca btl_openib_warn_default_gid_prefix 0" &
wait
echo "Pencil Default Inverse"
python launch.py --jobs bwunicluster/pencil/benchmarks_base2.json --hosts $HOST16x0 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -t 2 -p1 8 -p2 2 -b ../benchmarks/bwunicluster/gpu4/inverse" --id 1 --mpi_params "--mca btl_openib_warn_default_gid_prefix 0" &
python launch.py --jobs bwunicluster/pencil/benchmarks_base0.json --hosts $HOST16x1 --gpus 4 --affinity 0:0-9 0:0-9 1:0-9 1:0-9 --build_dir "build_gpu4" --global_params "-c -t 2 -p1 8 -p2 2 -b ../benchmarks/bwunicluster/gpu4/inverse --opt 1" --id 2 --mpi_params "--mca btl_openib_warn_default_gid_prefix 0" &
wait

echo "all done"