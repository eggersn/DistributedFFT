#!/bin/bash
#SBATCH -p "gpu_8"
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --ntasks 16
#SBATCH --nodes 2
#SBATCH --ntasks-per-node 8
#SBATCH --time=00:30:00
#SBATCH --job-name=gpu8_slab
#SBATCH --output=gpu8_slab.%j.out

# load modules
module load compiler/gnu/8.3.1
module load devel/cuda/11.0
module load devel/cmake/3.18
module load mpi/openmpi/4.1

# determine hosts
HOSTS=`mpirun -n 16 hostname | sort -n | sed -r "s/\.localdomain//"`

HOST16=`echo $HOSTS | tr "\n" "," | sed "s/,$//" | sed "s/,/ /g"`
HOST8x0=`echo $HOSTS | head -n 8 | tr "\n" "," | sed "s/,$//" | sed "s/,/ /g"`
HOST8x1=`echo $HOSTS | tail -n 8 | tr "\n" "," | sed "s/,$//" | sed "s/,/ /g"`

# build
cd /home/st/st_us-051200/st_st160727/DistributedFFT/
rm -rf build 
mkdir build
cd build

cmake ..
cmake --build .

sleep 5
cd ..

# start python script
python launch.py --jobs bwunicluster/slab/zy_then_x.json --hosts $HOST16 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 
python launch.py --jobs bwunicluster/slab/zy_then_x.json --hosts $HOST8x0 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --global_params "-p 8" --id 1 & 
python launch.py --jobs bwunicluster/slab/zy_then_x.json --hosts $HOST8x1 --gpus 8 --affinity 0:0-9 0:0-9 0:0-9 0:0-9 1:0-9 1:0-9 1:0-9 1:0-9 --global_params "-p 8 --opt 1" --id 2 &  

wait

