#!/bin/bash

#SBATCH --job-name=red_cnn
#SBATCH --nodes 1
#SBATCH --threads-per-core=1    # do not use hyperthreads (i.e. CPUs = physical cores below)
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8384                # total memory per node (4 GB per cpu-core is default)
#SBATCH --ntasks-per-node 1     # number of processors per node
#SBATCH --gpus-per-node 1             #GPU per node
#SBATCH --partition=dgx_normal_q # slurm partition
#SBATCH --time=24:30:00          # time limit
#SBATCH -A HPCBIGDATA2           # account name

module reset

module restore cu117


export load_mode=1
export mode='train'
export test_iters=1000
export batch_size=32

conda init
source ~/.bashrc
conda activate test
srun --unbuffered python main.py --load_mode $load_mode --mode $mode