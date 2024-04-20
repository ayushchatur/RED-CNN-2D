#!/bin/bash

#SBATCH --job-name=test_red_2d
#SBATCH --partition=dgx_normal_q
#SBATCH --time=1:00:00
#SBATCH -A HPCBIGDATA2
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48

module reset
module load Anaconda3/2020.11

export load_mode=1
export mode='test'
export test_iters=1000
export batch_size=32

source activate env/tc/cpu/py39_base
python3 main.py --load_mode $load_mode --mode $mode