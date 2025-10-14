#!/bin/bash

#SBATCH -J example #（作业名称）

#SBATCH -p cnall #（使用cnall队列）

#SBATCH -N 1 # (使用1个节点)

#SBATCH -o stdout.%j #（屏幕输出）

#SBATCH -e stderr.%j #（错误输出）

#SBATCH --ntasks-per-node=56 #（每个节点占用的核数）

# 替换为自己的miniconda3路径
source /home/sccomp/miniconda3/etc/profile.d/conda.sh
# 替换为自己的alphafold3环境名称
conda activate af3

# 输入和参数位置，改为自己的路径
python run_alphafold.py --json_path=/home/sccomp/WORK/alphafold3/xfold/processed/37aa_2JO9.json --model_dir=/home/sccomp/WORK/alphafold3/weights --norun_data_pipeline --output_dir=output --fastnn=False