#!/bin/bash

#SBATCH -J example       # 作业名称
#SBATCH -p cnmix         # 使用队列
#SBATCH -N 1             # 使用节点数
#SBATCH -o stdout.%j      # 标准输出
#SBATCH -e stderr.%j      # 错误输出
#SBATCH --ntasks-per-node=56  # 每个节点使用的核数

# 加载环境
source ~/.bashrc
conda activate af3       # 替换为自己的alphafold3环境名称

# ================================
# 修改为自己的路径
INPUT_JSON=/home/sccomp/WORK/alphafold3/xfold/processed/37aa_2JO9.json
MODEL_DIR=/WORK/sccomp/weights # 无需替换，这是公共可读目录
OUTPUT_DIR=output

# 运行脚本
python run_alphafold.py \
    --json_path=$INPUT_JSON \
    --model_dir=$MODEL_DIR \
    --norun_data_pipeline \
    --output_dir=$OUTPUT_DIR \
    --fastnn=False

