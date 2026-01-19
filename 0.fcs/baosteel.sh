#!/bin/bash

# 确保在 0.fcs 目录下运行
echo "Running FCS experiment for Baosteel dataset..."

# 第一阶段：初始任务训练 (Task 0)
echo "Starting Stage 1..."
python main.py --config=./exps/fcs/baosteel/5/first_stage.json

# 第二阶段：增量任务训练 (Task 1+)
# 注意：second_stage.json 中应正确配置 "ckpt_path" 指向 first_stage 生成的日志目录
echo "Starting Stage 2..."
python main.py --config=./exps/fcs/baosteel/5/second_stage.json

echo "Baosteel experiment finished."