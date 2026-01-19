#!/bin/bash

echo "Running FCS experiment for PCB dataset..."

# 第一阶段
echo "Starting Stage 1..."
python main.py --config=./exps/fcs/pcb/5/first_stage.json

# 第二阶段
echo "Starting Stage 2..."
python main.py --config=./exps/fcs/pcb/5/second_stage.json

echo "PCB experiment finished."