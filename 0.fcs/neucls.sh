#!/bin/bash

echo "Running FCS experiment for NEU-CLS dataset..."

# 第一阶段
echo "Starting Stage 1..."
python main.py --config=./exps/fcs/neucls/5/first_stage.json

# 第二阶段
echo "Starting Stage 2..."
python main.py --config=./exps/fcs/neucls/5/second_stage.json

echo "NEU-CLS experiment finished."