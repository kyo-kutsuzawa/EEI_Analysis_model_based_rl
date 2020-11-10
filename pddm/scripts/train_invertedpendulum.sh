#!/bin/sh
python ~/Documents/pddm/scripts/train.py --config ../config_furuta_fs1/furuta_inverted_pendulum_rand.txt --output_dir ../output/ip --use_gpu --gpu_frac=0.2
python ~/Documents/pddm/scripts/train.py --config ../config_furuta_fs1/furuta_inverted_pendulum_mppi.txt --output_dir ../output/ip --use_gpu --gpu_frac=0.2
