#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1,3
CONFIG=$1
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/train.py $CONFIG 
#${@:3}