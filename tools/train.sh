#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2,3

#python tools/train.py configs/smpr/smpr_r50_caffe_fpn_bn_1x_coco.py

CONFIG=$1
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/train.py $CONFIG ${@:3}