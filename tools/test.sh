export CUDA_VISIBLE_DEVICES=2,3

python tools/test.py configs/smpr/smpr_fcos_r50_caffe_fpn_bn_1x_coco.py \
    work_dirs/smpr/epoch_1.pth \
    --show-dir smpr_fcos_r50_caffe_fpn_bn_1x_results