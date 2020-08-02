_base_ = ['smpr_fcos_r50_torch_fpn_bn_1x_coco.py'
]

# dataset settings
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4)
