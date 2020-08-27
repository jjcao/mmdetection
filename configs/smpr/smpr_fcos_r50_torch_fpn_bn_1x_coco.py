# Compared with using 'open-mmlab://detectron/resnet50_caffe'
# 1. the loss is too big
# 2. model size is also bigger
# so it is abandoned. 
_base_ = ['smpr_fcos_r50_caffe_fpn_bn_1x_coco.py'
]

model = dict(
    pretrained='torchvision://resnet50'
)

# dataset settings
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4)

# lr=0.001 for batchsize =8
optimizer = dict(type='SGD', lr=0.001)
