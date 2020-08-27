_base_ = ['smpr_fcos_r50_caffe_fpn_bn_1x_coco.py']

model = dict(
    bbox_head=dict(
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)
)
