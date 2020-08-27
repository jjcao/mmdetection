_base_ = ['smpr_fcos_r50_caffe_fpn_bn_1x_coco.py']

model = dict(
    bbox_head=dict(
        loss_heatmap=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=4.0)
)