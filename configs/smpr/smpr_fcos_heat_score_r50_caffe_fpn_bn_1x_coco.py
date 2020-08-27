_base_ = ['smpr_fcos_r50_caffe_fpn_bn_1x_coco.py']

model = dict(
    bbox_head=dict(
        loss_heatmap=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=4.0),
        loss_rescore=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0),
    )
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1
)
# optimizer
# lr=0.005 for batchsize =8
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001,
    paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.) )
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
    