_base_ = ['../_base_/datasets/coco_pose.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    type='SMPR', # The name of detector
    pretrained='open-mmlab://detectron/resnet50_caffe', 
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        style='caffe'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        extra_convs_on_inputs=False,  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='SMPRHead',
        num_classes=2,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_kpt_init=dict(type='KptSmoothL1Loss', beta=0.11, loss_weight=0.05),
        loss_kpt_refine=dict(type='KptSmoothL1Loss', beta=0.11, loss_weight=0.1), # question of jjcao, why not loss_weight=1.0
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        center_sampling=True,
        center_sample_radius=1.5)
)

# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxOKSAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.1,
    nms=dict(type='nms', iou_thr=0.5),
    max_per_img=20)

    
# optimizer
# lr=0.005 for batchsize =8
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001,
    paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.) )
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', warmup='constant',
    warmup_iters=500, warmup_ratio=1.0 / 3, step=[21, 24])

total_epochs = 25

# runtime settings
work_dir = './work_dirs/smpr'
