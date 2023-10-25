default_scope = 'mmrotate'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=100),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='RotLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ],
    name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False
custom_hooks = [
    dict(type='mmdet.NumClassCheckHook'),
    dict(
        type='EMAHook',
        ema_type='mmdet.ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
]
max_epochs = 3000
base_lr = 0.00025
interval = 10
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=3000, val_interval=10)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-05, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=4.1666666666666667e-07,
        begin=90,
        end=3000,
        T_max=2910,
        by_epoch=True,
        convert_to_iter_based=True),
]
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.00025, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))
dataset_type = 'DOTADataset'
data_root = 'container_dataset/'
backend_args = None
train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=None),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.RandomResize',
        resize_type='mmdet.Resize',
        scale=(
            1024,
            1024,
        ),
        ratio_range=(
            3.0,
            5.0,
        ),
        keep_ratio=True),
    dict(
        type='RandomRotate',
        prob=0.5,
        angle_range=180,
        rect_obj_labels=[
            9,
            11,
        ]),
    dict(type='mmdet.RandomCrop', crop_size=(
        1024,
        1024,
    )),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=[
            'horizontal',
            'vertical',
            'diagonal',
        ]),
    dict(
        type='mmdet.Pad',
        size=(
            1024,
            1024,
        ),
        pad_val=dict(img=(
            114,
            114,
            114,
        ))),
    dict(type='mmdet.PackDetInputs'),
]
val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=None),
    dict(type='mmdet.Resize', scale=(
        1024,
        1024,
    ), keep_ratio=True),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.Pad',
        size=(
            1024,
            1024,
        ),
        pad_val=dict(img=(
            114,
            114,
            114,
        ))),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        )),
]
test_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=None),
    dict(type='mmdet.Resize', scale=(
        1024,
        1024,
    ), keep_ratio=True),
    dict(
        type='mmdet.Pad',
        size=(
            1024,
            1024,
        ),
        pad_val=dict(img=(
            114,
            114,
            114,
        ))),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        )),
]
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    pin_memory=False,
    dataset=dict(
        type='DOTADataset',
        data_root='container_dataset/',
        ann_file='train_lbl',
        data_prefix=dict(img_path='train_images'),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=[
            dict(type='mmdet.LoadImageFromFile', backend_args=None),
            dict(
                type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
            dict(
                type='ConvertBoxType',
                box_type_mapping=dict(gt_bboxes='rbox')),
            dict(
                type='mmdet.RandomResize',
                resize_type='mmdet.Resize',
                scale=(
                    1024,
                    1024,
                ),
                ratio_range=(
                    3.0,
                    5.0,
                ),
                keep_ratio=True),
            dict(
                type='RandomRotate',
                prob=0.5,
                angle_range=180,
                rect_obj_labels=[
                    9,
                    11,
                ]),
            dict(type='mmdet.RandomCrop', crop_size=(
                1024,
                1024,
            )),
            dict(type='mmdet.YOLOXHSVRandomAug'),
            dict(
                type='mmdet.RandomFlip',
                prob=0.75,
                direction=[
                    'horizontal',
                    'vertical',
                    'diagonal',
                ]),
            dict(
                type='mmdet.Pad',
                size=(
                    1024,
                    1024,
                ),
                pad_val=dict(img=(
                    114,
                    114,
                    114,
                ))),
            dict(type='mmdet.PackDetInputs'),
        ],
        metainfo=dict(classes=('container', ), palette=[
            (
                165,
                42,
                42,
            ),
        ])))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='DOTADataset',
        data_root='container_dataset/',
        ann_file='train_VAL_patch_lbl',
        data_prefix=dict(img_path='train_VAL_patch_img'),
        test_mode=True,
        pipeline=[
            dict(type='mmdet.LoadImageFromFile', backend_args=None),
            dict(type='mmdet.Resize', scale=(
                1024,
                1024,
            ), keep_ratio=True),
            dict(
                type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
            dict(
                type='ConvertBoxType',
                box_type_mapping=dict(gt_bboxes='rbox')),
            dict(
                type='mmdet.Pad',
                size=(
                    1024,
                    1024,
                ),
                pad_val=dict(img=(
                    114,
                    114,
                    114,
                ))),
            dict(
                type='mmdet.PackDetInputs',
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                )),
        ],
        metainfo=dict(classes=('container', ), palette=[
            (
                165,
                42,
                42,
            ),
        ])))
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='DOTADataset',
        data_root='container_dataset/',
        ann_file='trainval/annfiles/',
        data_prefix=dict(img_path='trainval/images/'),
        test_mode=True,
        pipeline=[
            dict(type='mmdet.LoadImageFromFile', backend_args=None),
            dict(type='mmdet.Resize', scale=(
                1024,
                1024,
            ), keep_ratio=True),
            dict(
                type='mmdet.Pad',
                size=(
                    1024,
                    1024,
                ),
                pad_val=dict(img=(
                    114,
                    114,
                    114,
                ))),
            dict(
                type='mmdet.PackDetInputs',
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                )),
        ],
        metainfo=dict(classes=('container', ), palette=[
            (
                165,
                42,
                42,
            ),
        ])))
val_evaluator = dict(type='DOTAMetric', metric='mAP')
test_evaluator = dict(type='DOTAMetric', metric='mAP')
checkpoint = ''
angle_version = 'le90'
model = dict(
    type='mmdet.RTMDet',
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[
            103.53,
            116.28,
            123.675,
        ],
        std=[
            57.375,
            57.12,
            58.395,
        ],
        bgr_to_rgb=False,
        boxtype2tensor=False,
        batch_augments=None),
    backbone=dict(
        type='mmdet.CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=1,
        widen_factor=1,
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=None),
    neck=dict(
        type='mmdet.CSPNeXtPAFPN',
        in_channels=[
            256,
            512,
            1024,
        ],
        out_channels=256,
        num_csp_blocks=3,
        expand_ratio=0.5,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=None),
    bbox_head=dict(
        type='RotatedRTMDetSepBNHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=2,
        feat_channels=256,
        angle_version='le90',
        anchor_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0, strides=[
                8,
                16,
                32,
            ]),
        bbox_coder=dict(type='DistanceAnglePointCoder', angle_version='le90'),
        loss_cls=dict(
            type='mmdet.QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='RotatedIoULoss', mode='linear', loss_weight=2.0),
        with_objectness=False,
        exp_on_reg=True,
        share_conv=True,
        pred_kernel_size=1,
        use_hbbox_loss=False,
        scale_angle=False,
        loss_angle=None,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=None),
    train_cfg=dict(
        assigner=dict(
            type='mmdet.DynamicSoftLabelAssigner',
            iou_calculator=dict(type='RBboxOverlaps2D'),
            topk=13),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms_rotated', iou_threshold=0.1),
        max_per_img=2000),
    init_cfg=None)
coco_ckpt = ''
work_dir = './VER1'
seed = 199002
device = 'cuda'
evaluation = dict(save_best='mAP')
