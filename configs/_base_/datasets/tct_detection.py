dataset_type = 'CocoDataset'
data_root = 'data/coco/'

annotations_root = '../data/TCT_30000/'
image_root = '../data/TCT_30000/'
classes = ('normal', 'ascus', 'asch','lsil', 'hsil_scc_omn', 'agc_adenocarcinoma_em', 'vaginalis', 'monilia','dysbacteriosis_herpes_act', 'ec')


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=annotations_root + 'train30000-cat10.json',
        img_prefix=image_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=annotations_root + 'val10000-cat10.json',
        img_prefix=image_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=annotations_root + 'test10000-cat10.json',
        img_prefix=image_root,
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')