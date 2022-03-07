
shcedule_times = 4
lr_times = 10

# optimizer
optimizer = dict(type='SGD', lr=0.02*lr_times, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[4*shcedule_times, 8*shcedule_times, 10*shcedule_times, 11*shcedule_times])
runner = dict(type='EpochBasedRunner', max_epochs=12*shcedule_times)
