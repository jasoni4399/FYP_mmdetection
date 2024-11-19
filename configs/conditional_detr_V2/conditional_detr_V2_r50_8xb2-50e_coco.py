_base_ = ['../detr/detr_r50_8xb2-150e_coco.py']
model = dict(
    type='ConditionalDETR_V2',
    num_queries=300,
    content_width=0.4,
    content_height=0.4,
    decoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(
                _delete_=True,
                embed_dims=256,
                num_heads=8,
                attn_drop=0.1,
                cross_attn=False),
            cross_attn_cfg=dict(
                _delete_=True,
                embed_dims=256,
                num_heads=8,
                attn_drop=0.1,
                cross_attn=True))),
    bbox_head=dict(
        type='ConditionalDETRHead',
        loss_cls=dict(
            _delete_=True,
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])))


log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(type='JsonLoggerHook'),
        dict(type='FileLoggerHook', log_dir='./work_dirs/logs/')
    ])

# learning policy
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=18, val_interval=1)

param_scheduler = [dict(type='MultiStepLR', end=18, milestones=[17])]

auto_scale_lr = dict(enable=True, base_batch_size=32)
