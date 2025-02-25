import torch
import gc
gc.collect()
torch.cuda.empty_cache()
_base_ = ['../detr/detr_r50_8xb2-150e_coco.py']
model = dict(
    type='ConditionalDETR',
    num_queries=300,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        # Enable dilation in stage4 (last stage)
        dilations=(1, 1, 1, 2),  
        strides=(1, 2, 2, 1),    #Adjust strides to (1,2,2,1) for DC5
        out_indices=(3,),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
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

# learning policy
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=17, val_interval=1)

param_scheduler = [dict(type='MultiStepLR', end=17, milestones=[15])]

auto_scale_lr = dict(enable=True, base_batch_size=32)
