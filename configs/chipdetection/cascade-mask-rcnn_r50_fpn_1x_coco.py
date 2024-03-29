# from mmengine.optim.scheduler.lr_scheduler import LinearLR, MultiStepLR
_base_ = [
    '../_base_/models/cascade-mask-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model =dict(
    rpn_head=dict(
        anchor_generator=dict(
        type='AnchorGenerator',
        scales=[2],
        ratios=[1.0],
        strides=[4, 8, 16, 32, 512])
    ),
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=2,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=2,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=2,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ],
        mask_head=dict(
            num_classes=2
        )
    )

)


# Modify dataset related settings
data_root = '/root/autodl-tmp/chip_barcode_project/chip_img_sr_cocodetection/'
metainfo = {
    'classes': ('chip','hole', ),
    'palette': [
        (220, 20, 60),
        (20,220,60),
    ]
}

train_pipeline = [
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(2560, 1920), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Mosaic',img_scale=(2560,1920)),
    dict(type="PhotoMetricDistortion"),
    dict(type="YOLOXHSVRandomAug"),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='Resize', scale=(2560, 1920), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]





train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        data_root=data_root+"train_coco/",
        metainfo=metainfo,
        ann_file='train.json',
        data_prefix=dict(img='')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root+"val_coco",
        metainfo=metainfo,
        ann_file='val.json',
        data_prefix=dict(img='')))
test_dataloader = val_dataloader


# Modify metric related settings
val_evaluator = dict(ann_file=data_root + 'val_coco/val.json')
test_evaluator = val_evaluator

train_cfg = dict(max_epochs=50, val_interval=1)
# # learning rate
# param_scheduler = [
#     dict(type=LinearLR, start_factor=0.001, by_epoch=False, begin=0, end=500),
#     dict(
#         type=MultiStepLR,
#         begin=0,
#         end=50,
#         by_epoch=True,
#         milestones=[40, 45],
#         gamma=0.1)
# ]

#load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
load_from ="/root/autodl-tmp/cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth"