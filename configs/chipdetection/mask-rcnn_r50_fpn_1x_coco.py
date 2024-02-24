_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]


model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=2), mask_head=dict(num_classes=2)))


# Modify dataset related settings
data_root = '/root/autodl-tmp/chip_barcode_project/chip_img_sr_cocodetection/'
metainfo = {
    'classes': ('chip','hole', ),
    'palette': [
        (220, 20, 60),
        (20,220,60),
    ]
}
train_dataloader = dict(
    batch_size=1,
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


#load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
load_from ="/root/autodl-tmp/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth"


