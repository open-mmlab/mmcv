## How to use S3

The purpose of this tutorial is to show how to use Amazon Simple Storage Service (Amazon S3) as the storage backend for model training and storage.

> In v1.4.4 and later, AWSBackend is provided to support reading and writing data from s3.

### Installation

If you have never used AWS services, you need to register [AWS account](https://aws.amazon.com/) first and configure [S3 bucket](https://docs.aws.amazon.com/AmazonS3/latest/userguide/GetStartedWithS3.html).

In order to support access to AWS services on your PC, you need to install awscli and python sdk `boto3` provided by AWS. The specific steps are as follows:

- Install dependent packages

```bash
pip install awscli
pip install boto3
```

- Configure aws

```bash
# configure your aws credentials
# Note that region must match your bucket region.
aws configure
# AWS Access Key ID [****************JQGK]:
# AWS Secret Access Key [****************3gUp]:
# Default region name [ap-east-1]:
# Default output format [json]:
```

### Reading and writing data from S3

```python
import mmcv

img_path = 'tests/data/color.jpg'
with open(img_path, 'rb') as f:
    img_buff = f.read()

s3_path = 's3://yourbucket/demo/img.jpg'
# infer the file client according to uri automatically
file_client = mmcv.FileClient.infer_client(uri=s3_path)

# get the file client according to the backend parameter
# file_client = mmcv.FileClient(backend='aws')

# get the file client according to the prefix parameter
# file_client = mmcv.FileClient(prefix='s3')

# write image to S3
file_client.put(img_buff, s3_path)

# download image from S3
img_buff_with_s3 = file_client.get(s3_path).tobytes()
assert img_buff == img_buff_with_s3

# delete file in S3 path
file_client.remove(s3_path)

# lists subdirectories and files in the specified directory
print(list(file_client.list_dir_or_file('s3://yourbucket')))
```

### Example of mmdetection

You can use S3 by modifying the config file. Before using, you need to ensure that there is corresponding data in Amazon S3, which can be used as the file back-end training model. The complete config file is at the end of the text.

#### Reading pretrained model from S3

```python
model = dict(
    type='FasterRCNN',
    backbone=dict(
        type='ResNet',
        ...
        init_cfg=dict(type='Pretrained', checkpoint='s3://yourbucket/demo/pretrained/resnet50_batch256_imagenet_20200708-cfb998bf.pth')),
```

#### Reading checkpoint from S3

```python
load_from = 's3://yourbucket/demo/pretrained/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
resume_from = None
workflow = [('train', 1)]
```

#### Saving checkpoint to S3

```python
# checkpoint saving
checkpoint_config = dict(interval=1, max_keep_ckpts=2, out_dir='s3://yourbucket/demo/ckpt/')  # yapf:disable
```

#### Saving best checkpoint to S3

```python
evaluation = dict(interval=1, save_best='bbox', out_dir='s3://yourbucket/demo/ckpt/')
```

#### Saving logs to S3

The training log will be backed up to the specified s3 path after the training.

```python
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', out_dir='s3://yourbucket/demo/logs/'),
        # dict(type='TensorboardLoggerHook')
    ])  # yapf:enable
```

You can also set `keep_local=False`, the local training log will be deleted after being backed up to the specified S3 path.

```python
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', out_dir='s3://yourbucket/demo/logs/', keep_local=False),
        # dict(type='TensorboardLoggerHook')
    ])# yapf:enable
```

#### Reading dataset from S3

Without changing the original config of mmdetection, you can obtain the data path by parsing the label file, and map the local path to S3 path when reading the data.

```python
file_client_args = dict(
    backend='aws',
    path_mapping=dict({
        'data/coco/': 's3://yourbucket/demo/data/coco/',
        'data/coco/': 's3://yourbucket/demo/data/coco/'
    }))
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True,
    file_client_args=file_client_args),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
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
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline,
        file_client_args=file_client_args),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline,
        file_client_args=file_client_args),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline,
        file_client_args=file_client_args))
```

#### Training faster-rcnn with dataset from S3

The config file is created by [faster_rcnn_r50_fpn_1x_coco.py](https://github.com/open-mmlab/mmdetection/blob/master/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py) in mmdetection, which mainly shows how to modify the config file to use the S3 data to training the model.

```python
_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

file_client_args = dict(
    backend='aws',
    path_mapping=dict({
        'data/coco/': 's3://yourbucket/demo/data/coco/',
        'data/coco/': 's3://yourbucket/demo/data/coco/'
    }))
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        file_client_args=file_client_args),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
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
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(pipeline=train_pipeline, file_client_args=file_client_args),
    val=dict(pipeline=test_pipeline, file_client_args=file_client_args),
    test=dict(pipeline=test_pipeline, file_client_args=file_client_args))

evaluation = dict(
    interval=1, save_best='bbox', out_dir='s3://yourbucket/demo/ckpt/')

model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='s3://yourbucket/demo/pretrained/resnet50_batch256_imagenet_20200708-cfb998bf.pth'
        )))

# finetune
load_from = 's3://yourbucket/demo/pretrained/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# checkpoint saving
checkpoint_config = dict(
    interval=1, max_keep_ckpts=2, out_dir='s3://yourbucket/demo/ckpt/')

log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook', out_dir='s3://yourbucket/demo/logs/'),
        # dict(type='TensorboardLoggerHook')
    ])  # yapf:enable

```
