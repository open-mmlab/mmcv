## 使用 S3 的数据训练模型

本教程的目的是为了展示如何使用 Amazon Simple Storage Service (Amazon S3) 作为存储后端进行模型的训练与存储。

> 该特性从 v1.4.7 版本开始支持

### AWS 配置

如果你从未使用过 AWS 服务，你需要先注册 [AWS账号](https://aws.amazon.com/) 并配置好 [S3存储桶](https://docs.aws.amazon.com/AmazonS3/latest/userguide/GetStartedWithS3.html)。
为了支持在你的个人电脑对AWS服务进行访问，你需要安装AWS提供的awscli和python sdk `boto3`。具体步骤如下:
- 安装依赖库

```bash
pip install awscli
pip install boto3
```

- 配置 aws

```bash
# configure your aws credentials
# Note that region must match your bucket region.
aws configure
# AWS Access Key ID [****************JQGK]:
# AWS Secret Access Key [****************3gUp]:
# Default region name [ap-east-1]:
# Default output format [json]:
```

### 读写 S3 数据

```python
import mmcv

img_path = 'tests/data/color.jpg'
with open(img_path, 'rb') as f:
    img_buff = f.read()

s3_path = 's3://yourbucket/demo/img.jpg'
# 根据文件前缀自动推理处理的文件后端
file_client = mmcv.FileClient.infer_client(uri=s3_path)

# 使用 backend 参数指定处理的文件后端
# file_client = mmcv.FileClient(backend='aws')

# 使用 prefix 参数指定处理的路径前缀，FileClient 根据前缀选择对应的文件后端
# file_client = mmcv.FileClient(prefix='s3')

# 将图像写入 s3
file_client.put(img_buff, s3_path)

# 从 s3 读取图像
img_buff_with_s3 = file_client.get(s3_path).tobytes()
assert img_buff == img_buff_with_s3

# 删除 s3 路径的文件
file_client.remove(s3_path)

# 列出指定目录的子目录和文件
print(list(file_client.list_dir_or_file('s3://yourbucket')))
```

### 以 MMDetection 为例

通过 config 文件来实现本地数据存储到 S3 数据存储的替换。完整配置文件在文末，需要用户在个人 Amazon S3 上传对应的数据即可实现以 Amazon S3 作为文件后端训练模型。

#### 从 S3 读取 pretrain

```python
model = dict(
    type='FasterRCNN',
    backbone=dict(
        type='ResNet',
        ...
        init_cfg=dict(type='Pretrained', checkpoint='s3://yourbucket/demo/pretrained/resnet50_batch256_imagenet_20200708-cfb998bf.pth')),
```

#### 从 S3 读取 checkpoint

```python
load_from = 's3://yourbucket/demo/pretrained/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
resume_from = None
workflow = [('train', 1)]
```

#### Checkpoint 保存至 S3

```python
# checkpoint saving
checkpoint_config = dict(interval=1, max_keep_ckpts=2, out_dir='s3://yourbucket/demo/ckpt/')  # yapf:disable
```

#### EvalHook 保存最优 checkpoint 至 S3

```python
evaluation = dict(interval=1, save_best='bbox', out_dir='s3://yourbucket/demo/ckpt/')
```

#### 训练日志保存至 S3

训练日志会在训练结束后备份至指定的路径
```python
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', out_dir='s3://yourbucket/demo/logs/'),
        # dict(type='TensorboardLoggerHook')
    ])  # yapf:enable
```

也可以通过设置 `keep_local = False`，则会在备份至指定的 S3 路径后删除本地的训练日志
```python
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', out_dir='s3://yourbucket/demo/logs/', keep_local=False),
        # dict(type='TensorboardLoggerHook')
    ])# yapf:enable
```

#### 读取 S3 的训练数据

在不改动 MMDetection 原有 config 的情况下，可以通过解析标签文件获得数据路径，在读取数据的时候将本地路径映射为 S3 路径即可。

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

#### 使用 S3 上的数据训练 faster_rcnn

配置文件由 MMDetection 中的 [faster_rcnn_r50_fpn_1x_coco.py](https://github.com/open-mmlab/mmdetection/blob/master/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py) 修改得到，主要展示了如何通过修改配置文件实现使用 S3 的数据训练模型。

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
