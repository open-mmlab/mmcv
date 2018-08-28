# model settings
model = 'resnet18'
# dataset settings
data_root = '/mnt/SSD/dataset/cifar10'
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]
batch_size = 64

# optimizer and learning rate
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=5e-4)
lr_policy = dict(policy='step', step=2)

# runtime settings
work_dir = './demo'
gpus = range(2)
data_workers = 2  # data workers per gpu
checkpoint_cfg = dict(interval=1)  # save checkpoint at every epoch
workflow = [('train', 1), ('val', 1)]
max_epoch = 6
resume_from = None
load_from = None

# logging settings
log_level = 'INFO'
log_cfg = dict(
    # log at every 50 iterations
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook', log_dir=work_dir + '/log'),
    ])
