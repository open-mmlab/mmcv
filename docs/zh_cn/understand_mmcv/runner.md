## 执行器

执行器模块负责模型训练过程调度，主要目的是让用户使用更少的代码以及灵活可配置方式开启训练。其具备如下核心特性:

- 支持以 `EpochBasedRunner` 和 `IterBasedRunner` 为单位的迭代模式以满足不同场景
- 支持定制工作流以满足训练过程中各状态自由切换，目前支持训练和验证两个工作流。工作流可以简单理解为一个完成的训练和验证迭代过程。
- 配合各类默认和自定义 Hook，对外提供了灵活扩展能力

### EpochBasedRunner

顾名思义，`EpochBasedRunner` 是指以 epoch 为周期的工作流，例如设置 workflow = \[('train', 2), ('val', 1)\] 表示循环迭代地训练 2 个 epoch，然后验证 1 个 epoch。MMDetection 目标检测框架默认采用的是 `EpochBasedRunner`。

其抽象逻辑如下所示：

```python
# 训练终止条件
while curr_epoch < max_epochs:
    # 遍历用户设置的工作流，例如 workflow = [('train', 2)，('val', 1)]
    for i, flow in enumerate(workflow):
        # mode 是工作流函数，例如 train, epochs 是迭代次数
        mode, epochs = flow
        # 要么调用 self.train()，要么调用 self.val()
        epoch_runner = getattr(self, mode)
        # 运行对应工作流函数
        for _ in range(epochs):
            epoch_runner(data_loaders[i], **kwargs)
```

目前支持训练和验证两个工作流，以训练函数为例，其抽象逻辑是：

```python
# epoch_runner 目前可以是 train 或者 val
def train(self, data_loader, **kwargs):
    # 遍历 dataset，共返回一个 epoch 的 batch 数据
    for i, data_batch in enumerate(data_loader):
        self.call_hook('before_train_iter')
        # 验证时候 train_mode=False
        self.run_iter(data_batch, train_mode=True, **kwargs)
        self.call_hook('after_train_iter')
   self.call_hook('after_train_epoch')
```

### IterBasedRunner

不同于 `EpochBasedRunner`，`IterBasedRunner` 是指以 iter 为周期的工作流，例如设置 workflow = \[('train', 2)， ('val', 1)\] 表示循环迭代的训练 2 个 iter，然后验证 1 个 iter，MMSegmentation 语义分割框架默认采用的是  `IterBasedRunner`。

其抽象逻辑如下所示：

```python
# 虽然是 iter 单位，但是某些场合需要 epoch 信息，由 IterLoader 提供
iter_loaders = [IterLoader(x) for x in data_loaders]
# 训练终止条件
while curr_iter < max_iters:
    # 遍历用户设置的工作流，例如 workflow = [('train', 2)， ('val', 1)]
    for i, flow in enumerate(workflow):
        # mode 是工作流函数，例如 train, iters 是迭代次数
        mode, iters = flow
        # 要么调用 self.train()，要么调用 self.val()
        iter_runner = getattr(self, mode)
        # 运行对应工作流函数
        for _ in range(iters):
            iter_runner(iter_loaders[i], **kwargs)
```

目前支持训练和验证两个工作流，以验证函数为例，其抽象逻辑是：

```python
# iter_runner 目前可以是 train 或者 val
def val(self, data_loader, **kwargs):
    # 获取 batch 数据，用于一次迭代
    data_batch = next(data_loader)
    self.call_hook('before_val_iter')
    outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
    self.outputs = outputs
    self.call_hook('after_val_iter')
```

除了上述基础功能外，`EpochBasedRunner` 和 `IterBasedRunner` 还提供了 resume 、 save_checkpoint 和注册 hook 功能。

### 一个简单例子

以最常用的分类任务为例详细说明 `runner` 的使用方法。 开启任何一个训练任务，都需要包括如下步骤：

**(1) dataloader、model 和优化器等类初始化**

```python
# 模型类初始化
model=...
# 优化器类初始化，典型值 cfg.optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer = build_optimizer(model, cfg.optimizer)
# 工作流对应的 dataloader 初始化
data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            ...) for ds in dataset
    ]
```

**(2) runner 类初始化**

```python
runner = build_runner(
    # cfg.runner 典型配置为
    # runner = dict(type='EpochBasedRunner', max_epochs=200)
    cfg.runner,
    default_args=dict(
        model=model,
        batch_processor=None,
        optimizer=optimizer,
        logger=logger))
```

**(3) 注册默认训练所必须的 hook，和用户自定义 hook**

```python
# 注册定制必需的 hook
runner.register_training_hooks(
    # lr相关配置，典型为
    # lr_config = dict(policy='step', step=[100, 150])
    cfg.lr_config,
    # 优化相关配置，例如 grad_clip 等
    optimizer_config,
    # 权重保存相关配置，典型为
    # checkpoint_config = dict(interval=1)，每个单位都保存权重
    cfg.checkpoint_config,
    # 日志相关配置
    cfg.log_config,
    ...)

# 注册用户自定义 hook
# 例如想使用 ema 功能，则可以设置 custom_hooks=[dict(type='EMAHook')]
if cfg.get('custom_hooks', None):
    custom_hooks = cfg.custom_hooks
    for hook_cfg in cfg.custom_hooks:
        hook_cfg = hook_cfg.copy()
        priority = hook_cfg.pop('priority', 'NORMAL')
        hook = build_from_cfg(hook_cfg, HOOKS)
        runner.register_hook(hook, priority=priority)
```

然后可以进行 resume 或者 load_checkpoint 对权重进行加载。

**(4) 开启训练流**

```python
# workflow 典型为 workflow = [('train', 1)]
# 此时就真正开启了训练
runner.run(data_loaders, cfg.workflow)
```

关于 workflow 设置，以 `EpochBasedRunner` 为例，详情如下：

- 假设只想运行训练工作流，则可以设置 workflow = \[('train', 1)\]，表示只进行迭代训练
- 假设想运行训练和验证工作流，则可以设置 workflow = \[('train',  3), ('val', 1)\]，表示先训练 3 个 epoch ，然后切换到 val 工作流，运行 1 个 epoch，然后循环，直到训练 epoch 次数达到指定值
- 工作流设置还自由定制，例如你可以先验证再训练 workflow = \[('val', 1), ('train', 1)\]

上述代码都已经封装到了各个代码库的 train.py 中，用户只需要设置相应的配置即可，上述流程会自动运行。
