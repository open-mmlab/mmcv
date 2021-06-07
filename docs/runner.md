## Runner

The runner class is designed to manage the training. It eases the training process with less code demanded from users while staying flexible and configurable. The main features are as listed: 

- Support `EpochBasedRunner` and `IterBasedRunner` for different scenarios.
- Support customized workflow to allow switching between different modes while training. Currently, supported modes are train and val.
- Enable extensibility through various Hooks, including hooks defined in MMCV and customized ones. 

## EpochBasedRunner
As its name indicates, workflow in `EpochBasedRunner` should be set based on epochs. For example, [('train', 2), ('val', 1)] means running 2 epochs for training and 1 epoch for validation, iteratively. And each epoch may contain multiple iters. Currently, MMDetection uses `EpochBasedRunner` by default.

Let's take a look at its core logic:

```python
# the condition to stop training
while curr_epoch < max_epochs:
    # traverse the workflow.
    # e.g. workflow = [('train', 2), ('val', 1)]
    for i, flow in enumerate(workflow):
        # mode(e.g. train) determines which function to run
        mode, epochs = flow
        epoch_runner = getattr(self, mode)
        # execute the corresponding function
        for _ in range(epochs):
            epoch_runner(data_loaders[i], **kwargs)
```

Currently, we support 2 modes: train and val. Let's take a look at their core logic:

```python
# Currently, epoch_runner could be either train or val
def epoch_runner(self, data_loader, **kwargs):
    # traverse the dataset and get batch data for 1 epoch
    for i, data_batch in enumerate(data_loader):
        self.call_hook('before_train(val)_iter')
        self.run_iter(data_batch, train_mode=True or False, **kwargs)
        self.call_hook('after_train(val)_iter')
   self.call_hook('after_train(val)_epoch')
```

## IterBasedRunner
Different from `EpochBasedRunner`, workflow in `IterBasedRunner` should be set based on iterations. For example, [('train', 2), ('val', 1)] means running 2 iters for training and 1 iter for validation, iteratively. Currently, MMSegmentation uses `IterBasedRunner` by default.

Let's take a look at its core logic:

```python
# Although we set workflow by iters here, we might also need info on the epochs in some using cases. That can be provided by IterLoader.
iter_loaders = [IterLoader(x) for x in data_loaders]
# the condition to stop training
while curr_iter < max_iters:
    # traverse the workflow.
    # e.g. workflow = [('train', 2), ('val', 1)]
    for i, flow in enumerate(workflow):
        # mode(e.g. train) determines which function to run
        mode, iters = flow
        iter_runner = getattr(self, mode)
        # execute the corresponding function
        for _ in range(iters):
            iter_runner(iter_loaders[i], **kwargs)
```
Currently, we support 2 modes: train and val. Let's take a look at their core logic:

```python
# Currently, epoch_runner could be either train or val
def iter_runner(self, data_loader, **kwargs):
    # get batch data for 1 iter
    data_batch = next(data_loader)
    self.call_hook('before_train(val)_iter')
    outputs = self.model.train(val)_step(data_batch, self.optimizer, **kwargs)
    self.outputs = outputs
    self.call_hook('after_train(val)_iter')
```

Other than the basic functionalities explained above, `EpochBasedRunner` and `IterBasedRunner` provide methods such as `resume`, `save_checkpoint` and `register_hook`. In case you are not familiar with the term Hook mentioned earlier, we also provide a [tutorial]() about it. Essntially, a hook is functionality to alter or augument the code behaviors through predefined api. It allows users to have their own code called under certain circumstances. It makes code extensible in a non-intrusive manner.

## A Simple Example

We will walk you through the usage of runner with a classification task. The following code only contains essential steps for demonstration purpose. Please see [the link]() for the complete code. The following steps are neccesary for any training tasks.

**(1) Initialize dataloader, model, optimizer, etc.**

```python
# initialize model
model=...
# initialize optimizer, typically, we set: cfg.optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer = build_optimizer(model, cfg.optimizer)
# intialize the dataloader corresponding to the workflow(train/val)
data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            ...) for ds in dataset
    ]
```

**(2) Initialize runner**

```python
runner = build_runner(
    # cfg.runner is typically set as:
    # runner = dict(type='EpochBasedRunner', max_epochs=200)
    cfg.runner,
    default_args=dict(
        model=model,
        batch_processor=None,
        optimizer=optimizer,
        logger=logger))
```

**(3) Register training hooks and customized hooks.**

```python
# register defalt hooks neccesary for traning
runner.register_training_hooks(
    # configs of learning rate，it is typically set as:
    # lr_config = dict(policy='step', step=[100, 150])
    cfg.lr_config, 
    # configuration of optimizer, e.g. grad_clip
    optimizer_config,
    # configuration of saving checkpoints, it is typically set as:
    # checkpoint_config = dict(interval=1)，saving checkpoints every epochs
    cfg.checkpoint_config, 
    # configuration of logs 
    cfg.log_config,
    ...)

# register customized hooks
# say we want to enable ema, then we could set custom_hooks=[dict(type='EMAHook')]
if cfg.get('custom_hooks', None):
    custom_hooks = cfg.custom_hooks
    for hook_cfg in cfg.custom_hooks:
        hook_cfg = hook_cfg.copy()
        priority = hook_cfg.pop('priority', 'NORMAL')
        hook = build_from_cfg(hook_cfg, HOOKS)
        runner.register_hook(hook, priority=priority)
```

Then, we can use `resume` or `load_checkpoint` to load existing weights.

**(4) Start training**

```python
# workflow is typically set as: workflow = [('train', 1)]
# here the training begins.
runner.run(data_loaders, cfg.workflow)
```

Let's take `EpochBasedRunner` for example and go a little bit into details about setting workflow:

- Say we only want to put train in the workflow, then we can set: workflow = [('train', 1)]. The runner will only execute train iteratively in this case.
- Say we want to put both train and val in the workflow, then we can set: workflow = [('train', 3), ('val',1)]. The runner will first execute train for 3 epochs and then switch to val mode and execute val for 1 epoch. The workflow will be repeated until the current epoch hit the max_epochs.
- Workflow is highly flexible. Therefore, you can set workflow = [('val', 1), ('train',1)] if you would like the runner to validate first and train after.

The code we demostrated above is already in `train.py` in MM repositories. Simply modify the corresponding keys in the configuration files and the script will execute expected workflow automatically.


