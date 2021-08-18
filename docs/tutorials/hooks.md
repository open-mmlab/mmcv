## Hooks

### EvalHook

There are two kinds of evaluation hooks in MMCV, which will regularly perform
evaluation in a given interval. One is `EvalHook` for performing in
non-distributed environment, and the other is `DistEvalHook` for distributed
environment. The only difference of them is that DistEvalHook will broadcast
BatchNorm's buffers of rank 0 to other ranks to avoid the inconsistent
performance of models in different ranks.

```{note}
Before the evaluation hooks emerged, we can also perform evaluation by
setting the 'val' mode in workflow like `workflow = [('train', 1), ('val', 1)]`
which means performing evaluation after training one epoch. However, the 'val'
mode is not enough powerful and scalable. Now, we no longer recommend adding
'val' to workflow and evaluationhooks is strongly recommended.
```

#### Examples

+ Evaluate in a given interval

    ```python
    >>> runner = EpochBasedRunner(...)
    >>> val_dataloader = ...
    >>> # perform evaluation every 5 epochs
    >>> eval_hook = EvalHook(val_dataloader, interval=5)
    ```

+ Save the best checkpoint

    ```python
    >>> runner = EpochBasedRunner(...)
    >>> val_dataloader = ...
    >>> # assuming the best epoch is 5, its best checkpoint name will be
    >>> # 'best_acc_5.pth'
    >>> eval_hook = EvalHook(val_dataloader, save_best='acc')
    ```

+ Enable evaluation before the training starts

    ```python
    >>> runner = EpochBasedRunner(...)
    >>> val_dataloader = ...
    >>> # when resuming checkpoint from epoch=5 and setting the start param to
    >>> # be less than or equal than 5 it will evaluate before the training
    >>> eval_hook = EvalHook(val_dataloader, start=5)
    ```

+ Evaluate in distributed environment

    ```python
    >>> runner = EpochBasedRunner(...)
    >>> val_dataloader = ...
    >>> eval_hook = DistEvalHook(val_dataloader)
    ```

+ Evaluataion for `IterBasedRunner`

    ```python
    >>> runner = IterBasedRunner(...)
    >>> val_dataloader = ...
    >>> eval_hook = EvalHook(val_dataloader, by_epoch=False)
    ```

#### More examples

+ Set the `save_best` param to `auto`

    If ``save_best`` is ``auto``, the first key of the returned `OrderedDict`
    result will be used. Note that the result is returned by
    `val_dataloader.dataset.evaluate(...)`.

    ```python
    >>> runner = EpochBasedRunner(...)
    >>> val_dataloader = ...
    >>> # assume the result is {'acc': 99, 'mAP': 50} and the best epoch
    >>> # is 5, its filename will be 'best_acc_5.pth'
    >>> eval_hook = EvalHook(val_dataloader, save_best='auto')
    ```
