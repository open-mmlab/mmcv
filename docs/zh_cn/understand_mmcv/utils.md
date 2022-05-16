## 辅助函数

### 进度条

如果你想跟踪函数批处理任务的进度，可以使用 `track_progress` 。它能以进度条的形式展示任务的完成情况以及剩余任务所需的时间（内部实现为for循环）。

```python
import mmcv

def func(item):
    # 执行相关操作
    pass

tasks = [item_1, item_2, ..., item_n]

mmcv.track_progress(func, tasks)
```

效果如下
![progress](../../en/_static/progress.*)

如果你想可视化多进程任务的进度，你可以使用 `track_parallel_progress` 。

```python
mmcv.track_parallel_progress(func, tasks, 8)  # 8 workers
```

![progress](../../_static/parallel_progress.*)

如果你想要迭代或枚举数据列表并可视化进度,你可以使用 `track_iter_progress` 。

```python
import mmcv

tasks = [item_1, item_2, ..., item_n]

for task in mmcv.track_iter_progress(tasks):
    # do something like print
    print(task)

for i, task in enumerate(mmcv.track_iter_progress(tasks)):
    # do something like print
    print(i)
    print(task)
```

### 计时器

mmcv提供的 `Timer` 可以很方便地计算代码块的执行时间。

```python
import time

with mmcv.Timer():
    # simulate some code block
    time.sleep(1)
```

你也可以使用 `since_start()` 和 `since_last_check()` 。前者返回计时器启动后的运行时长，后者返回最近一次查看计时器后的运行时长。

```python
timer = mmcv.Timer()
# code block 1 here
print(timer.since_start())
# code block 2 here
print(timer.since_last_check())
print(timer.since_start())
```
