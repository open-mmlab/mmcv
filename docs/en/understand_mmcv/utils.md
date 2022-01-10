## Utils

### ProgressBar

If you want to apply a method to a list of items and track the progress, `track_progress`
is a good choice. It will display a progress bar to tell the progress and ETA.

```python
import mmcv

def func(item):
    # do something
    pass

tasks = [item_1, item_2, ..., item_n]

mmcv.track_progress(func, tasks)
```

The output is like the following.

![progress](../_static/progress.*)

There is another method `track_parallel_progress`, which wraps multiprocessing and
progress visualization.

```python
mmcv.track_parallel_progress(func, tasks, 8)  # 8 workers
```

![progress](../_static/parallel_progress.*)

If you want to iterate or enumerate a list of items and track the progress, `track_iter_progress`
is a good choice. It will display a progress bar to tell the progress and ETA.

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

### Timer

It is convenient to compute the runtime of a code block with `Timer`.

```python
import time

with mmcv.Timer():
    # simulate some code block
    time.sleep(1)
```

or try with `since_start()` and `since_last_check()`. This former can
return the runtime since the timer starts and the latter will return the time
since the last time checked.

```python
timer = mmcv.Timer()
# code block 1 here
print(timer.since_start())
# code block 2 here
print(timer.since_last_check())
print(timer.since_start())
```
