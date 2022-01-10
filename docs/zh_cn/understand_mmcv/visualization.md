## 可视化

`mmcv` 可以展示图像以及标注（目前只支持标注框）

```python
# 展示图像文件
mmcv.imshow('a.jpg')

# 展示已加载的图像
img = np.random.rand(100, 100, 3)
mmcv.imshow(img)

# 展示带有标注框的图像
img = np.random.rand(100, 100, 3)
bboxes = np.array([[0, 0, 50, 50], [20, 20, 60, 60]])
mmcv.imshow_bboxes(img, bboxes)
```

`mmcv` 也可以展示特殊的图像，例如光流

```python
flow = mmcv.flowread('test.flo')
mmcv.flowshow(flow)
```
