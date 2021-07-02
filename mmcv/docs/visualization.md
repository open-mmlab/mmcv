## Visualization

`mmcv` can show images and annotations (currently supported types include bounding boxes).

```python
# show an image file
mmcv.imshow('a.jpg')

# show a loaded image
img = np.random.rand(100, 100, 3)
mmcv.imshow(img)

# show image with bounding boxes
img = np.random.rand(100, 100, 3)
bboxes = np.array([[0, 0, 50, 50], [20, 20, 60, 60]])
mmcv.imshow_bboxes(img, bboxes)
```

`mmcv` can also visualize special images such as optical flows.

```python
flow = mmcv.flowread('test.flo')
mmcv.flowshow(flow)
```
