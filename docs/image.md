## Image

This module provides some image processing methods.

### Read/Write/Show
To read or write images files, use `imread` or `imwrite`.

```python
import mmcv

img = mmcv.imread('test.jpg')
img = mmcv.imread('test.jpg', flag='grayscale')
img_ = mmcv.imread(img) # nothing will happen, img_ = img
mmcv.imwrite(img, 'out.jpg')
```

To read images from bytes

```python
with open('test.jpg', 'rb') as f:
    data = f.read()
img = mmcv.imfrombytes(data)
```

To show an image file or a loaded image

```python
mmcv.imshow('tests/data/color.jpg')

for i in range(10):
    img = np.random.randint(256, size=(100, 100, 3), dtype=np.uint8)
    mmcv.imshow(img, win_name='test image', wait_time=200)
```

### Resize
There are three resize methods. All `imresize_*` methods have a parameter `return_scale`,
if this param is `False`, then the return value is merely the resized image, otherwise
is a tuple (resized_img, scale).

```python
# resize to a given size
mmcv.imresize(img, (1000, 600), return_scale=True)

# resize to the same size of another image
mmcv.imresize_like(img, dst_img, return_scale=False)

# resize by a ratio
mmcv.imrescale(img, 0.5)

# resize so that the max edge no longer than 1000, short edge no longer than 800
# without changing the aspect ratio
mmcv.imrescale(img, (1000, 800))
```

### Color space conversion
Supported conversion methods:
- bgr2gray
- gray2bgr
- bgr2rgb
- rgb2bgr
- bgr2hsv
- hsv2bgr

```python
img = mmcv.imread('tests/data/color.jpg')
img1 = mmcv.bgr2rgb(img)
img2 = mmcv.rgb2gray(img1)
img3 = mmcv.bgr2hsv(img)
```

### Crop
Support single/multiple crop.

```python
import mmcv
import numpy as np

img = mmcv.read_img('tests/data/color.jpg')
bboxes = np.array([10, 10, 100, 120])  # x1, y1, x2, y2
patch = mmcv.crop_img(img, bboxes)
bboxes = np.array([[10, 10, 100, 120], [0, 0, 50, 50]])
patches = mmcv.crop_img(img, bboxes)
```

Resizing cropped patches.
```python
# upsample patches by 1.2x
patches = mmcv.crop_img(img, bboxes, scale_ratio=1.2)
```

### Padding
Pad an image to specific size with given values.

```python
img = mmcv.read_img('tests/data/color.jpg')
img = mmcv.pad_img(img, (1000, 1200), pad_val=0)
img = mmcv.pad_img(img, (1000, 1200), pad_val=[100, 50, 200])
```