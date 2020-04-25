# Copyright (c) Open-MMLab. All rights reserved.
import cv2
import numpy as np

import mmcv
from .color import color_val
from .image import imshow


class Figure:
    """A Figure for rendering and plotting images and other elements.

    Elements like bboxes, label texts, masks can be rendered on a figure.
    They can be rendered with corresponding methods `plot_xxx()` and overlayed
    with the "+" operator. A figure can be created with either `Figure.empty()`
    or `Figure.from_img()`. It provides two methods `show()` and `save()` for
    plotting.

    Args:
        data (ndarray): The numpy array.

    Examples:
        >>> img_file = 'tests/data/color.jpg'
        >>> bboxes = np.array([[10, 10, 200, 200], [200, 200, 300, 250]])
        >>> labels = np.array([1, 2])
        >>> fig = Figure.from_img(img_file) + plot_bboxes(bboxes) + \
        >>>       plot_bbox_labels(labels)
        >>> fig.save('out.jpg')
        >>> # fig.show()
    """

    @staticmethod
    def empty(shape):
        """Create an empty figure with all zeros.

        Args:
            shape (tuple[int]): The shape (h, w) of the figure.

        Returns:
            Figure: The created figure.
        """
        assert mmcv.is_tuple_of(shape, int) and len(shape) == 2
        zeros = np.zeros(shape + (3, ))
        return Figure(zeros)

    @staticmethod
    def from_img(img):
        """Create a figure from an image.

        Args:
            img (str | ndarray): Image path or loaded image.
                Same as :func:`mmcv.imread`.

        Returns:
            Figure: The created figure.
        """
        return Figure(mmcv.imread(img))

    def __init__(self, data):
        if not isinstance(data, np.ndarray):
            raise TypeError(
                f'data should be a numpy array, but got {type(data)}')
        self._data = data
        self._height = self.data.shape[0]
        self._width = self.data.shape[1]

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    def __copy__(self):
        return Figure(self._data.copy())

    def __add__(self, func):
        assert callable(func)
        return Figure(func(self._data.copy()))

    def __repr__(self):
        return f'Figure({self._data!r})'

    def show(self, win_name='', wait_time=0):
        """Show the figure with opencv GUI.

        Args:
            win_name (str): The window name. Default: ''.
            wait_time (int): Value of waitKey param. Default: 0
        """
        imshow(self._data, win_name, wait_time)

    def save(self, filepath):
        """Save the figure to an image.

        Args:
            filepath (str): The filepath to be saved.
        """
        mmcv.imwrite(self._data, filepath)


def plot_bboxes(bboxes, color='green', top_k=-1, thickness=1):
    """Plot bboxes on a figure.

    Args:
        bboxes (ndarray): An ndarray of shape (n, 4) or (n, 5). The first 4
            columns indicates (left, top, right, bottom), and the 5th column
            indicates scores.
        colors (str | tuple | Color): BBox color.
        top_k (int): Plot the first k bboxes only. Default: -1, which means
            ploting all bboxes.
        thickness (int): Line thickness. Default: 1.
    """

    assert isinstance(bboxes, np.ndarray)
    assert bboxes.ndim == 2
    assert bboxes.shape[1] in [4, 5]

    def plot(img):
        _bboxes = bboxes.astype(np.int32)
        _color = color_val(color)
        if top_k <= 0:
            _top_k = _bboxes.shape[0]
        else:
            _top_k = min(top_k, _bboxes.shape[0])
        for j in range(_top_k):
            left_top = (_bboxes[j, 0], _bboxes[j, 1])
            right_bottom = (_bboxes[j, 2], _bboxes[j, 3])
            cv2.rectangle(
                img, left_top, right_bottom, _color, thickness=thickness)
        return img

    return plot


def plot_bbox_labels(bboxes,
                     labels,
                     class_names=None,
                     color='green',
                     score_thr=0,
                     font_scale=0.5):
    """Plot bbox labels on a figure.

    Args:
        bboxes (ndarray): An ndarray of shape (n, 4) or (n, 5). The first 4
            columns indicates (left, top, right, bottom), and the 5th column
            indicates scores.
        labels (ndarray): An ndarray of shape (n, ).
        class_names (Sequence[str]): The names of each class. If not set, the
            names are "class1", "class2", ... for label 1, 2, ....
        colors (str | tuple | Color): Text color.
        score_thr (float): BBoxes below this threshold will be ignored.
            Default: 0.
        font_scale (int): Font scale of the label texts. Default: 0.5.
    """
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] in [4, 5]
    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
    text_color = color_val(color)

    def plot(img):
        for bbox, label in zip(bboxes, labels):
            bbox_int = bbox.astype(np.int32)
            label_text = class_names[
                label] if class_names is not None else f'class{label}'
            if len(bbox) > 4:
                label_text += '|{:.02f}'.format(bbox[-1])
            cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                        cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
        return img

    return plot


# def plot_rle_masks(masks, labels, score_thr=0):

#     assert len(masks) == len(labels)

#     def plot(img):
#         import pycocotools.mask as maskUtils
#         np.random.seed(42)
#         color_map = [
#             np.random.randint(0, 256, (1, 3), dtype=np.uint8)
#             for _ in range(max(labels) + 1)
#         ]
#         # draw segmentation masks
#         for mask, label in zip(masks, labels):
#             color_mask = color_map[label]
#             mask = maskUtils.decode(mask).astype(np.bool)
#             img[mask] = img[mask] * 0.5 + color_mask * 0.5
#         return img

#     return plot
