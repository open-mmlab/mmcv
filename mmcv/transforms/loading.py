# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
from .builder import TRANSFORMS
from .base import BaseTransform


@TRANSFORMS.register_module()
class LoadImageFromFile(BaseTransform):
    """Load an image from file.

    Required Key:

    - img_path

    Modified Key:

    - img
    - width
    - height
    - ori_width
    - ori_height

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:`mmcv.imfrombytes`.
            See :fun:`mmcv.imfrombytes` for details.
            Defaults to 'cv2'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(
        self,
        to_float32: bool = False,
        color_type: str = 'color',
        imdecode_backend: str = 'cv2',
        file_client_args: dict = dict(backend='disk')
    ) -> None:
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def transform(self, results: dict) -> dict:
        """Functions to load image.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        filename = results['img_path']
        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        height, width = img.shape[:2]
        results['height'] = height
        results['width'] = width
        results['ori_height'] = height
        results['ori_width'] = width
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"imdecode_backend='{self.imdecode_backend}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str


class LoadAnnotation(BaseTransform):
    """ Load and process the `instances` and `seg_map` annotation provided by
    dataset. `LoadAnnotation` loads only one image annotation.

    The annotation format is as the following:

    ```python
    {
        'instances':
        [
            {
            # List of 4 numbers representing the bounding box of the
            # instance, in (x1, y1, x2, y2) order.
            'bbox': [x1, y1, x2, y2],

            # Label of image classification.
            'bbox_label': 1,

            # Used in key point detection.
            # Can only load the format of [x1, y1, v1,…, xn, yn, vn]. v[i]
            # means the visibility of this keypoint. n must be equal to the
            # number of keypoint categories.
            'keypoints': [x1, y1, v1, ..., xn, yn, vn]
            }
        ]
        # Filename of semantic or panoptic segmentation ground truth file.
        'seg_map': 'a/b/c'
    }
    ```
    After load and process:

    ```python
    {
        'gt_bboxes': np.ndarray(N, 4) # In (x1, y1, x2, y2) order, float type.
        'gt_bboxes_labels': np.ndarray(N, ) # In int type.
        'gt_semantic_seg': np.ndarray (H, W) # In uint8 type.
        'gt_keypoints': np.ndarray(N, NK, 3) # in (x, y, v) order, float type.

    }
    ```

    Required Key:

    - instances
        - bbox (optional)
        - bbox_label (optional)
        - keypoints
    - seg_map

    Added Key:

    - gt_bboxes
    - gt_bboxes_labels
    - gt_semantic_seg
    - gt_keypoints

    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
             Default: True.
        with_label (bool): Whether to parse and load the label annotation.
            Default: True.
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Default: False.
        with_kps (bool): Whether to parse and load the keypoints annotation.
             Default: False.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:`mmcv.imfrombytes`.
            See :fun:`mmcv.imfrombytes` for details.
            Defaults to 'cv2'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(
        self,
        with_bbox: bool = True,
        with_label: bool = True,
        with_seg: bool = False,
        with_kps: bool = False,
        imdecode_backend: str = 'cv2',
        file_client_args: dict = dict(backend='disk')
    ) -> None:
        super().__init__()
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_seg = with_seg
        self.with_kps = with_kps
        self.imdecode_backend = imdecode_backend
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_bboxes(self, results: dict) -> None:
        """Private function to load bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmcv.BaseDataset`.
        Returns:
            dict: The dict contains loaded bounding box annotations.
        """
        gt_bboxes = []
        for instance in results['instances']:
            gt_bboxes.append(instance['bbox'])
        results['gt_bboxes'] = np.array(gt_bboxes)
        return results

    def _load_labels(self, results: dict) -> None:
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj :obj:`mmcv.BaseDataset`.

        Returns:
            dict: The dict contains loaded label annotations.
        """
        gt_bboxes_labels = []
        for instance in results['instances']:
            gt_bboxes_labels.append(instance['bbox_label'])
        results['gt_bboxes_labels'] = np.array(gt_bboxes_labels)
        return results

    def _load_semantic_seg(self, results: dict) -> None:
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:`mmcv.BaseDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        img_bytes = self.file_client.get(results['seg_map'])
        results['gt_semantic_seg'] = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze()

    def _load_kps(self, results: dict) -> None:
        """Private function to load keypoints annotations.

        Args:
            results (dict): Result dict from :obj:`mmcv.BaseDataset`.
        Returns:
            dict: The dict contains loaded keypoints annotations.
        """
        gt_keypoints = []
        for instance in results['instances']:
            gt_keypoints.append(instance['keypoints'])
        results['gt_keypoints'] = np.array(gt_keypoints).reshape(
            (len(gt_keypoints), -1, 3))

    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmcv.BaseDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label and
                semantic segmentation and keypoints annotations.
        """

        if self.with_bbox:
            self._load_bboxes(results)
        if self.with_label:
            self._load_labels(results)
        if self.with_seg:
            self._load_semantic_seg(results)
        if self.with_kps:
            self._load_kps(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_seg={self.with_seg}, '
        repr_str += f'with_kps={self.with_kps}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'file_client_args={self.file_client_args})'
        return repr_str
