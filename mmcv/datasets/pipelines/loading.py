import os.path as osp

import numpy as np

import mmcv
from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadImageFromFile:
    """Load an image from file.

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes()`.
            Defaults to 'color'.
        imdecode_backend (str): The backend argument for
            :func:`mmcv.imfrombytes()`. Defaults to 'cv2'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.

    Example:
        >>> from mmcv.datasets.pipelines import LoadImageFromFile
        >>> transform = LoadImageFromFile()
        >>> input_dict = {
        ...     'img_prefix': '/home/xxx/datasets/train',
        ...     'filename': 'first.png'
        ... }
        >>> output_dict = transform(input_dict)
        >>> for k, v in output_dict.items():
        ...     print(k, type(v))
        img_prefix <class 'str'>
        filename <class 'str'>
        img_fields <class 'list'>
        ori_filename <class 'str'>
        img <class 'numpy.ndarray'>
        img_shape <class 'tuple'>
        ori_shape <class 'tuple'>
        height <class 'int'>
        width <class 'int'>
    """

    def __init__(self,
                 to_float32: bool = False,
                 color_type: str = 'color',
                 imdecode_backend: str = 'cv2',
                 file_client_args: dict = dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results: dict) -> dict:
        """Call function to load image from file.

        Args:
            results (dict): Required keys are 'img_prefix' (optional) and
                'filename'.

        Returns:
            dict: Image loading results.

            - Added keys are "ori_filename", "img_fields", "img",
              "img_shape", "ori_shape" (same as "img_shape"),
              "height" and "width".
            - Updated key is "filename".
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        assert 'filename' in results, 'LoadImageFromFile requires key '\
            '"filename". Please check your pipelines.'
        if results.get('img_prefix', None) is not None:
            filename = osp.join(results['img_prefix'], results['filename'])
        else:
            filename = results['filename']

        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        if self.to_float32:
            img = img.astype(np.float32)

        results['img_fields'] = ['img']
        results['ori_filename'] = results['filename']
        results['filename'] = filename
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['height'] = img.shape[0]
        results['width'] = img.shape[1]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"imdecode_backend='{self.imdecode_backend}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str
