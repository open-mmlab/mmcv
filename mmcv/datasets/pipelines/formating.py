from collections.abc import Sequence

from mmcv.parallel import DataContainer
from ..builder import PIPELINES


@PIPELINES.register_module()
class ToDataContainer:
    """Convert results to :obj:`mmcv.DataContainer` by given fields.

    Args:
        fields (Sequence[dict]): Each field is a dict like
            ``dict(key='xxx', **kwargs)``. The ``key`` in result will
            be converted to :obj:`mmcv.DataContainer` with ``**kwargs``.
            Default: ``(dict(key='img', stack=True),)``.

    Example:
        >>> from mmcv.image.io import imread
        >>> from mmcv.datasets.pipelines import ToDataContainer
        >>> img = imread('img_path')  # img_path is the path of image
        >>> cfg = {'fields': ({'key': 'img', 'stack': True}, )}
        >>> results = {
        ...     'img': img,
        ...     'img_fields': ['img']
        ... }
        >>> to_data_container = ToDataContainer(**cfg)
        >>> results = to_data_container(results)
    """

    def __init__(self, fields: Sequence = (dict(key='img', stack=True), )):
        self.fields = fields

    def __call__(self, results):
        """Call function to convert data in results to
        :obj:`mmcv.DataContainer`.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data converted to \
                :obj:`mmcv.DataContainer`.
        """

        for field in self.fields:
            field = field.copy()
            key = field.pop('key')
            results[key] = DataContainer(results[key], **field)
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(fields={self.fields})'
