import pytest

from mmcv.runner import TRANSFORMS, BaseTransform, DataPipeline


@TRANSFORMS.register_module()
class TransformA(BaseTransform):

    def __call__(self, data):
        data['a'] = 1
        return data

    @property
    def required_keys(self):
        return []

    @property
    def updated_keys(self):
        return ['a']


@TRANSFORMS.register_module()
class TransformB1(BaseTransform):

    def __call__(self, data):
        data['b'] = data['a']
        return data

    @property
    def required_keys(self):
        return ['a']

    @property
    def updated_keys(self):
        return ['b']


@TRANSFORMS.register_module()
class TransformB2(BaseTransform):

    def __call__(self, data):
        data['b'] = 10
        return data

    @property
    def required_keys(self):
        return []

    @property
    def updated_keys(self):
        return ['b']


@TRANSFORMS.register_module()
class TransformC(BaseTransform):

    def __call__(self, data):
        data['b'] += 1
        data['c'] = 1
        return data

    @property
    def required_keys(self):
        return ['b']

    @property
    def updated_keys(self):
        return ['b', 'c']


@TRANSFORMS.register_module()
class TransformD(BaseTransform):

    def __call__(self, data):
        return None

    @property
    def required_keys(self):
        return []

    @property
    def updated_keys(self):
        return []


def test_datapipeline():
    data_pipeline = DataPipeline([TransformA(), TransformB1()])
    result = data_pipeline(dict())
    assert result == dict(a=1, b=1)
    assert str(data_pipeline) == (
        'DataPipeline(\n    TransformA,\n    TransformB1,\n)')

    data_pipeline = DataPipeline(
        [dict(type='TransformA'),
         dict(type='TransformB1')])
    result = data_pipeline(dict())
    assert result == dict(a=1, b=1)

    data_pipeline = DataPipeline([
        dict(type='TransformA'),
        dict(type='TransformB1'),
        dict(type='TransformB2'),
        dict(type='TransformC')
    ])
    result = data_pipeline(dict())
    assert result == dict(a=1, b=11, c=1)

    data_pipeline = DataPipeline([])
    result = data_pipeline(dict())
    assert result == dict()

    data_pipeline = DataPipeline(
        [dict(type='TransformA'),
         dict(type='TransformD')])
    result = data_pipeline(dict())
    assert result is None

    # transforms must be a list of dict or BaseTransform objects
    with pytest.raises(TypeError):
        data_pipeline = DataPipeline(dict())

    # transforms must be a list of dict or BaseTransform objects
    with pytest.raises(TypeError):
        data_pipeline = DataPipeline([1])

    # required keys are not provided
    with pytest.raises(KeyError):
        data_pipeline = DataPipeline(
            [dict(type='TransformA'),
             dict(type='TransformC')])

    # input data must be a dict
    with pytest.raises(TypeError):
        data_pipeline = DataPipeline([TransformA(), TransformB1()])
        result = data_pipeline(0)

    # input data does not provide required keys
    with pytest.raises(KeyError):
        data_pipeline = DataPipeline([TransformC()])
        result = data_pipeline(dict())
