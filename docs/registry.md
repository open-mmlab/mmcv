## Registry

MMCV implements [registry](https://github.com/open-mmlab/mmcv/blob/master/mmcv/utils/registry.py) to manage different modules that share similar functionalities, e.g., backbones, head, and necks, in detectors.
Most MM-started projects use registry to manage modules of datasets and models, such as [MMDetection](https://github.com/open-mmlab/mmdetection), [MMDetection3D](https://github.com/open-mmlab/mmdetection3d), [MMClassification](https://github.com/open-mmlab/mmclassification), [MMEditing](https://github.com/open-mmlab/mmediting), etc.

To use `Registry` to manage your modules in the codebase, there are three steps as below.

1. Create an registry
2. Create a build method
3. Use this registry to manage the modules

### A Simple Example

Here we show a simple example of using registry to manage modules in a package.
You can find many more complex examples in OpenMMLab's MM-started projects.

Assuming we wand to implement a series of Dataset Converter for converting different formats of data to the expected data format.
We create directory as a package named `converters`.

```python
import


```
