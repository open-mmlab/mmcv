## CNN

The subpackage `mmcv.cnn` ships with lots of CNN backbones and the number is growing.
Different from models provided in `torchvision`, these models are designed
for actual usage in different cases, with highly customizable interfaces.

Taking ResNet as an example, you can specify the following things:

- number of stages
- stage of output feature maps
- location of the stride 2 convolutional layer
- strides and dilation of each stage
- whether to freeze some first stages
- whether to fix BN stats
- whether to fix BN parameters
- whether to use checkpoint to save memory
