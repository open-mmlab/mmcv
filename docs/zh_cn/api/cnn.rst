.. role:: hidden
    :class: hidden-section

mmcv.cnn
===================================

.. contents:: mmcv.cnn
   :depth: 2
   :local:
   :backlinks: top

.. currentmodule:: mmcv.cnn

Module
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   ContextBlock
   Conv2d
   Conv3d
   ConvAWS2d
   ConvModule
   ConvTranspose2d
   ConvTranspose3d
   ConvWS2d
   DepthwiseSeparableConvModule
   GeneralizedAttention
   HSigmoid
   HSwish
   LayerScale
   Linear
   MaxPool2d
   MaxPool3d
   NonLocal1d
   NonLocal2d
   NonLocal3d
   Scale
   Swish

Build Function
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   build_activation_layer
   build_conv_layer
   build_norm_layer
   build_padding_layer
   build_plugin_layer
   build_upsample_layer

Miscellaneous
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   fuse_conv_bn
   conv_ws_2d
   is_norm
   make_res_layer
   make_vgg_layer
   get_model_complexity_info
