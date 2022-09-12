.. role:: hidden
    :class: hidden-section

mmcv.image
===================================

.. contents:: mmcv.image
   :depth: 2
   :local:
   :backlinks: top

.. currentmodule:: mmcv.image

IO
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   imfrombytes
   imread
   imwrite
   use_backend

Color Space
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   bgr2gray
   bgr2hls
   bgr2hsv
   bgr2rgb
   bgr2ycbcr
   gray2bgr
   gray2rgb
   hls2bgr
   hsv2bgr
   imconvert
   rgb2bgr
   rgb2gray
   rgb2ycbcr
   ycbcr2bgr
   ycbcr2rgb

Geometric
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   cutout
   imcrop
   imflip
   impad
   impad_to_multiple
   imrescale
   imresize
   imresize_like
   imresize_to_multiple
   imrotate
   imshear
   imtranslate
   rescale_size

Photometric
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   adjust_brightness
   adjust_color
   adjust_contrast
   adjust_hue
   adjust_lighting
   adjust_sharpness
   auto_contrast
   clahe
   imdenormalize
   imequalize
   iminvert
   imnormalize
   lut_transform
   posterize
   solarize

Miscellaneous
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   tensor2imgs
