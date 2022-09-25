.. role:: hidden
    :class: hidden-section

mmcv.video
===================================

.. contents:: mmcv.video
   :depth: 2
   :local:
   :backlinks: top

.. currentmodule:: mmcv.video

IO
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   VideoReader
   Cache

.. autosummary::
   :toctree: generated
   :nosignatures:

   frames2video

Optical Flow
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   dequantize_flow
   flow_from_bytes
   flow_warp
   flowread
   flowwrite
   quantize_flow
   sparse_flow_from_bytes

Video Processing
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   concat_video
   convert_video
   cut_video
   resize_video
