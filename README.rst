MMCV
====

.. image:: https://img.shields.io/pypi/v/mmcv
  :target: https://pypi.org/project/mmcv

.. image:: https://github.com/open-mmlab/mmcv/workflows/build/badge.svg
  :target: https://github.com/open-mmlab/mmcv/actions

.. image:: https://codecov.io/gh/open-mmlab/mmcv/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/open-mmlab/mmcv

.. image:: 	https://img.shields.io/github/license/open-mmlab/mmcv.svg
  :target: https://github.com/open-mmlab/mmcv/blob/master/LICENSE


Introduction
------------

MMCV is a foundational python library for computer vision research and supports many
research projects in MMLAB, such as `MMDetection <https://github.com/open-mmlab/mmdetection>`_
and `MMAction <https://github.com/open-mmlab/mmaction>`_.

It provides the following functionalities.

- Universal IO APIs
- Image processing
- Video processing
- Image and annotation visualization
- Useful utilities (progress bar, timer, ...)
- PyTorch runner with hooking mechanism
- Various CNN architectures

See the `documentation <http://mmcv.readthedocs.io/en/latest>`_ for more features and usage.

Note: MMCV requires Python 3.6+.


Installation
------------

Try and start with

.. code::

    pip install mmcv


or install from source

.. code::

    git clone https://github.com/open-mmlab/mmcv.git
    cd mmcv
    pip install -e .

Note: If you would like to use :code:`opencv-python-headless` instead of :code:`opencv-python`,
e.g., in a minimum container environment or servers without GUI,
you can first install it before installing MMCV to skip the installation of :code:`opencv-python`.
