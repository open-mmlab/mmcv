from mmcv.opencv_info import USE_OPENCV2

if not USE_OPENCV2:
    from cv2 import (CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FPS,
                     CAP_PROP_FRAME_COUNT, CAP_PROP_FOURCC,
                     CAP_PROP_POS_FRAMES, VideoWriter_fourcc)
else:
    from cv2.cv import CV_CAP_PROP_FRAME_WIDTH as CAP_PROP_FRAME_WIDTH
    from cv2.cv import CV_CAP_PROP_FRAME_HEIGHT as CAP_PROP_FRAME_HEIGHT
    from cv2.cv import CV_CAP_PROP_FPS as CAP_PROP_FPS
    from cv2.cv import CV_CAP_PROP_FRAME_COUNT as CAP_PROP_FRAME_COUNT
    from cv2.cv import CV_CAP_PROP_FOURCC as CAP_PROP_FOURCC
    from cv2.cv import CV_CAP_PROP_POS_FRAMES as CAP_PROP_POS_FRAMES
    from cv2.cv import CV_FOURCC as VideoWriter_fourcc

from .io import *
from .processing import *
