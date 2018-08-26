import cv2


def use_opencv2():
    return cv2.__version__.split('.')[0] == '2'


USE_OPENCV2 = use_opencv2()
