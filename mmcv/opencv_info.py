import cv2


def use_opencv2():
    try:
        major_version = cv2.__version__.split('.')[0]
    except TypeError:  # solves doc generation issue
        major_version = 4
    return major_version == '2'


USE_OPENCV2 = use_opencv2()
