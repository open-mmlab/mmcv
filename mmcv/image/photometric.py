import cv2
import numpy as np

from .colorspace import bgr2gray


def imnormalize(img, mean, std, to_rgb=True):
    """Normalize an image with mean and std.

    Args:
        img (ndarray): Image to be normalized.
        mean (ndarray): The mean to be used for normalize.
        std (ndarray): The std to be used for normalize.
        to_rgb (bool): Whether to convert to rgb.

    Returns:
        ndarray: The normalized image.
    """
    img = img.copy().astype(np.float32)
    return imnormalize_(img, mean, std, to_rgb)


def imnormalize_(img, mean, std, to_rgb=True):
    """Inplace normalize an image with mean and std.

    Args:
        img (ndarray): Image to be normalized.
        mean (ndarray): The mean to be used for normalize.
        std (ndarray): The std to be used for normalize.
        to_rgb (bool): Whether to convert to rgb.

    Returns:
        ndarray: The normalized image.
    """
    # cv2 inplace normalization does not accept uint8
    assert img.dtype != np.uint8
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    if to_rgb:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
    cv2.subtract(img, mean, img)  # inplace
    cv2.multiply(img, stdinv, img)  # inplace
    return img


def imdenormalize(img, mean, std, to_bgr=True):
    assert img.dtype != np.uint8
    mean = mean.reshape(1, -1).astype(np.float64)
    std = std.reshape(1, -1).astype(np.float64)
    img = cv2.multiply(img, std)  # make a copy
    cv2.add(img, mean, img)  # inplace
    if to_bgr:
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)  # inplace
    return img


def iminvert(img):
    """Invert (negate) an image.

    Args:
        img (ndarray): Image to be inverted.

    Returns:
        ndarray: The inverted image.
    """
    return np.full_like(img, 255) - img


def solarize(img, thr=128):
    """Solarize an image (invert all pixel values above a threshold)

    Args:
        img (ndarray): Image to be solarized.
        thr (int): Threshold for solarizing (0 - 255).

    Returns:
        ndarray: The solarized image.
    """
    img = np.where(img < thr, img, 255 - img)
    return img


def posterize(img, bits):
    """Posterize an image (reduce the number of bits for each color channel)

    Args:
        img (ndarray): Image to be posterized.
        bits (int): Number of bits (1 to 8) to use for posterizing.

    Returns:
        ndarray: The posterized image.
    """
    shift = 8 - bits
    img = np.left_shift(np.right_shift(img, shift), shift)
    return img


def color(img, alpha=1, beta=None, gamma=0):
    """Color the image. It blends the source image and its gray.

    image with the format: `img * alpha + gray_img * beta + gamma`.

    Args:
        img (ndarray): The input source image.
        alpha (int | float): Weight for the source image. Default 1.
        beta (int | float): Weight for the converted gray image.
            If None, it's assigned the value (1 - `alpha`).
        gamma (int | float): Scalar added to each sum.
            Same as :func:`cv2.addWeighted`. Default 0.

    Returns:
        ndarray: Colored image.
    """
    gray_img = bgr2gray(img)
    gray_img = np.tile(gray_img[..., None], [1, 1, 3])
    if beta is None:
        beta = 1 - alpha
    colored_img = cv2.addWeighted(img, alpha, gray_img, beta, gamma)
    return colored_img
