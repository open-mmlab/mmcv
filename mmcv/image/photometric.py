import cv2
import numpy as np

from ..utils import is_tuple_of
from .colorspace import bgr2gray, gray2bgr


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


def adjust_color(img, alpha=1, beta=None, gamma=0):
    """It blends the source image and its gray image:

    ``output = img * alpha + gray_img * beta + gamma``

    Args:
        img (ndarray): The input source image.
        alpha (int | float): Weight for the source image. Default 1.
        beta (int | float): Weight for the converted gray image.
            If None, it's assigned the value (1 - `alpha`).
        gamma (int | float): Scalar added to each sum.
            Same as :func:`cv2.addWeighted`. Default 0.

    Returns:
        ndarray: Colored image which has the same size and dtype as input.
    """
    gray_img = bgr2gray(img)
    gray_img = np.tile(gray_img[..., None], [1, 1, 3])
    if beta is None:
        beta = 1 - alpha
    colored_img = cv2.addWeighted(img, alpha, gray_img, beta, gamma)
    if not colored_img.dtype == np.uint8:
        # Note when the dtype of `img` is not defaultly `np.uint8`
        # (e.g. np.float32), the value in `colored_img` got from cv2
        # is not guaranteed to be in range [0, 255], so here clip
        # is needed.
        colored_img = np.clip(colored_img, 0, 255)
    return colored_img


def imequalize(img):
    """Equalize the image histogram.

    This function applies a non-linear mapping to the input image,
    in order to create a uniform distribution of grayscale values
    in the output image.

    Args:
        img (ndarray): Image to be equalized.

    Returns:
        ndarray: The equalized image.
    """

    def _scale_channel(im, c):
        """Scale the data in the corresponding channel."""
        im = im[:, :, c]
        # Compute the histogram of the image channel.
        histo = np.histogram(im, 256, (0, 255))[0]
        # For computing the step, filter out the nonzeros.
        nonzero_histo = histo[histo > 0]
        step = (np.sum(nonzero_histo) - nonzero_histo[-1]) // 255
        if not step:
            lut = np.array(range(256))
        else:
            # Compute the cumulative sum, shifted by step // 2
            # and then normalized by step.
            lut = (np.cumsum(histo) + (step // 2)) // step
            # Shift lut, prepending with 0.
            lut = np.concatenate([[0], lut[:-1]], 0)
        # If step is zero, return the original image.
        # Otherwise, index from lut.
        return np.where(np.equal(step, 0), im, lut[im])

    # Scales each channel independently and then stacks
    # the result.
    s1 = _scale_channel(img, 0)
    s2 = _scale_channel(img, 1)
    s3 = _scale_channel(img, 2)
    equalized_img = np.stack([s1, s2, s3], axis=-1)
    return equalized_img


def adjust_brightness(img, factor=1.):
    """Adjust image brightness.

    This function controls the brightness of an image. An
    enhancement factor of 0.0 gives a black image.
    A factor of 1.0 gives the original image. This function
    blends the source image and the degenerated black image:

    ``output = img * factor + degenerated * (1 - factor)``

    Args:
        img (ndarray): Image to be brightened.
        factor (float): A value controls the enhancement.
            Factor 1.0 returns the original image, lower
            factors mean less color (brightness, contrast,
            etc), and higher values more. Default 1.

    Returns:
        ndarray: The brightened image.
    """
    degenerated = np.zeros_like(img)
    # Note manually convert the dtype to np.float32, to
    # achieve as close results as PIL.ImageEnhance.Brightness.
    # Set beta=1-factor, and gamma=0
    brightened_img = cv2.addWeighted(
        img.astype(np.float32), factor, degenerated.astype(np.float32),
        1 - factor, 0)
    return brightened_img.astype(img.dtype)


def adjust_contrast(img, factor=1.):
    """Adjust image contrast.

    This function controls the contrast of an image. An
    enhancement factor of 0.0 gives a solid grey
    image. A factor of 1.0 gives the original image. It
    blends the source image and the degenerated mean image:

    ``output = img * factor + degenerated * (1 - factor)``

    Args:
        img (ndarray): Image to be contrasted. BGR order.
        factor (float): Same as :func:`mmcv.adjust_brightness`.

    Returns:
        ndarray: The contrasted image.
    """
    gray_img = bgr2gray(img)
    hist = np.histogram(gray_img, 256, (0, 255))[0]
    mean = round(np.sum(gray_img) / np.sum(hist))
    degenerated = (np.ones_like(img[..., 0]) * mean).astype(img.dtype)
    degenerated = gray2bgr(degenerated)
    contrasted_img = cv2.addWeighted(
        img.astype(np.float32), factor, degenerated.astype(np.float32),
        1 - factor, 0)
    return contrasted_img.astype(img.dtype)


def lut_transform(img, lut_table):
    """Transform array by look-up table.

    The function lut_transform fills the output array with values from the
    look-up table. Indices of the entries are taken from the input array.

    Args:
        img (ndarray): Image to be transformed.
        lut_table (ndarray): look-up table of 256 elements; in case of
            multi-channel input array, the table should either have a single
            channel (in this case the same table is used for all channels) or
            the same number of channels as in the input array.

    Returns:
        ndarray: The transformed image.
    """
    assert isinstance(img, np.ndarray)
    assert 0 <= np.min(img) and np.max(img) <= 255
    assert isinstance(lut_table, np.ndarray)
    assert lut_table.shape == (256, )

    return cv2.LUT(np.array(img, dtype=np.uint8), lut_table)


def clahe(img, clip_limit=40.0, tile_grid_size=(8, 8)):
    """Use CLAHE method to process the image.

    See `ZUIDERVELD,K. Contrast Limited Adaptive Histogram Equalization[J].
    Graphics Gems, 1994:474-485.` for more information.

    Args:
        img (ndarray): Image to be processed.
        clip_limit (float): Threshold for contrast limiting. Default: 40.0.
        tile_grid_size (tuple[int]): Size of grid for histogram equalization.
            Input image will be divided into equally sized rectangular tiles.
            It defines the number of tiles in row and column. Default: (8, 8).

    Returns:
        ndarray: The processed image.
    """
    assert isinstance(img, np.ndarray)
    assert img.ndim == 2
    assert isinstance(clip_limit, (float, int))
    assert is_tuple_of(tile_grid_size, int)
    assert len(tile_grid_size) == 2

    clahe = cv2.createCLAHE(clip_limit, tile_grid_size)
    return clahe.apply(np.array(img, dtype=np.uint8))
