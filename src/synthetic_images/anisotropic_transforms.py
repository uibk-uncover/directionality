from scipy.ndimage import convolve1d
import numpy as np
from PIL import Image


def anisotropic_resize_reconstruct(img, scale_y=1., scale_x=1., interpolation=Image.Resampling.LANCZOS):
    """
    Resize image from [height, width] to [scale_y * height, scale_x * width].
    Afterwards, scale the resized image back to the original resolution [height, width]

    :param img: 2D grayscale image. If the dtype is uint8, the resulting image is also of dtype uint8. If the dtype is float, the resulting image's dtype is float.
    :param scale_y: scale height by given factor, then scale back to original height
    :param scale_x: scale width by given factor, then scale back to original width
    :param interpolation: interpolation method
    :return: image of shape [height, width]
    """

    height, width = img.shape[:2]

    # Scale to intermediate resolution
    scaled_height = int(np.round(height * scale_y))
    scaled_width = int(np.round(width * scale_x))

    im = Image.fromarray(img)
    scaled_im = im.resize((scaled_width, scaled_height), interpolation)

    # Scale back to original resolution
    reconstructed_im = scaled_im.resize((width, height), interpolation)

    return np.array(reconstructed_im)


def smooth_horizontal(img, kernel_size):
    """
    Smooth an image along the horizontal direction using a Hanning window
    :param img: 2D grayscale image. Pixel values can have an arbitrary range.
    :param kernel_size: length of the Hanning window
    :return: filtered image of same shape
    """
    kernel = np.hanning(kernel_size)
    kernel /= kernel.sum()

    smoothed_img = convolve1d(img, weights=kernel, axis=1)
    return smoothed_img


def smooth_vertical(img, kernel_size):
    """
    Smooth an image along the vertical direction using a Hanning window
    :param img: 2D grayscale image. Pixel values can have an arbitrary range.
    :param kernel_size: length of the Hanning window
    :return: filtered image of same shape
    """
    kernel = np.hanning(kernel_size)
    kernel /= kernel.sum()

    smoothed_img = convolve1d(img, weights=kernel, axis=0)
    return smoothed_img
