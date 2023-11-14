import cv2
import numpy as np
from typing import Tuple

def Convert_to_YUV(image, clipLimit: int, tileGridSize: Tuple[int, int]):
    """
    Convert the RGB image to YUV and apply local luminance correction on the Y channel.
    Args:
        image (numpy.ndarray): The image matrix to be processed.
        clipLimit (float): Limit for contrast enhancement in the adaptive histogram equalization (CLAHE) algorithm.
        tileGridSize (tuple): Size of each tile in the grid used for computing the adaptive histogram.
    Returns:
        img_output (numpy.ndarray): The processed image with applied local luminance correction.
    """
    # Convert the image to YUV
    img_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

    # Separate the Y, U, and V channels
    Y, U, V = cv2.split(img_yuv)

    # Apply local luminance correction on the Y channel
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    Y_corr = clahe.apply(Y)

    # Merge the corrected channels into a single YUV image
    img_yuv_corr = cv2.merge((Y_corr, U, V))

    # Convert the image back to RGB
    img_rgb_corr = cv2.cvtColor(img_yuv_corr, cv2.COLOR_YUV2RGB)

    return img_rgb_corr


def Cropping_Image(image, rotated_angle: float, crop_left: int, crop_right: int, crop_top: int, crop_bottom: int):
    """
    Perform image processing to crop and rotate an image.

    Args:
        image (numpy.ndarray): The image to be processed.
        rotated_angle (float): The rotation angle in degrees for the image.
        crop_left (int): The number of pixels to crop from the left side.
        crop_right (int): The number of pixels to crop from the right side.
        crop_top (int): The number of pixels to crop from the top side.
        crop_bottom (int): The number of pixels to crop from the bottom side.
    Returns:
        numpy.ndarray: The cropped and rotated image.
    """

    # Rotate the image
    angle = rotated_angle
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    img_rotated = cv2.warpAffine(image, M, (cols, rows))

    # Crop the image
    cropped_img = img_rotated[crop_top:crop_bottom, crop_left:crop_right]

    # Return the cropped and rotated image
    return cropped_img


def correction(img: np.ndarray, params_gamma: float, params_Blur: Tuple[int, int]):
    """
    Apply a set of corrections to an image to enhance image quality.

    :param img: The image to be corrected.
    :type img: np.ndarray

    :param params_gamma: Parameter for gamma correction.
    :type params_gamma: float

    :param params_Blur: Kernel size for Gaussian filter.
    :type params_Blur: tuple[int, int]

    :return: The corrected image.
    :rtype: np.ndarray
    """

    # Convert the image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian filter for noise reduction and smoothing
    img = cv2.GaussianBlur(img, params_Blur, 0)

    # Apply gamma correction to increase contrast
    gamma = params_gamma
    img = np.power(img / 255.0, gamma)
    img = np.uint8(img * 255.0)

    # Apply dilation to close holes in contours
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img = cv2.dilate(img, kernel, iterations=1)

    return img


