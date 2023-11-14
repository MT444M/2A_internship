import cv2
import numpy as np


def calculate_average_pixel_values(coords_boxes, frame):
    """
    Calculate the average pixel values for each sensor in the current frame.

    Args:
        coords_boxes (list): List of coordinates of the corners of each sensor.
        frame (np.ndarray): RGB image frame.

    Returns:
        tuple: A tuple containing the average pixel values for grayscale, red, green, and blue channels, respectively.

    """
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    average_values_gray = []
    average_values_red = []
    average_values_green = []
    average_values_blue = []

    for sensor in coords_boxes:
        # Extract the coordinates of each corner of the sensor
        x1, y1, x2, y2, x3, y3, x4, y4 = sensor[0][0], sensor[0][1], sensor[1][0], sensor[1][1], sensor[2][0], \
                                         sensor[2][1], sensor[3][0], sensor[3][1]

        # Extract the rectangle from the image
        pts = np.array([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])
        rect = cv2.boundingRect(pts)
        x, y, w, h = rect

        # Resize the rectangle
        reduction_percentage = 0.5  # To reduce the size by 40%
        new_width = int(w * reduction_percentage)
        new_height = int(h * reduction_percentage)
        x_offset = int((w - new_width) / 2)
        y_offset = int((h - new_height) / 2)
        x = x + x_offset
        y = y + y_offset
        w = new_width
        h = new_height

        # Extract the reduced ROI from the RGB frame
        ROI_rgb = frame[y:y + h, x:x + w]

        # Calculate the average pixel values in the ROI for each channel (Red, Green, Blue)
        average_value_gray = np.mean(frame_gray[y:y + h, x:x + w])
        average_value_rgb = np.mean(ROI_rgb, axis=(0, 1))

        # Save the average values with 10 decimal places
        average_values_gray.append(np.round(average_value_gray, 10))
        average_values_red.append(np.round(average_value_rgb[2], 10))
        average_values_green.append(np.round(average_value_rgb[1], 10))
        average_values_blue.append(np.round(average_value_rgb[0], 10))

    return average_values_gray, average_values_red, average_values_green, average_values_blue


def subtract_first_value(df, column_name):
    """
    Subtract the first value in a column from all subsequent values in the column.

    Args:
        df (pandas.DataFrame): DataFrame containing the data.
        column_name (str): Name of the column to process.

    Returns:
        pandas.DataFrame: DataFrame with the updated values.

    """
    first_value = df[column_name].iloc[0]  # Get the first value in the column
    # Subtract the first value from subsequent rows
    df[column_name].iloc[1:] = df[column_name].iloc[1:].subtract(first_value)
    return df
