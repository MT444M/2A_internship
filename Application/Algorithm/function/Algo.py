from typing import Tuple
import cv2
import function
from function import Param
import pandas as pd
from tkinter import messagebox


def extract_data2(frame, coords_sensors, crop=None, rotation_angle=None):
    """
    Extracts average pixel values from a frame for each sensor.

    Args:
        frame (numpy.ndarray): The frame to process.
        coords_sensors (list of list): Matrix of all vertices of all sensors.
        crop (tuple[int, int, int, int]): Crop parameters for frame cropping (left, right, top, bottom).
        rotation_angle (float): Rotation angle for frame rotation.

    Returns:
        tuple: A tuple containing average gray, green, red, and blue values for each sensor.
    """

    # Original Image cropped
    Ori_crop = function.Cropping_Image(frame, rotation_angle, crop[0], crop[1], crop[2], crop[3])

    # Calculate the average pixel values for each sensor
    average_gray, average_red, average_green, average_blue = function.calculate_average_pixel_values(
        coords_sensors, Ori_crop)
    return gray_values, green_values, red_values, blue_values

def extract_data(video, type_extract, crop=None, rotation_angle=None, coords_sensors=None, custom_value_extract=None):
    """
    Launches the algorithm to process video frames and extract average pixel values.

    Args:
        video (cv2.VideoCapture): Video source to process.
        type_extract (str): Type of extraction ('Per frame' or 'Per second').
        custom_value_extract:
        crop (tuple[int, int, int, int]): Crop parameters for frame cropping (left, right, top, bottom).
        rotation_angle (float): Rotation angle for frame rotation.
        coords_sensors: matrix of all vertices of all sensors
    Returns:
        tuple: A tuple containing average gray, green, red, and blue values for each sensor.
    """
    # Create lists to store the average pixel values
    average_gray_values = []
    average_red_values = []
    average_green_values = []
    average_blue_values = []
    # Initialize len_coords with a default value
    len_coords = 0

    # Define the sampling rate (number of frames per second)
    frames_in_second = 0
    if type_extract == 'Per frame':
        sampling_rate = 1
    elif type_extract == 'Per second':
        sampling_rate = 30
    elif type_extract == 'custom':
        sampling_rate = custom_value_extract
    else:
        sampling_rate = 1

    # Iterate through the frames of the video
    frame_count = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        # Original Image cropped
        Ori_crop = function.Cropping_Image(frame, rotation_angle, crop[0], crop[1], crop[2], crop[3])
        len_coords = len(coords_sensors)
        frames_in_second += 1

        # Check if the current frame is the last frame within a second
        if frame_count % sampling_rate == sampling_rate:
            # Calculate the average pixel values for each sensor within a second
            if frames_in_second > 0:
                average_gray, average_red, average_green, average_blue = function.calculate_average_pixel_values(
                    coords_sensors, Ori_crop)
                average_gray_values.append([frame_count] + average_gray)
                average_red_values.append([frame_count] + average_red)
                average_green_values.append([frame_count] + average_green)
                average_blue_values.append([frame_count] + average_blue)

        # Update the frame count
        frame_count += 1

    # Return the average pixel values, frame count, and length of coords_sensors
    return len_coords, average_gray_values, average_green_values, average_red_values, average_blue_values


def create_grid(nbre_lines, nbre_corners, width_sensor, separate_distance, coord_grid=None):
    """
    :param nbre_lines:
    :param nbre_corners:
    :param width_sensor:
    :param separate_distance:
    :param coord_grid:
    :return:
    """
    # Create three separate partial grids
    nbre_line = nbre_lines
    nbre_corners = nbre_corners
    width_sensor = width_sensor
    separate_distance = separate_distance  # pixel
    # Coordinate of the 3 vertices of the grid
    x1 = coord_grid[0][0]
    y1 = coord_grid[0][1]
    x2 = coord_grid[1][0]
    y2 = coord_grid[1][1]
    x3 = coord_grid[2][0]
    y3 = coord_grid[2][1]
    grid_1 = function.create_partial_grid(nbre_line, nbre_corners, width_sensor, separate_distance, x1, y1)
    grid_2 = function.create_partial_grid(nbre_line, nbre_corners, width_sensor, separate_distance, x2, y2)
    grid_3 = function.create_partial_grid(nbre_line, nbre_corners, width_sensor, separate_distance, x3, y3)

    # Assemble the three partial grids into a complete grid
    complete_grid = grid_1 + grid_2 + grid_3
    return complete_grid


def auto_detection4(video, nbre_corners, nbre_lines, max_frames=100,  rotated_angle=None, crop=None):
    """
    Apply image processing techniques and the detection algorithm to detect automatically corners (vertexes of sensors)
    :param nbre_corners: Number of corners to detect per line
    :param nbre_lines: Number of lines to detect  x-axis
    :param video: Video object for processing frames.
    :param max_frames: Maximum number of frames to process.
    :param rotated_angle: (float): Rotation angle for frame rotation.
    :param crop: (tuple[int, int, int, int]): Crop parameters for frame cropping (left, right, top, bottom).
    :return: Detected corners (auto_grid).
    """
    for frame_count in range(max_frames):
        ret, frame = video.read()
        if not ret:
            break

        # Perform processing and detection on the frame
        frame_YUV = function.Convert_to_YUV(frame, Param.clipLimit, Param.tileGridSize)
        Cropping_frame = function.Cropping_Image(frame_YUV, rotated_angle, crop[0], crop[1], crop[2], crop[3])
        corrected_frame = function.correction(Cropping_frame, Param.gamma, Param.blur_size)

        # Detect sensors in each corrected frame
        all_corners_in_Matrix = function.get_Coords_sensors(corrected_frame, Param.params_Canny,
                                                            Param.nbre_lines_selected)

        # Check the detection criteria
        if function.check_detect_criteria(all_corners_in_Matrix, nbre_corners, nbre_lines):
            process_corners = function.processing_corners(all_corners_in_Matrix)
            messagebox.showinfo("Detection Over", f"Detection Complete after Frame N°: {frame_count + 1}")
            return process_corners

        # messagebox.showinfo("Frame Number", f"Frame N°: {frame_count + 1}")

    # Release the video capture
    if video.isOpened():
        video.release()

    return None


def algo_extraction(video, detect_mode, type_extract, crop=None, rotation_angle=None, coords_sensors_grid=None):
    """
    Launches the algorithm to process video frames and extract average pixel values.

    Args:
        video (cv2.VideoCapture): Video source to process.
        detect_mode (str): Detection mode ('auto' or 'manual').
        type_extract (str): Type of extraction ('Per frame' or 'Per second').
        crop (tuple[int, int, int, int]): Crop parameters for frame cropping (left, right, top, bottom).
        rotation_angle (float): Rotation angle for frame rotation.
        coords_sensors_grid

    Returns:
        tuple: A tuple containing average gray, green, red, and blue values for each sensor.
    """
    # Create lists to store the average pixel values
    average_gray_values = []
    average_red_values = []
    average_green_values = []
    average_blue_values = []


    # Define the sampling rate (number of frames per second)
    if type_extract == 'Per frame':
        sampling_rate = 1
    elif type_extract == 'Per second':
        sampling_rate = 30

    # Iterate through the frames of the video
    frame_count = 0
    frames_in_second = 0
    detection_complete = False
    detected_corners = None

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        if detect_mode == "auto":
            if not detection_complete:
                # Perform processing and detection on the frame
                frame_YUV = function.Convert_to_YUV(frame, Param.clipLimit, Param.tileGridSize)
                Cropping_frame = function.Cropping_Image(frame_YUV, rotation_angle, crop[0], crop[1], crop[2], crop[3])
                corrected_frame = function.correction(Cropping_frame, Param.gamma, Param.blur_size)

                # Detect sensors in each corrected frame
                all_corners_in_Matrix = function.get_Coords_sensors(corrected_frame, Param.params_Canny,
                                                                    Param.nbre_lines_selected)
                process_corners = function.processing_corners(all_corners_in_Matrix)

                # Check the detection criteria
                if function.check_detect_criteria(all_corners_in_Matrix):
                    # Use the detected corners in all other frames
                    detected_corners = process_corners
                    detection_complete = True
            # Original Image cropped
            Ori_crop = function.Cropping_Image(frame, rotation_angle, crop[0], crop[1], crop[2], crop[3])

        elif detect_mode == "manual":
            Ori_crop = function.Cropping_Image(frame, rotation_angle, crop[0], crop[1], crop[2], crop[3])

        frames_in_second += 1
        # Check if the current frame is the last frame within a second
        if frame_count % sampling_rate == sampling_rate - 1:
            if frames_in_second > 0:
                coords_sensors = function.extract_boxes_from_lines(
                    detected_corners) if detect_mode == "auto" else coords_sensors_grid
                average_gray, average_red, average_green, average_blue = function.calculate_average_pixel_values(
                    coords_sensors, Ori_crop)
                average_gray_values.append([frame_count] + average_gray)
                average_red_values.append([frame_count] + average_red)
                average_green_values.append([frame_count] + average_green)
                average_blue_values.append([frame_count] + average_blue)

        frame_count += 1
        print("frame N°:", frame_count)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    return average_gray_values, average_green_values, average_red_values, average_blue_values
