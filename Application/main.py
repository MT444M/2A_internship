import cv2
import function
from function import Param
import pandas as pd



# Load the video
video = cv2.VideoCapture("Data files/1st_record_47m.mp4")
# Variables to store the sensor corners
detected_corners = None
# Control variable to stop processing
detection_complete = False
# Create lists to store the average pixel values
average_gray_values = []
average_red_values = []
average_green_values = []
average_blue_values = []

# Define the output filename
output_filenames = Param.output_filenames


# Specify the value of n
n = 1
# Define the sampling rate (number of frames per second)
sampling_rate = 30  # Adjust this value based on your requirements
frames_in_second = 0

# Iterate through the frames of the video
frame_count = 0
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    if frame_count % n == 0:  # Check if the current frame is the nth frame
        if not detection_complete:
            # Perform processing and detection on the frame
            # Apply correction on each frame
            frame_YUV = function.Convert_to_YUV(frame, Param.clipLimit, Param.tileGridSize)
            Cropping_frame = function.Cropping_Image(frame_YUV, Param.rot_angle, Param.crop_left, Param.crop_right,
                                                     Param.crop_top, Param.crop_bottom)
            corrected_frame = function.correction(Cropping_frame, Param.gamma, Param.blur_size)

            # Detect sensors in each corrected frame
            all_corners_in_Matrix = function.get_Coords_sensors(corrected_frame, Param.params_Canny, Param.nbre_lines_selected)
            process_corners = function.processing_corners(all_corners_in_Matrix)

            # Check the detection criteria
            if function.check_detect_criteria(all_corners_in_Matrix):
                # Use the detected corners in all other frames
                detected_corners = process_corners
                detection_complete = True

        # Original Image cropped
        Ori_crop = function.Cropping_Image(frame, Param.rot_angle, Param.crop_left, Param.crop_right, Param.crop_top,
                                           Param.crop_bottom)
        frames_in_second += 1
        # Display the frame with the results
        if detected_corners is not None:
            coords_sensors = function.extract_boxes_from_lines(detected_corners)
            # Check if the current frame is the last frame within a second
            if frame_count % sampling_rate == sampling_rate - 1:
                # Calculate the average pixel values for each sensor within a second
                if frames_in_second > 0:
                    average_gray, average_red, average_green, average_blue = function.calculate_average_pixel_values(
                        coords_sensors, Ori_crop)
                    average_gray_values.append([frame_count] + average_gray)
                    average_red_values.append([frame_count] + average_red)
                    average_green_values.append([frame_count] + average_green)
                    average_blue_values.append([frame_count] + average_blue)

            for ligne_sensor in detected_corners:
                for case in ligne_sensor:
                    cv2.circle(Ori_crop, case, 3, (250, 0, 0), -1)

        # Display the image with corners
        cv2.imshow("Image with Corners", Ori_crop)

    # Update the frame count
    frame_count += 1
    print("frame N°: ", frame_count)
    # Wait for 1 millisecond before moving to the next frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()




# Create DataFrames with the average pixel values
len_coords_boxes = len(coords_sensors)
df_gray = pd.DataFrame(average_gray_values, columns=['Frame'] + [f'Sensor_{i+1}' for i in range(len_coords_boxes)])
df_red = pd.DataFrame(average_red_values, columns=['Frame'] + [f'Sensor_{i+1}' for i in range(len_coords_boxes)])
df_green = pd.DataFrame(average_green_values, columns=['Frame'] + [f'Sensor_{i+1}' for i in range(len_coords_boxes)])
df_blue = pd.DataFrame(average_blue_values, columns=['Frame'] + [f'Sensor_{i+1}' for i in range(len_coords_boxes)])

# Save the average pixel values to separate CSV files
df_gray.to_csv("Data files/" + output_filenames[0], index=False)
df_red.to_csv("Data files/" + output_filenames[1], index=False)
df_green.to_csv("Data files/" + output_filenames[2], index=False)
df_blue.to_csv("Data files/" + output_filenames[3], index=False)
