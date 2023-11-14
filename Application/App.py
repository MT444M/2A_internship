#!/usr/bin/python3
import datetime
import os
import pathlib
import threading
import time
import tkinter.filedialog as filedialog
from tkinter import messagebox, simpledialog

import cv2
import numpy as np
import pandas as pd
import pygubu
from PIL import Image, ImageTk

import function

# Get the path to the desktop directory
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

# Open a dialog box to select the folder to install the data
data_files_path = filedialog.askdirectory(initialdir=desktop_path, title="Select Data Files Folder")

# Check if the user selected a folder
if data_files_path:
    # Create the "Data files" directory if it doesn't exist
    data_files_path = os.path.join(data_files_path, "Data files")
    if not os.path.exists(data_files_path):
        os.makedirs(data_files_path)


PROJECT_PATH = pathlib.Path(__file__).parent
PROJECT_UI = PROJECT_PATH / "TKinter_pygubu2.ui"


class TkinterPygubu2App:
    def __init__(self, master=None):
        self.builder = builder = pygubu.Builder()
        builder.add_resource_path(PROJECT_PATH)
        builder.add_from_file(PROJECT_UI)
        # Main widget
        self.mainwindow = builder.get_object("toplevel1", master)

        self.Crop_bottom = None
        self.Crop_left = None
        self.Crop_top = None
        self.crop_right = None
        self.Rotated_angle = None
        self.Input = None
        self.Detect_method = None
        self.Extract_type = None
        self.coor_clickedpoint = None
        self.grid_vertices_label = None
        self.zoom_value = None
        self.timer = None
        self.No_corners_line = None

        # Add the video attribute
        self.video = None
        self.webcam = None
        self.frame = None
        self.processing_complete = False
        self.frame_count_thread = None
        self.video_processing_thread = None

        self.y = None
        self.x = None
        self.grid_points = []  # Initialize grid_points as an empty list
        self.zoom_level = 1.0  # Initial zoom level
        self.zoomed_frame = None  # Variable to store the zoomed frame
        self.grid = None  # Grid of corners
        self.coords_boxes = None # 4 vertices of each sensor
        self.video_filepath = None  # define the video_filepath attribute
        self.auto_grid = None  # define the automatically determined grid
        self.auto_coords_sensors = None
        self.output_filenames = None
        self.save_data = None
        self.custom_value = None
        self.processing_complete = None
        self.launch_print = None

        self.webcam_index = None
        self.sampling_rate = 1
        self.frame_count = 0
        self.df_gray = None
        self.df_red = None
        self.df_blue = None
        self.df_green = None

        self.len_coords = None
        self.average_blue_values = None
        self.average_red_values = None
        self.average_green_values = None
        self.average_gray_values = None

        self.stop_event = False
        self.update_frame_active = None

        # Initialize stop_event as a threading.Event instance
        self.stop_event = threading.Event()

        builder.import_variables(self,
                                 ['Crop_bottom',
                                  'Crop_left',
                                  'Crop_top',
                                  'crop_right',
                                  'Rotated_angle',
                                  'Input',
                                  'Detect_method',
                                  'Extract_type',
                                  'coor_clickedpoint',
                                  'grid_vertices_label',
                                  'zoom_value',
                                  'timer',
                                  'No_corners_line',
                                  'launch_print',
                                  'save_data'
                                  ])

        builder.connect_callbacks(self)

        # Mutable data structure to store the frame
        self.frame_data = [None]
        # Initialize the update_canvas_active flag
        self.update_canvas_active = True

    def run(self):
        self.mainwindow.mainloop()

    def stop_processes(self):

        # Release video sources
        if self.video is not None:
            self.video.release()
            self.video = None

        if self.webcam is not None:
            self.webcam.release()
            # Sleep for a short duration to allow camera stream to fully stop
            time.sleep(0.5)
            self.webcam = None

        # Set the event to signal threads to stop gracefully
        self.stop_event.set()
        # Wait for threads to finish
        if self.frame_count_thread is not None:
            self.frame_count_thread.join(timeout=1)

        if self.video_processing_thread is not None:
            self.video_processing_thread.join(timeout=1)

        # Clear the canvas
        canvas = self.builder.get_object("canvas")
        canvas.delete("all")
        # Reinitialize any necessary variables and states
        self.processing_complete = False
        # Stop the update_canvas function
        self.update_canvas_active = False
        self.update_frame_active = False

    def update_canvas(self):

        # Retrieve the canvas widget from the builder
        canvas = self.builder.get_object("canvas")
        # Check if the update_canvas_active flag is set to False
        if not self.update_canvas_active:
            return

        # Clear the canvas
        canvas.delete("all")

        # Draw the frame on the canvas
        if self.frame is not None:
            # Convert the frame to PIL ImageTk format
            image = Image.fromarray(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))
            photo = ImageTk.PhotoImage(image)

            # Create an upper layer for the frame
            frame_layer = canvas.create_image(20, 20, anchor="nw", image=photo)
            canvas.image = photo  # Store a reference to prevent garbage collection
            canvas.tag_lower(frame_layer)


        if self.Detect_method.get() == "manual":
            if self.video or self.webcam:
                # Retrieve the rotation angle and crop parameters
                angle = self.Rotated_angle.get()
                crop = (self.Crop_left.get(), -self.crop_right.get(), self.Crop_top.get(), -self.Crop_bottom.get())

                # Read the next frame from the video or webcam
                if self.video:
                    ret, frame = self.video.read()
                else:
                    ret, frame = self.webcam.read()

                if ret:
                    # Rotate the frame based on the selected angle
                    rows, cols = frame.shape[:2]
                    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
                    rotated_frame = cv2.warpAffine(frame, M, (cols, rows))
                    # Perform cropping
                    cropped_frame = rotated_frame[crop[2]:crop[3], crop[0]:crop[1]]

                    # Convert the frame to PIL ImageTk format
                    image = Image.fromarray(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))
                    photo = ImageTk.PhotoImage(image)
                    # Update the image on the canvas
                    frame_layer = canvas.create_image(20, 20, anchor="nw", image=photo)
                    canvas.image = photo  # Store a reference to prevent garbage collection
                    canvas.tag_lower(frame_layer)

                    # Register a mouse click event handler on the canvas
                    canvas.bind("<Button-1>", self.canvas_click_handler)

        # Initialize grid_to_display and coords_sensors_to_display as empty lists
        grid_to_display = []
        coords_sensors_to_display = []
        # Determine which grid and coords_sensors to display based on detection mode
        if self.Detect_method.get() == "manual":
            if self.grid is not None:
                grid_to_display = self.grid
                coords_sensors_to_display = self.coords_boxes
        elif self.Detect_method.get() == "auto":
            if self.auto_grid is not None:
                grid_to_display = self.auto_grid
                coords_sensors_to_display = self.auto_coords_sensors
                grid_to_display = [[(x + 20, y + 20) for x, y in line] for line in grid_to_display]

        # Draw the grid on the canvas
        if grid_to_display:
            # Iterate over each line in the grid
            for line in grid_to_display:
                # Iterate over each corner in the line
                for corner in line:
                    # Retrieve the coordinates of the corner
                    x, y = corner
                    # Draw a circle representing the corner on the canvas
                    radius = 3
                    color = "red" if self.Detect_method.get() == "manual" else "blue"
                    raise_grid = canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=color)
                    canvas.tag_raise(raise_grid)



        # Draw the rectangles on the canvas
        for sensor in coords_sensors_to_display:
            x1, y1, x2, y2, x3, y3, x4, y4 = (
                sensor[0][0],
                sensor[0][1],
                sensor[1][0],
                sensor[1][1],
                sensor[2][0],
                sensor[2][1],
                sensor[3][0],
                sensor[3][1],
            )

            # Create the rectangle using OpenCV
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
            # Add the offset (20, 20) to the rectangle coordinates for auto-detection mode
            if self.Detect_method.get() == "auto":
                x += 20
                y += 20
            # Draw the rectangle on the canvas
            canvas.create_rectangle(x, y, x + w, y + h, outline="white")
            #canvas.tag_raise(rect)

        # Call this function again after a delay (e.g., 30 milliseconds)
        canvas.after(30, self.update_canvas)

    def disable_auto_light_correction(self, camera):
        # Vérifier si la caméra est ouverte
        if not camera.isOpened():
            raise ValueError("La caméra n'est pas ouverte.")

        # Désactiver la correction automatique de la lumière
        camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        camera.set(cv2.CAP_PROP_AUTO_WB, 0)


    def update_frame(self, video, crop_and_rotate=True):
        # Retrieve the canvas widget from the builder
        canvas = self.builder.get_object("canvas")

        # Read the first frame from the video
        ret, frame = video.read()
        if ret:
            # Perform cropping and rotation on the frame if enabled
            if crop_and_rotate:
                crop_left = self.Crop_left.get()
                crop_right = -self.crop_right.get()
                crop_top = self.Crop_top.get()
                crop_bottom = -self.Crop_bottom.get()

                angle = self.Rotated_angle.get()
                rows, cols = frame.shape[:2]
                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
                rotated_frame = cv2.warpAffine(frame, M, (cols, rows))

                cropped_frame = rotated_frame[crop_top:crop_bottom, crop_left:crop_right]

                # Update the self.frame variable with the cropped and rotated frame
                self.frame = cropped_frame

            else:
                # Update the self.frame variable with the original frame
                self.frame = frame

            # Convert the frame to PIL ImageTk format
            image = Image.fromarray(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))
            photo = ImageTk.PhotoImage(image)

            # Update the image on the canvas
            canvas.create_image(20, 20, anchor="nw", image=photo)
            canvas.image = photo  # Store a reference to prevent garbage collection
        # Check if manual detection is active
        if not self.update_frame_active:
            return  # Stop updating frames when manual detection is active

        # Call this function again after a delay (e.g., 30 milliseconds)
        canvas.after(30, lambda: self.update_frame(video, crop_and_rotate))


    def input_select(self, widget_id):
        # Retrieve the selected input source from self.Input
        selected_input = self.Input.get()

        self.update_frame_active = True

        # Handle the selected input source
        if selected_input == "video":
            # Open a file dialog to choose a video file
            self.video_filepath = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4")])
            if self.video_filepath:
                # Load the video using OpenCV
                self.video = cv2.VideoCapture(self.video_filepath)  # Set the self.video attribute
                # Start updating the frame on the canvas
                self.update_frame(self.video, crop_and_rotate=True)

        elif selected_input == "camera":
            # Use a dialog box to prompt the user to enter the webcam index
            self.webcam_index = simpledialog.askinteger("Webcam Index", "Enter the webcam index (e.g., 0, 1, 2):",
                                                   initialvalue=1)

            # Attempt to open the webcam capture with the user-provided index
            self.webcam = cv2.VideoCapture(self.webcam_index, cv2.CAP_DSHOW)
            # Check if the webcam is successfully opened
            if not self.webcam.isOpened():
                messagebox.showinfo("Error", f"Failed to open webcam {self.webcam_index}.")
                return

            # Set the desired frame rate (e.g., 30 FPS)
            desired_frame_rate = 15
            #self.webcam.set(cv2.CAP_PROP_FPS, desired_frame_rate)

            # Désactiver la correction automatique de la lumière
            self.disable_auto_light_correction(self.webcam)

            # Start updating the frame on the canvas
            self.update_frame(self.webcam, crop_and_rotate=True)


    def canvas_click_handler(self, event):
        # Retrieve the selected detection method from self.Detection_method
        selected_detection_method = self.Detect_method.get()
        if selected_detection_method == "manual":
            # Retrieve the clicked pixel coordinates
            self.x = event.x
            self.y = event.y
            # display the coordinate of the clicked point
            self.coor_clickedpoint.set("(x: " + str(self.x) + " y: " + str(self.y) + ")")

    def detect_method_select(self, widget_id):
        # Retrieve the selected detection method from self.Detect_method
        selected_detection_method = self.Detect_method.get()

        # Set the update_canvas_active flag to True
        self.update_canvas_active = True

        if selected_detection_method == "auto":
            if self.video or self.webcam:
                # Retrieve the rotation angle and crop parameters
                angle = self.Rotated_angle.get()
                crop = (self.Crop_left.get(), -self.crop_right.get(), self.Crop_top.get(), -self.Crop_bottom.get())

                if self.video:
                    # Restart the video
                    self.video.release()
                    self.video = cv2.VideoCapture(self.video_filepath)
                    video_source = self.video
                else:
                    video_source = self.webcam

                # Show a dialog box to get the values for nbre_corners and nbre_lines
                input_str = simpledialog.askstring("Input",
                                                   "Enter the number of sensors to detect x-axis (x2) and y-axis (x2)"
                                                   " \n separated by a comma:",
                                                   initialvalue="16,18")

                # Check if the user provided input
                if input_str is not None:
                    # Split the input string to get individual values for nbre_corners and nbre_lines
                    values = input_str.split(',')
                    if len(values) == 2:
                        nbre_corners = int(values[0].strip())
                        nbre_lines = int(values[1].strip())

                        # Create a thread for auto-detection
                        auto_detection_thread = threading.Thread(target=self.perform_auto_detection,
                                                                 args=(video_source, angle, crop, nbre_corners, nbre_lines))
                        auto_detection_thread.start()
                    else:
                        # Handle the case when the input format is incorrect
                        messagebox.showwarning("Warning",
                                               "Invalid input format. Please provide two values separated by a comma.")
                else:
                    # Handle the case when the user canceled the dialog
                    messagebox.showwarning("Warning", "Auto-detection canceled. Please try again.")


        elif selected_detection_method == "manual":
            # Set the manual_grid_definition flag to True when starting manual grid definition
            self.update_canvas()
            # Stop the update_frame function update_canvas will perform the display
            self.update_frame_active = False

    def perform_auto_detection(self, video_source, angle, crop, nbre_corners, nbre_lines):
        self.auto_grid = function.auto_detection4(video_source, nbre_corners, nbre_lines, 100, angle, crop)

        if self.auto_grid is not None:
            self.auto_coords_sensors = function.extract_boxes_from_lines(self.auto_grid)
            if self.auto_coords_sensors is not None:
                self.auto_grid = self.auto_grid.tolist()
                self.update_canvas()

    import threading

    def Valid_clickedpoint_coord(self):
        # Access the x and y values as self variables
        x = self.x
        y = self.y

        # Define and store the coordinates of the three points
        self.grid_points.append((x, y))

        self.grid_vertices_label.set(str(self.grid_points))

        # Check if three points have been clicked
        if len(self.grid_points) == 3:
            self.update_canvas_active = True
            no_corners_per_line = self.No_corners_line.get()
            if not no_corners_per_line or not no_corners_per_line.isdigit() or int(no_corners_per_line) <= 0 or int(
                    no_corners_per_line) > 18:
                no_corners_per_line = 16  # Set default value

            self.grid = function.create_grid(6, int(no_corners_per_line), 43, 11, self.grid_points)
            self.coords_boxes = function.extract_boxes_from_lines(self.grid)
            # Clear the grid_points list for the next set of clicks
            self.grid_points = []
        else:
            if self.grid is not None:
                self.grid.clear()
                self.grid = []
                self.coords_boxes = []
            self.update_canvas_active = False
        # Update the canvas with the modified frame
        self.update_canvas()

    def extract_method_select(self, widget_id):
        # Retrieve the selected extraction method from self.Extract_type
        selected_extraction_method = self.Extract_type.get()

        # Handle the selected extraction method
        if selected_extraction_method == "Per frame":
            self.sampling_rate = 1
        elif selected_extraction_method == "Per second":
            self.sampling_rate = 30
        elif selected_extraction_method == "custom":
            custom_value_extract = simpledialog.askstring("Custom Value", "Enter the custom value:")
            if custom_value_extract is not None:
                try:
                    self.sampling_rate = int(custom_value_extract)
                except ValueError:
                    messagebox.showerror("Invalid Input",
                                         "Please enter a valid integer value for the custom sampling rate.")
                    return
            else:
                # The user clicked cancel or closed the dialog, handle it accordingly (e.g., set a default value)
                self.sampling_rate = 1


    def launch_algorithm(self, widget_id):

        # Restart the video
        if self.video:
            self.video.release()
            self.video = cv2.VideoCapture(self.video_filepath)

        if self.webcam:
            self.webcam.release()
            self.webcam = cv2.VideoCapture(self.webcam_index, cv2.CAP_DSHOW)
            # Désactiver la correction automatique de la lumière
            self.disable_auto_light_correction(self.webcam)

        # Define the output filename
        # Get current time
        current_time = datetime.datetime.now()
        # Create a custom date format to include the time in the file name
        time_format = current_time.strftime("%Y-%m-%d_%H-%M-%S")

        # Define the output file name
        output_filenames = [
            f'P_value_{time_format}_Gray.csv',
            f'P_value_{time_format}_Red.csv',
            f'P_value_{time_format}_Green.csv',
            f'P_value_{time_format}_Blue.csv'
        ]
        self.output_filenames = output_filenames

        # Initialize variables for frame processing
        self.frame_count = 0

        self.stop_event = threading.Event()  # Initialize a threading event
        # Start the frame count and video processing threads
        self.frame_count_thread = threading.Thread(target=self.update_frame_count, args=(self.stop_event,))
        self.frame_count_thread.start()

        self.video_processing_thread = threading.Thread(target=self.process_video, args=(self.stop_event,))
        self.video_processing_thread.start()

        # Update the button text to "Stop & Save"
        self.save_data.set("Stop & Save")
        self.launch_print.set("In process ...")

    def process_video(self, stop_event):
        try:
            # Execute the algorithm and perform video processing here
            self.update_canvas_active = False
            self.update_frame_active = False


            detect_mode = self.Detect_method.get()
            crop = [self.Crop_left.get(), -self.crop_right.get(), self.Crop_top.get(), -self.Crop_bottom.get()]
            rotation_angle = int(self.Rotated_angle.get())

            if detect_mode == "manual":
                coords_sensors = self.coords_boxes
                # Apply offset correction to coords_sensors
                coords_sensors = [[(x - 20, y - 20) for x, y in sensor] for sensor in coords_sensors]
            else:
                coords_sensors = self.auto_coords_sensors

            # Create lists to store the average pixel values
            self.average_gray_values = []
            self.average_red_values = []
            self.average_green_values = []
            self.average_blue_values = []


            # Check if coords_sensors is not empty and there's video/webcam source
            if coords_sensors and (self.video or self.webcam):
                # Initialize len_coords with a default value
                self.len_coords = len(coords_sensors)
                while not stop_event.is_set():
                    frame_count = self.frame_count
                    sampling_rate = self.sampling_rate

                    # Read the next frame from the video or webcam
                    if self.video:
                        ret, frame = self.video.read()
                    else:
                        ret, frame = self.webcam.read()

                    if not ret:
                        break

                    # Check if the current frame is the last frame within a second
                    if frame_count % sampling_rate == 0:
                        # Calculate the average pixel values for each sensor within a second
                        # Original Image cropped
                        Ori_crop = function.Cropping_Image(frame, rotation_angle, crop[0], crop[1], crop[2], crop[3])
                        average_gray, average_red, average_green, average_blue = function.calculate_average_pixel_values(
                            coords_sensors, Ori_crop)
                        self.average_gray_values.append([self.frame_count] + average_gray)
                        self.average_red_values.append([self.frame_count] + average_red)
                        self.average_green_values.append([self.frame_count] + average_green)
                        self.average_blue_values.append([self.frame_count] + average_blue)
                        print(self.frame_count)

                    # Update frame count and frames_in_second
                    self.frame_count += 1

        except Exception:
            # If any exception occurs, show a simple error message in a messagebox
            messagebox.showerror("Error encountered",
                                 "Please restart the application. "
                                 "data may not be saved")

    def update_frame_count(self, stop_event):
        count_timer = 0
        total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT)) if self.video else None
        while not stop_event.is_set():
            count_timer += 1
            # Update the frame number on the app
            self.timer.set(str(count_timer) + "s")
            time.sleep(1)  # Adjust the sleep time as needed

            # Check if the video processing is complete (frame_count reaches total_frames)
            if total_frames is not None and self.frame_count >= total_frames:
                # Stop updating the timer
                break

    def stop_and_save(self, widget_id):
        # Set the event to signal threads to stop gracefully
        self.stop_event.set()
        # Wait for threads to finish
        if self.frame_count_thread is not None:
            self.frame_count_thread.join()
        if self.video_processing_thread is not None:
            self.video_processing_thread.join()
        # Clear the event to allow restarting the video processing thread
        self.stop_event.clear()

        # Create DataFrames with the average pixel values
        self.df_gray = pd.DataFrame(self.average_gray_values,
                                    columns=['Frame'] + [f'Sensor_{i + 1}' for i in range(self.len_coords)])
        self.df_red = pd.DataFrame(self.average_red_values,
                                   columns=['Frame'] + [f'Sensor_{i + 1}' for i in range(self.len_coords)])
        self.df_green = pd.DataFrame(self.average_green_values,
                                     columns=['Frame'] + [f'Sensor_{i + 1}' for i in range(self.len_coords)])
        self.df_blue = pd.DataFrame(self.average_blue_values,
                                    columns=['Frame'] + [f'Sensor_{i + 1}' for i in range(self.len_coords)])

        # Save the average pixel values to separate CSV files in the "Data files" directory
        self.df_gray.to_csv(os.path.join(data_files_path, self.output_filenames[0]), index=False)
        self.df_red.to_csv(os.path.join(data_files_path, self.output_filenames[1]), index=False)
        self.df_green.to_csv(os.path.join(data_files_path, self.output_filenames[2]), index=False)
        self.df_blue.to_csv(os.path.join(data_files_path, self.output_filenames[3]), index=False)


        self.save_data.set('Data saved')
        self.launch_print.set("Relaunch")

    def restart_application(self):
        # Show a message box to ask for confirmation
        restart_confirmation = messagebox.askyesno("Restart", "Are you sure you want to restart the application?")
        if restart_confirmation:
            self.update_frame_active = False
            # Stop ongoing processes
            self.stop_processes()
            # Reset the variables to their initial values
            self.Crop_bottom.set(1)
            self.Crop_left.set(0)
            self.Crop_top.set(0)
            self.crop_right.set(1)
            self.Rotated_angle.set(0)
            self.Detect_method.set("auto")
            self.Extract_type.set("Per frame")
            self.grid_vertices_label.set("...")
            self.coor_clickedpoint.set(". . .")
            self.timer.set("...")
            self.No_corners_line.set("No corners/line (16)")
            self.grid = []  # reinitialize grid_points as an empty list
            self.coords_boxes = []  # Initialize as an empty list
            self.auto_grid = []
            self.auto_coords_sensors = []
            self.output_filenames = []
            self.frame = None
            self.video_filepath = None
            self.save_data.set('. . .')
            self.launch_print.set("LAUNCH")
            self.frame_count = 0
            self.sampling_rate = 1

    def callback(self, event=None):
        pass


if __name__ == "__main__":
    app = TkinterPygubu2App()
    app.run()

