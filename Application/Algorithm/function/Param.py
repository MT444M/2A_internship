import datetime
import pandas as pd

#=======================================================================================================================
#--------------------------------------------------Image Processing-------------------------------------------------
#=======================================================================================================================
# Local Correction
clipLimit = 3.0  # Limit for contrast enhancement in the adaptive histogram equalization (CLAHE) algorithm.
tileGridSize = (1, 1)  # Size of each tile in the grid used for computing the adaptive histogram.

# Cropping Image
rot_angle = -2.0  # rotation angle value
crop_left = 50  # number of pixels to crop on the left side
crop_right = -120  # number of pixels to crop on the right side
crop_top = 15  # number of pixels to crop at the top
crop_bottom = -35  # number of pixels to crop at the bottom

# Correction of gamma and noise
gamma = 1.5  # gamma correction value for image enhancement
blur_size = (9, 9)  # kernel size for the Gaussian blur filter to reduce noise


#=======================================================================================================================
#--------------------------------------------------Detection of sensors-------------------------------------------------
#=======================================================================================================================
###                        <<<<<<<<<<<<<<<<<<<<<<<<<<<<Main Params>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
params_Canny = (0, 40)
nbre_lines_selected = 150

#-----------------------------------------------------------------------------------------------------------------------
bilateral_filter_size = 7  # size of the bilateral filter for noise removal
parallel_threshold = 64  # intersections of 2 lines occurred on distance > {}*image_size is assumed as parallel
parallel_angle_threshold = 0.04  # merge two parallel line clusters if angle difference is < {} (in radians)
two_line_cluster_threshold = 1.0  # angle difference between two line clusters of chess table should be < {} (in
# radians)
dbscan_eps_intersection_clustering = 10
dbscan_eps_duplicate_elimination = 3
polynomial_degree = 3  # used for fitting polynomial to intersection data, the last step of the pipeline


#=======================================================================================================================
#--------------------------------------------------                    -------------------------------------------------
#=======================================================================================================================

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

#-----------------------------------------------------------------------------------------------------------------------
# Create a DataFrame to store the average pixel values
len_coords_boxes = 72
#df = pd.DataFrame(columns=['Frame'] + [f'Capteur_{i+1}' for i in range(len_coords_boxes)])
df_gray = pd.DataFrame(columns=['Frame'] + [f'Sensor_{i+1}' for i in range(len_coords_boxes)])
df_red = pd.DataFrame(columns=['Frame'] + [f'Sensor_{i+1}' for i in range(len_coords_boxes)])
df_green = pd.DataFrame(columns=['Frame'] + [f'Sensor_{i+1}' for i in range(len_coords_boxes)])
df_blue = pd.DataFrame(columns=['Frame'] + [f'Sensor_{i+1}' for i in range(len_coords_boxes)])

