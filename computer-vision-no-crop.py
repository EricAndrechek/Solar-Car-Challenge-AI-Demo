import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys


np.warnings.filterwarnings('ignore')


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

def create_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

global_left_fit_average = []
global_right_fit_average = []
def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    global global_left_fit_average
    global global_right_fit_average
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            
            # It will fit the polynomial and the intercept and slope
            parameters = np.polyfit((x1, x2), (y1, y2), 1) 
            slope, intercept = parameters
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    if (len(left_fit) == 0):
        left_fit_average = global_left_fit_average
    else:
        left_fit_average = np.average(left_fit, axis=0)
        global_left_fit_average = left_fit_average

    right_fit_average = np.average(right_fit, axis=0)
    global_right_fit_average = right_fit_average
    left_line = create_coordinates(image, left_fit_average)
    right_line = create_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])


# for the purposes of our code, we are using a demo video instead of a live camera feed, but this can be modified to use a camera feed
video = cv2.VideoCapture("data/" + sys.argv[1])

gf_output_path = "data/" + sys.argv[1].split('.')[0] + '-nc-gray.' + sys.argv[1].split(".")[1]
bf_output_path = "data/" + sys.argv[1].split('.')[0] + '-nc-blur.' + sys.argv[1].split(".")[1]
cf_output_path = "data/" + sys.argv[1].split('.')[0] + '-nc-canny.' + sys.argv[1].split(".")[1]
li_output_path = "data/" + sys.argv[1].split('.')[0] + '-nc-line.' + sys.argv[1].split(".")[1]
ci_output_path = "data/" + sys.argv[1].split('.')[0] + '-nc-final.' + sys.argv[1].split(".")[1]

size = (int(video.get(3)),int(video.get(4)))
fps = video.get(5)

gf_out = cv2.VideoWriter(gf_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size, 0)
bf_out = cv2.VideoWriter(bf_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size, 0)
cf_out = cv2.VideoWriter(cf_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size, 0)
li_out = cv2.VideoWriter(li_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
ci_out = cv2.VideoWriter(ci_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

# loop through every frame in the video, for a live video this would read each frame of the live feed in a similar fashion
while(video.isOpened()):

    # get the frame out of the video, function returns whether the read happens correctly (we throw the value away with an underscore) along with the frame
    _, frame = video.read()

    # since computational time is important, we convert our frame to grayscale to save time
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # set each pixel to the average of its neighboring pixels in order to reduce noise
    # 5x5 is the dimensions of the Gaussian Matrix
    blur_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    # get the derivative of x and y changes in pixel contrasts
    # take the inverse tangent of the x derivative over the y derivative to find the rough angle of the edge being calculated
    # this algorithm further compares the pixels to their neighbors along their edge angle
    # it further checks the pixels perpendicular to the edge line to find where the border of the edge appears
    # we further then have a rough edge determined (length, angle, and width)
    # the 50 and 150 numbers are given as the values to determine what is too much or too little contrast for the edge
    canny_frame = cv2.Canny(blur_frame, 50, 150)

    hough = cv2.HoughLinesP(canny_frame, 2, np.pi / 180, 100, np.array([]), minLineLength = 100, maxLineGap = 50)

    averaged_lines = average_slope_intercept(frame, hough) 
    line_image = display_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    gf_out.write(gray_frame)
    bf_out.write(blur_frame)
    cf_out.write(canny_frame)
    li_out.write(line_image)
    ci_out.write(combo_image)
  
# close the video file
video.release() 

gf_out.release()
bf_out.release()
cf_out.release()
li_out.release()
ci_out.release()
  
# destroy all the windows that is currently on
cv2.destroyAllWindows() 