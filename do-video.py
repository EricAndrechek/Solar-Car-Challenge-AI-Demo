import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

np.warnings.filterwarnings('ignore')

# for the purposes of our code, we are using a demo video instead of a live camera feed, but this can be modified to use a camera feed
video = cv2.VideoCapture("data/car-crash.mp4")

size = (1920, 1080)
fps = video.get(5)
frame_count = video.get(7)
frame_num = 0

out = cv2.VideoWriter("crash.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

while(video.isOpened()):

    # get the frame out of the video, function returns whether the read happens correctly (we throw the value away with an underscore) along with the frame
    _, frame = video.read()

    cv2.imwrite("data/frame.jpg", frame)
    success,image = video.read()
    os.system("./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights data/frame.jpg")

    img = cv2.imread("predictions.jpg", cv2.IMREAD_COLOR)
    out.write(img)

    frame_num = frame_num + 1

    print((frame_num/300) * 100)


# close the video file
video.release()

out.release()