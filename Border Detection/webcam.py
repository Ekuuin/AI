from random import gauss
from typing import final
import cv2 as cv
import numpy as np

#Kernel used in morphologyEx function
kernel = np.ones((3,3), np.uint8)
#Starting video capture at device index = 1, must be 0 if you only have one camera
webcam = cv.VideoCapture(1)

#Evaluate if video capture have started
if not webcam.isOpened():
    print("No video found")
    exit()

#While video capture is working...
while(True):
    #Store captured frames in variable
    typeWebcam, frames = webcam.read()
    #Apply grayscale filter to frames
    grayVideo = cv.cvtColor(frames, cv.COLOR_BGR2GRAY)
    #Apply GaussianBlur to frames, matrix of 5x5
    gaussianVideo = cv.GaussianBlur(grayVideo, (5,5), 0)
    #Detect borders using Canny function
    cannyVideo = cv.Canny(gaussianVideo, 0, 100)
    #Use transformation to eliminate noise from image
    finalVideo = cv.morphologyEx(cannyVideo, cv.MORPH_CLOSE, kernel)
    cv.imshow("Live", finalVideo)
    #Find contours and show them
    contours, _ = cv.findContours(finalVideo, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    finalVideo = cv.drawContours(frames, contours, -1, (0,0,200), 2)

    

    if (cv.waitKey(1)==ord("q")):
        break

webcam.release()
cv.destroyAllWindows()