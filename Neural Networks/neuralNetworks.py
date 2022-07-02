import cv2 as cv
import os
import imutils 

path1 = "./testImages"

if not os.path.exists(path1):
    os.makedirs(path1)

noises = cv.CascadeClassifier("haarcascades\haarcascade_frontalface_default.xml")
id = 0

webcam = cv.VideoCapture(0)
if not webcam.isOpened():
    print("No video found")
    exit()

while True:
    response, frames = webcam.read()
    if response == False:
        break
    frames = imutils.resize(frames, width=640)
    grayFrame = cv.cvtColor(frames, cv.COLOR_BGR2GRAY)
    idCapture = frames.copy()

    faces = noises.detectMultiScale(grayFrame, 1.3, 5)

    for (x,y,c1,c2) in faces:
        cv.rectangle(frames, (x,y), (x+c1,y+c2), (255,0,0) , 2)
        capturedFace = idCapture[y:y+c2, x:x+c1]
        capturedFace = cv.resize(capturedFace, (150,150), interpolation=cv.INTER_CUBIC)
        cv.imwrite(path1 + "/img_{}.jpg".format(id), capturedFace)
        id+=1


    cv.imshow("Live", frames)

    if cv.waitKey(1) == ord("s"):
        break

webcam.release()
cv.destroyAllWindows()