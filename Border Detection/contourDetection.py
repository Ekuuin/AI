import cv2 as cv
from matplotlib.pyplot import contour
import numpy as np

def orderPoints(points):
    n_points = np.concatenate([points[0], points[1], points[2], points[3]]).tolist()
    y = sorted(n_points, key=lambda n_points: n_points[1])
    x = y[0:2]
    x = sorted(x, key=lambda x: x[0])
    x2 = y[2:4]
    x2 = sorted(x2, key=lambda x2:x2[0])
    return [x[0], x[1], x2[0], x2[1]]

def aligmentImage(frame, width, height):
    alignmentFrame = None
    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    _, umbral = cv.threshold(grayFrame, 150, 255, cv.THRESH_BINARY)
    cv.imshow("Umbral", umbral)
    contours = cv.findContours(umbral, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
    contours = sorted(contours, key=cv.contourArea, reverse=True)[:1]
    
    for c in contours:
        epsilon = 0.01*cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, epsilon, True)
        if len(approx) == 4:
            points = orderPoints(approx)
            points2 = np.float32(points)
            points3 = np.float32([[0,0], [width, 0], [0, height], [width, height]])
            M = cv.getPerspectiveTransform(points2, points3)
            alignmentFrame = cv.warpPerspective(frame, M, (width, height))
    return alignmentFrame


captureVideo = cv.VideoCapture(1)

while True:
    webcamType, frames = captureVideo.read()
    if webcamType == False:
        break
    imgA6 = aligmentImage(frames, 677, 480)
    if(imgA6 is not None):
        points = []
        grayFrame = cv.cvtColor(imgA6, cv.COLOR_BGR2GRAY)
        blurFrame = cv.GaussianBlur(grayFrame, (5,5), 1)
        _, binaryFrame = cv.threshold(blurFrame, 0, 255, cv.THRESH_OTSU+cv.THRESH_BINARY_INV)
        contours = cv.findContours(binaryFrame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
        cv.drawContours(imgA6, contours, -1, (0,0,200), 1)
        suma1 = 0.0
        suma2 = 0.0
        for c2 in contours:
            area = cv.contourArea(c2)
            labels = cv.moments(c2)
            if labels["m00"] == 0:
                labels["m00"] = 1.0
            x = int(labels["m10"]/labels["m00"])
            y = int(labels["m01"]/labels["m00"])
            print(f"{area}")
            if area < 11400 and area > 10000:
                font = cv.FONT_HERSHEY_SIMPLEX
                cv.putText(imgA6, "5mxn", (x,y), font, 0.9, (0,0,255), 2)
                suma1+=5
            if area < 9000 and area > 7000:
                font = cv.FONT_HERSHEY_SIMPLEX
                cv.putText(imgA6, "1mxn", (x,y), font, 0.9, (0,0,255), 2)
                suma1+=1

            
        cv.imshow("Live", imgA6)
    
    if cv.waitKey(1) == ord("q"):
        break

captureVideo.release()
cv.destroyAllWindows()


