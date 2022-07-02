import cv2
import numpy as np

valorGauss = 3
valorKernel = 35

img = cv2.imread("monedassoles.jpg")
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgGrayBlur = cv2.GaussianBlur(imgGray, (valorGauss,valorKernel), 0)
_, imgBinary = cv2.threshold(imgGrayBlur, 200, 255, cv2.THRESH_BINARY)
cannyImg = cv2.Canny(imgGrayBlur, 50, 100)

kernel = np.ones((valorKernel,valorKernel), np.uint8)
cannyImg = cv2.morphologyEx(cannyImg, cv2.MORPH_CLOSE, kernel)
cv2.imshow("Results Binary", cannyImg)

contours, hierarchy = cv2.findContours(cannyImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cannyImg = cv2.drawContours(img, contours, -1, (255,0,0) , 2)


cv2.imshow("Results Canny", cannyImg)

print(f"El numero de monedas es: {len(contours)}")

cv2.waitKey(0)
cv2.destroyAllWindows()