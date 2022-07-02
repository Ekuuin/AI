import cv2
import matplotlib.pyplot as plt

img = cv2.imread("contorno.jpg")
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_,imgBinary = cv2.threshold(imgGray, 100, 255, cv2.THRESH_BINARY)
contours, jerarquia = cv2.findContours(imgBinary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
imgContours = cv2.drawContours(img, contours, 5, (0,255,0), 3)

cv2.imshow("Contours", imgContours)



cv2.waitKey(0)
cv2.destroyAllWindows()