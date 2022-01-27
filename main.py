import cv2
import numpy as np

img = cv2.imread("map.png")
#cv2.imshow("frame 1", img)

imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#cv2.imshow("frame 2", imgGray)


w = int(img.shape[1]*0.6)
h = int(img.shape[0]*0.6)
imgRes = cv2.resize(img, (w,h), interpolation = cv2.INTER_AREA)
imgRes = cv2.cvtColor(imgRes, cv2.COLOR_RGB2HSV)

lr = np.array([0,100,100])
ur = np.array([100,255,255])
mask = cv2.inRange(imgRes, lr,ur)
cv2.imshow("frame 3 ", mask)




print(imgRes.shape)


cv2.waitKey(0)