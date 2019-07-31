import cv2
import numpy
import numpy as np
lowerBound = np.array([100, 80, 40])
upperBound = np.array([140, 255, 255])
com = cv2.VideoCapture(0)
kernalOpen = np.ones((20,20))
kernalClose = np.ones((20,20))

while True:
	_, img = com.read()
	_2, img2 = com.read()
	img = cv2.resize(img, (700,500))
	#convert BGR to HSV
	imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	#create the mask
	mask = cv2.inRange(imgHSV, lowerBound, upperBound)
	#morphology
	maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernalOpen)
	maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernalClose)

	maskFinal = maskClose
	_, conts, h=cv2.findContours(maskFinal.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	cv2.drawContours(img, conts, -1, (0, 255, 0), 3) 
	cv2.imshow("maskOpen",maskOpen)
	cv2.imshow("mask", mask)	
	cv2.imshow("cam", img)
	cv2.imshow("Original", img2)
	cv2.waitKey(25)
	