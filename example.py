from __future__ import absolute_import
import cv2
import numpy as np

yStart = 90;
yEnd = 170;
xStart = 110;
xEnd = 230;

frame_gray = cv2.imread('/home/pi/pi_to_potter/test.png')
kernel = np.ones((5,5),np.uint8)
frame_gray = cv2.cvtColor(frame_gray,cv2.COLOR_BGR2GRAY)
th, frame_gray = cv2.threshold(frame_gray, 190, 255, cv2.THRESH_BINARY);
frame_gray.convertTo(frame_gray, -1, 2, 0);
frame_gray = cv2.resize(frame_gray,(5*(xEnd - xStart), 5*(yEnd - yStart)), interpolation = cv2.INTER_CUBIC)
#th, frame_gray = cv2.threshold(frame_gray, 190, 255, cv2.THRESH_BINARY);
frame_gray = cv2.dilate(frame_gray, kernel, iterations = 3)
cv2.imshow("test",frame_gray);

cv2.waitKey();
