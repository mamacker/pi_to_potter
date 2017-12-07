#/usr/bin/python
# -*- coding: utf-8 -*-
'''
  _\
  \
O O-O
 O O
  O
  
Raspberry Potter
Ollivander - Version 0.2 

Use your own wand or your interactive Harry Potter wands to control the IoT.  


Copyright (c) 2016 Sean O'Brien.  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
import io
import numpy as np
import argparse
import cv2
from cv2 import *
import picamera
import threading
from threading import Thread

import sys
import math
import time
import imutils

from imutils.video.pivideostream import PiVideoStream

print "Initializing point tracking"

# Parameters
lk_params = dict( winSize  = (25,25),
                  maxLevel = 4,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
blur_params = (4,4)
dilation_params = (5, 5)
movment_threshold = 80

active = False
frame_holder = None

# start capturing
vs = PiVideoStream().start()
time.sleep(2.0)
run_request = True
frame_holder = vs.read()
print "About to start."

def FrameReader():
    global frame_holder
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        cv2.flip(frame,1,frame)
        frame_holder = frame
        time.sleep(.150);

def Spell(spell):
    #clear all checks
    ig = [[0] for x in range(15)]
    #Invoke IoT (or any other) actions here
    cv2.putText(mask, spell, (5, 25),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0))
    if (spell=="Colovaria"):
	print "trinket_pin trigger"
    elif (spell=="Incendio"):
	print "switch_pin OFF"
	print "nox_pin OFF"
	print "incendio_pin ON"
    elif (spell=="Lumos"):
	print "switch_pin ON"
	print "nox_pin OFF"
	print "incendio_pin OFF"
    elif (spell=="Nox"):
	print "switch_pin OFF"
	print "nox_pin ON"
	print "incendio_pin OFF"
    print "CAST: %s" %spell


def IsGesture(a,b,c,d,i):
    #print "point: %s" % i
    #look for basic movements - TODO: trained gestures
    if ((a<(c-5))&(abs(b-d)<2)):
        ig[i].append("left")
    elif ((c<(a-5))&(abs(b-d)<2)):
        ig[i].append("right")
    elif ((b<(d-5))&(abs(a-c)<5)):
        ig[i].append("up")
    elif ((d<(b-5))&(abs(a-c)<5)):
        ig[i].append("down")
    #check for gesture patterns in array
    astr = ''.join(map(str, ig[i]))
    if "rightup" in astr:
        Spell("Lumos")
    elif "rightdown" in astr:
        Spell("Nox")
    elif "leftdown" in astr:
        Spell("Colovaria")
    elif "leftup" in astr:
        Spell("Incendio")    
    #print astr

dilate_kernel = np.ones(dilation_params, np.uint8)
def ProcessImage():
    global dilate_kernel, clahe, frame_holder
    frame = frame_holder.copy()
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    th, frame_gray = cv2.threshold(frame_gray, 230, 255, cv2.THRESH_BINARY);
    frame_gray = GaussianBlur(frame_gray,(9,9),1.5)
    frame_gray = cv2.dilate(frame_gray, dilate_kernel, iterations=1)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    frame_gray = clahe.apply(frame_gray)
    return frame_gray, frame

def FindWand():
    global old_frame,old_gray,p0,mask,ig,run_request
    try:
        last = time.time() 
        while True:
            now = time.time()
            if now - last > 4 or run_request:
                print "Running find..."
                old_gray, old_frame = ProcessImage()
                p0 = cv2.HoughCircles(old_gray,cv2.HOUGH_GRADIENT,3,50,param1=240,param2=8,minRadius=4,maxRadius=15)
                if p0 is not None:
                    p0.shape = (p0.shape[1], 1, p0.shape[2])
                    p0 = p0[:,:,0:2] 
                    mask = np.zeros_like(old_frame)
                    ig = [[0] for x in range(20)]
                last = time.time()
                run_request = False

            time.sleep(.3)
    except:
        e = sys.exc_info()[1]
        print "Error: %s" % e 
        exit

def TrackWand():
        global old_frame,old_gray,p0,mask,color,ig,img,frame, active, run_request
        print "Starting wand tracking..."
        try:
            color = (0,0,255)
            old_gray, old_frame = ProcessImage()

            # Take first frame and find circles in it
            p0 = cv2.HoughCircles(old_gray,cv2.HOUGH_GRADIENT,3,50,param1=240,param2=8,minRadius=4,maxRadius=15)
            if p0 is not None:
                p0.shape = (p0.shape[1], 1, p0.shape[2])
                p0 = p0[:,:,0:2]
                mask = np.zeros_like(old_frame)
        except:
            print "No points found"

	# Create a mask image for drawing purposes
        noPt = 0
	while True:
            try:
                active = False
                if p0 is not None:
                    active = True;
                    frame_gray, frame = ProcessImage();

                    # calculate optical flow
                    if len(p0) > 0:
                        print p0
                        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
                    else:
                        print "No points"
                        noPt = noPt + 1
                        if noPt > 5:
                            noPt = 0
                            run_request = True

                    # Select good points
                    good_new = p1[st==1]
                    good_old = p0[st==1]

                    # draw the tracks
                    for i,(new,old) in enumerate(zip(good_new,good_old)):
                        a,b = new.ravel()
                        c,d = old.ravel()
                        # only try to detect gesture on highly-rated points (below 10)
                        if (i<15):
                            IsGesture(a,b,c,d,i)
                        dist = math.hypot(a - c, b - d)
                        if (dist < movment_threshold):
                            cv2.line(mask, (a,b),(c,d),(0,255,0), 2)
                        cv2.circle(frame,(a,b),5,color,-1)
                        cv2.putText(frame, str(i), (a,b), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255)) 
                    img = cv2.add(frame,mask)
                    cv2.imshow("Raspberry Potter", img)
                else:
                    cv2.imshow("Original", frame)
                    run_request = True
                    print "Doing nuthing..."

                # Now update the previous frame and previous points
                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1,1,2)
            except IndexError:
                print "Index error - Tracking"  
                run_request = True
            except:
                e = sys.exc_info()[0]
                #print "Tracking Error: %s" % e 
            key = cv2.waitKey(10)
            if key in [27, ord('Q'), ord('q')]: # exit on ESC
                cv2.destroyAllWindows()
                break

try:
    t = Thread(target=FrameReader)
    t.start()
    find = Thread(target=FindWand)
    find.start()

    print "START incendio_pin ON and set switch off if video is running"
    TrackWand()
finally:
    cv2.destroyAllWindows()
    vs.stop()
