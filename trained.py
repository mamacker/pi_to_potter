#/usr/bin/python
# -*- coding: utf-8 -*-
import io
import numpy as np
import argparse
import cv2
from cv2 import *
import picamera
import threading
from threading import Thread
import pytesseract
from os import listdir
from os.path import isfile, join, isdir

import sys
import math
import time
import imutils

from imutils.video.pivideostream import PiVideoStream

print "Initializing point tracking"


parser = argparse.ArgumentParser(description='Cast some spells!  Recognize wand motions')
parser.add_argument('--train', help='Causes wand movement images to be stored for training selection.', action="store_true")

args = parser.parse_args()
print(args.train)

# Parameters
lk_params = dict( winSize  = (25,25),
                  maxLevel = 7,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
blur_params = (4,4)
dilation_params = (5, 5)
movment_threshold = 80

active = False

# start capturing
vs = PiVideoStream().start()
time.sleep(2.0)
run_request = True
frame_holder = vs.read()
frame = None
print "About to start."

knn = None
nameLookup = {}

def TrainOcr() :
    global knn, nameLookup
    labelNames = []
    labelIndexes = []
    trainingSet = []
    numPics = 0
    dirCount = 0
    mypath = "./Pictures/"
    for d in listdir(mypath):
        if isdir(join(mypath, d)):
            nameLookup[dirCount] = d
            dirCount = dirCount + 1
            for f in listdir(join(mypath,d)):
                if isfile(join(mypath,d,f)):
                    labelNames.append(d)
                    labelIndexes.append(dirCount-1)
                    trainingSet.append(join(mypath,d,f));
                    numPics = numPics + 1

    print "Training set..."
    print trainingSet

    print "Labels..."
    print labelNames

    print "Indexes..."
    print labelIndexes

    print "Lookup..."
    print nameLookup

    samples = []
    for i in range(0, numPics):
        img = cv2.imread(trainingSet[i])
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        samples.append(gray);
        npArray = np.array(samples)
        shapedArray = npArray.reshape(-1,400).astype(np.float32);

    # Initiate kNN, train the data, then test it with test data for k=1
    knn = cv2.ml.KNearest_create()
    knn.train(shapedArray, cv2.ml.ROW_SAMPLE, np.array(labelIndexes))

lastTrainer = None
def CheckOcr(img):
    global knn, nameLookup, args, lastTrainer

    size = (20,20)
    test_gray = cv2.resize(img,size,interpolation=cv2.INTER_LINEAR)
    if args.train and img != lastTrainer:
        cv2.imwrite("Pictures/char" + str(time.time()) + ".png", test_gray)
        lastTrainer = img
    imgArr = np.array(test_gray).astype(np.float32)
    sample = imgArr.reshape(-1,400).astype(np.float32)
    ret,result,neighbours,dist = knn.findNearest(sample,k=5)
    print "Match: " + nameLookup[ret]
    print ret, result, neighbours

def FrameReader():
    global frame_holder
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        cv2.flip(frame,1,frame)
        frame_holder = frame
        time.sleep(.03);

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


def GetPoints(image):
    #p0 = cv2.HoughCircles(image,cv2.HOUGH_GRADIENT,3,50,param1=240,param2=8,minRadius=2,maxRadius=15)

    p0 = cv2.goodFeaturesToTrack(image, 5, .01, 30)
    '''
    if p0 is not None:
        p0.shape = (p0.shape[1], 1, p0.shape[2])
        p0 = p0[:,:,0:2] 
    '''
    return p0;

dilate_kernel = np.ones(dilation_params, np.uint8)
def ProcessImage():
    global dilate_kernel, clahe, frame_holder
    frame = frame_holder.copy()
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    th, frame_gray = cv2.threshold(frame_gray, 230, 255, cv2.THRESH_BINARY);

    return frame_gray, frame

def FindWand():
    global old_frame,old_gray,p0,mask, line_mask, ig,run_request
    try:
        last = time.time() 
        while True:
            now = time.time()
            if run_request:
                old_gray, old_frame = ProcessImage()
                p0 = GetPoints(old_gray)
                if p0 is not None:
                    mask = np.zeros_like(old_frame)
                    line_mask = np.zeros_like(old_gray)
                    ig = [[0] for x in range(20)]
                    run_request = False
                last = time.time()

            time.sleep(.3)
    except:
        e = sys.exc_info()[1]
        print "Error: %s" % e 
        exit

def TrackWand():
        global old_frame,old_gray,p0,mask, line_mask, color,ig,frame, active, run_request
        print "Starting wand tracking..."
        color = (0,0,255)

	# Create a mask image for drawing purposes
        noPt = 0
	while True:
            try:
                active = False
                if p0 is not None:
                    active = True;
                    frame_gray, frame = ProcessImage();
                    cv2.imshow("Original", frame_gray)

                    # calculate optical flow
                    if len(p0) > 0:
                        noPt = 0
                        try:
                            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
                        except:
                            print "."
                            continue
                    else:
                        noPt = noPt + 1
                        if noPt > 10:
                            try:
                                im2, contours,hierarchy = cv2.findContours(line_mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                                cnt = contours[0]
                                x,y,w,h = cv2.boundingRect(cnt)
                                print "####################"
                                crop = line_mask[y-10:y+h+10,x-30:x+w+30]
                                CheckOcr(crop);
                                print "-------------------"
                            finally:
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
                        cv2.line(line_mask, (a,b),(c,d),(255,255,255), 10)
                        cv2.circle(frame,(a,b),5,color,-1)
                        #cv2.putText(frame, str(i), (a,b), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255)) 

                    #img_mask = cv2.add(frame_gray,im2)
                    cv2.imshow("Raspberry Potter", line_mask)
                else:
                    cv2.imshow("Original", frame)
                    run_request = True
                    time.sleep(.3)

                # Now update the previous frame and previous points
                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1,1,2)
            except IndexError:
                print "Index error - Tracking"  
                run_request = True
            except:
                None
                print sys.exc_info()
                #print "Tracking Error: %s" % e 
            key = cv2.waitKey(10)
            if key in [27, ord('Q'), ord('q')]: # exit on ESC
                cv2.destroyAllWindows()
                break

try:
    TrainOcr()
    t = Thread(target=FrameReader)
    t.start()
    find = Thread(target=FindWand)
    find.start()

    print "START incendio_pin ON and set switch off if video is running"
    time.sleep(2)
    TrackWand()
finally:
    cv2.destroyAllWindows()
    vs.stop()
