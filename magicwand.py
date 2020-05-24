#/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import argparse
import cv2
from cv2 import *
import threading
from threading import Thread
import os
import sys, traceback
import math
import time
import select
import requests
import CameraLED
from six.moves import range
from six.moves import zip
from pathlib import Path

import music
import ble
from spells import Spells
import server
import ml


#Figure out where your code is...
home_address = str(Path.home())
parser = argparse.ArgumentParser(description='Cast some spells!  Recognize wand motions')
parser.add_argument('--train', help='Causes wand movement images to be stored for training selection.', action="store_true")
parser.add_argument('--setup', help='show camera view', action="store_true", default=True)
parser.add_argument('--home', help='The path to your pi_to_potter download.', default=home_address)
parser.add_argument('--background_subtract', help='User background subtraction', action="store_true")
parser.add_argument('--use_ble', help='Use the BLE system for spells', action="store_true")

args = parser.parse_args()

spells = Spells(args);

print(f'Perform training? {args.train}')
print(f'Show the original camera view? {args.setup}')
print(f'Use background subtraction? {args.background_subtract}')

print(f'Make sure the files are all at: {home_address}/pi_to_potter/...')

# You might not have this package.
try:
    camera = CameraLED.CameraLED()
    camera.off()
except:
    pass

print("Initializing point tracking")

# get the size of the screen
width = 800
height = 480

# Parameters
lk_params = dict( winSize  = (25,25),
                  maxLevel = 10,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
blur_params = (4,4)
dilation_params = (5, 5)
movment_threshold = 80

active = False

# Start capturing
cap = cv2.VideoCapture(0)
p0 = None #Points holder
frameMissingPoints = 0 # Current number of frames without points. (After finding a few.)

time.sleep(2.0)
run_request = True

# Use these to narrow the field of view.
yStart = 0;
yEnd = 360;
xStart = 0;
xEnd = 480;

ret, image_data = cap.read();
frame_holder = image_data
frame_no_background = image_data
frame_holder = frame_holder[yStart:yEnd, xStart:xEnd]
cv2.flip(frame_holder,1,frame_holder)
frame = None

print("About to start.")

def RemoveBackground():
    """
    Thread for removing background
    """
    global frame_holder, frame_no_background

    fgbg = cv2.createBackgroundSubtractorMOG2()
    t = threading.currentThread()
    while getattr(t, "do_run", True):
        frameCopy = frame_holder.copy()

        # Subtract Background
        fgmask = fgbg.apply(frameCopy, learningRate=0.001)
        frame_no_background = cv2.bitwise_and(frameCopy, frameCopy, mask = fgmask)
        time.sleep(0.01)

def FrameReader():
    global frame_holder
    t = threading.currentThread()
    while getattr(t, "do_run", True):
        ret, image_data = cap.read();
        frame = image_data[yStart:yEnd, xStart:xEnd]

        cv2.flip(frame,1,frame)
        frame_holder = frame
        time.sleep(.03);

point_aging = [];
def trim_points():
    global point_aging
    indexesToDelete = []
    index = 0;
    for old_point in point_aging:
        if (time.time() - old_point["when"] > 15):
            old_point["times_seen"] = old_point["times_seen"] - 1;
            if old_point["times_seen"] <= 0:
                indexesToDelete.append(index);
                deleted = True;
                break;
        index += 1;

    for i in reversed(indexesToDelete):
        del point_aging[i];

def GetPoints(image):
    global point_aging
    start_points = cv2.goodFeaturesToTrack(image, maxCorners=5, qualityLevel=0.0001, minDistance=5)

    # Clean out aged points.
    trim_points();
    return start_points;
    index = 0;
    indexesToDelete = [];
    if (start_points is not None):
        for point in start_points:
            print("point: " + str(point));
            if len(point_aging) == 0:
                point_aging.append({"x": point[0][0], "y":point[0][1], "times_seen": 0, "when":time.time()});

            found = False;
            deleted = False;
            for old_point in point_aging:
                if nearPoints(point[0], old_point, 15):
                    found = True;
                    old_point["times_seen"] = old_point["times_seen"] + 1;
                    old_point["when"] = time.time();
                    if old_point["times_seen"] > 5:
                        indexesToDelete.append(index);
                        deleted = True;
                        break;
                    else:
                        print("Times seen: " + str(old_point["times_seen"]) + " x: " + str(old_point["x"]) + " y: " + str(old_point["y"]));

            if not found:
                point_aging.append({"x": point[0][0], "y":point[0][1], "times_seen": 0, "when": time.time()});

            index = index + 1;

    for i in reversed(indexesToDelete):
        start_points = np.delete(start_points, i, 0);

    return start_points;

kernel = np.ones((5,5),np.uint8)
def ProcessImage():
    global frame_holder, frame_no_background
    if args.background_subtract:
        frame = frame_no_background.copy()
    else:
        frame = frame_holder.copy()
     
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.resize(frame_gray,(5*(xEnd - xStart), 5*(yEnd - yStart)), interpolation = cv2.INTER_CUBIC)
    th, frame_gray = cv2.threshold(frame_gray, 180, 255, cv2.THRESH_BINARY);
    frame_gray = cv2.dilate(frame_gray, kernel, iterations = 3)

    return frame_gray, frame

audioProcess = None;
def FindWand():
    global old_frame,old_gray,p0,mask, line_mask, run_request, audioProcess
    try:
        last = time.time()
        t = threading.currentThread()
        print("Find wand...")
        while getattr(t, "do_run", True):
            now = time.time()
            if run_request:
                old_gray, old_frame = ProcessImage()
                p0 = GetPoints(old_gray)
                if p0 is not None:
                    frameMissingPoints = 0;
                    mask = np.zeros_like(old_frame)
                    line_mask = np.zeros_like(old_gray)
                    run_request = False
                    music.play_wav(f'{home_address}/pi_to_potter/music/twinkle.wav')
                else:
                    music.stop_wav()
                last = time.time()

            time.sleep(.3)
    except cv2.error as e:
        None
        print("Err:")
        print(e)
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print("*** print_exception:")
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                                  limit=2, file=sys.stdout)

def TrackWand():
        global old_frame, old_gray, p0, mask, frameMissingPoints, line_mask, color, active, run_request
        print("Starting wand tracking...")
        color = (0,0,255)
        frame_gray = None
        good_new = None

    # Create a mask image for drawing purposes
        noPt = 0
        while True:
            try:
                active = False
                if p0 is not None:
                    active = True;
                    frame_gray, frame = ProcessImage();
                    if frame is not None:
                        cv2.imshow("frame_gray", frame_gray)
                        small = cv2.resize(frame, (120, 120), interpolation = cv2.INTER_CUBIC)
                        cv2.imshow("gray", small)
                        if (args.background_subtract):
                            cv2.imshow("frame_no_background", frame_no_background)
                        #cv2.moveWindow("gray", 0, 0);
                        #cv2.moveWindow("frame_gray", 150, 30);
                    else:
                        print("No frame.")

                    # calculate optical flow
                    newPoints = False
                    if p0 is not None and len(p0) > 0:
                        noPt = 0
                        try:
                            if old_gray is not None and frame_gray is not None:
                                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
                                newPoints = True
                        except cv2.error as e:
                            None
                            print("cv err")
                            print(e)
                        except:
                            print(".")
                            exc_type, exc_value, exc_traceback = sys.exc_info()
                            print("*** print_exception:")
                            traceback.print_exception(exc_type, exc_value, exc_traceback,
                                                      limit=2, file=sys.stdout)
                            continue
                    else:
                        noPt = noPt + 1
                        if noPt > 10:
                            try:
                                contours, hierarchy = cv2.findContours(line_mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                                if (contours is not None and len(contours) > 0):
                                    cnt = contours[0]
                                    x,y,w,h = cv2.boundingRect(cnt)
                                    crop = line_mask[y-10:y+h+10,x-30:x+w+30]
                                    result = ml.CheckShape(crop);
                                    cv2.putText(line_mask, result, (0,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255))
                                    spells.cast(result)
                                    if line_mask is not None:
                                        show_line_mask = cv2.resize(line_mask, (120, 120), interpolation = cv2.INTER_CUBIC)
                                        if args.setup is not True:
                                            show_line_mask = cv2.resize(line_mask, (width, height), interpolation = cv2.INTER_CUBIC)
                                        cv2.imshow("Raspberry Potter", show_line_mask)
                                    line_mask = np.zeros_like(line_mask)
                                    print("")
                            except:
                                exc_type, exc_value, exc_traceback = sys.exc_info()
                                print("FindSpell: *** print_exception:")
                                traceback.print_exception(exc_type, exc_value, exc_traceback,
                                                          limit=2, file=sys.stdout)
                            finally:
                                noPt = 0
                                run_request = True

                    if newPoints:
                        # Select good points
                        good_new = p1[st==1]
                        good_old = p0[st==1]

                        # draw the tracks
                        for i,(new,old) in enumerate(zip(good_new,good_old)):
                            a,b = new.ravel()
                            c,d = old.ravel()
                            cv2.line(line_mask, (a,b),(c,d),(255,255,255), 10)

                        if line_mask is not None:
                            #cv2.moveWindow("Raspberry Potter", 0, 200);
                            show_line_mask = cv2.resize(line_mask, (120, 120), interpolation = cv2.INTER_CUBIC)
                            if args.setup is not True:
                                cv2.namedWindow("Raspberry Potter", cv2.WND_PROP_FULLSCREEN)
                                cv2.setWindowProperty("Raspberry Potter",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
                                show_line_mask = cv2.resize(line_mask, (width, height), interpolation = cv2.INTER_CUBIC)
                            cv2.imshow("Raspberry Potter", show_line_mask)

                        # Now update the previous frame and previous points
                        if frame_gray is not None:
                            old_gray = frame_gray.copy()
                    else:
                        # This frame didn't have any points... lets go a couple more
                        # keep the old image( don't update it )
                        frameMissingPoints += 1;
                        if (frameMissingPoints >= 5 or p0 == None):
                            # Now update the previous frame and previous points
                            if frame_gray is not None:
                                old_gray = frame_gray.copy()
                            p0 = None;
                        else:
                            print("Chance: " + str(frameMissingPoints));

                else:
                    run_request = True
                    time.sleep(.3)

                if good_new is not None:
                    p0 = good_new.reshape(-1,1,2)

            except IndexError:
                run_request = True
            except cv2.error as e:
                None
                #print "Cv2 Error"
                #print sys.exc_info()
            except TypeError as e:
                None
                print("Type error.")
                exc_type, exc_obj, exc_tb = sys.exc_info()
                print((exc_type, exc_tb.tb_lineno))
            except KeyboardInterrupt as e:
                raise e
            except:
                None
                print(sys.exc_info())
                print("Tracking Error: %s" % e) 
                print(e)
            key = cv2.waitKey(10)
            if key in [27, ord('Q'), ord('q')]: # exit on ESC
                cv2.destroyAllWindows()
                break


try:
    ml.TrainShapes(f'{home_address}/pi_to_potter')
    t = Thread(target=FrameReader)
    t.do_run = True
    t.start()

    tr = Thread(target=RemoveBackground)
    if args.background_subtract:
        tr.do_run = True
        tr.start()

    find = Thread(target=FindWand)
    find.do_run = True
    find.start()

    server = Thread(target=server.runServer)
    server.do_run = True
    server.start()

    print("\n\n\n----------------------------------------------------------------------------------\n")
    print("Windows will open when there are points to see!")
    print("There should only be white spots corresponding to the wand.  If there are MORE this wont work.")
    print("Use an IR light source and a reflector, and ensure the camera does not see halogen light,")
    print("nor sunlight - both are big IR sources.")
    print("----------------------------------------------------------------------------------\n\n\n\n")
    time.sleep(2)
    TrackWand()
except KeyboardInterrupt:
    print("Shutting down...")
finally:
    t.do_run = False
    if args.background_subtract:
        tr.do_run = False
    find.do_run = False
    t.join()
    if args.background_subtract:
        tr.join()
    find.join()
    cv2.destroyAllWindows()
    sys.exit(1)

