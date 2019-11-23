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
from os import listdir
from os.path import isfile, join, isdir

import sys, traceback
import math
import time

import v4l2capture
import select

import CameraLED
camera = CameraLED.CameraLED()
camera.off();

print "Initializing point tracking"

parser = argparse.ArgumentParser(description='Cast some spells!  Recognize wand motions')
parser.add_argument('--train', help='Causes wand movement images to be stored for training selection.', action="store_true")

parser.add_argument('--circles', help='Use circles to select wand location', action="store_true")


args = parser.parse_args()
print(args.train)
print(args.circles)

# Parameters
lk_params = dict( winSize  = (25,25),
                  maxLevel = 7,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
blur_params = (4,4)
dilation_params = (5, 5)
movment_threshold = 80

active = False

# start capturing
#vs = PiVideoStream().start()
# Open the video device.
vs = v4l2capture.Video_device("/dev/video0")
vs.create_buffers(30)
vs.queue_all_buffers()
vs.start()
p0 = None

time.sleep(2.0)
run_request = True
select.select((vs,),(),())

image_data = vs.read_and_queue()
frame_holder = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR)
cv2.flip(frame_holder,1,frame_holder)

frame = None
print "About to start."

knn = None
nameLookup = {}

def nearPoints(p1, p2, dist):
    point2 = p2;
    if (p2["x"] != None):
        point2[0] = p2["x"];
        point2[1] = p2["y"];

    distance = math.sqrt( ((p1[0]-point2[0])**2)+((p1[1]-point2[1])**2) )
    print "Comparing: " + str(p1[0]) + " " + str(p1[1]) + " " + str(point2[0]) + " " + str(point2[1]) + " distance: " + str(distance);
    return distance < dist;

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
    print ret, result, neighbours, dist
    if nameLookup[ret] is not None:
        print "Match: " + nameLookup[ret]
        return nameLookup[ret]
    else:
        return "error"

def FrameReader():
    global frame_holder
    t = threading.currentThread()
    while getattr(t, "do_run", True):
        select.select((vs,),(),())
        image_data = vs.read_and_queue()
        frame = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        cv2.flip(frame,1,frame)
        frame_holder = frame
        time.sleep(.03);

def Spell(spell):
    #Invoke IoT (or any other) actions here
    return
    if (spell=="center"):
	print "trinket_pin trigger"
    elif (spell=="circle"):
	print "switch_pin OFF"
	print "nox_pin OFF"
	print "incendio_pin ON"
    elif (spell=="eight"):
	print "switch_pin ON"
	print "nox_pin OFF"
	print "incendio_pin OFF"
    elif (spell=="left"):
	print "switch_pin OFF"
	print "nox_pin ON"
	print "incendio_pin OFF"
    elif (spell=="square"):
        None
    elif (spell=="swish"):
        None
    elif (spell=="tee"):
        None
    elif (spell=="triangle"):
        None
    elif (spell=="zee"):
        None
    print "CAST: %s" %spell

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
    #p0 = None
    #if args.circles is not True:
    '''
    start_points = cv2.goodFeaturesToTrack(image, 5, .01, 5)
    else:
    '''
    start_points = cv2.HoughCircles(image,cv2.HOUGH_GRADIENT,3,50,param1=240,param2=8,minRadius=2,maxRadius=10)

    if start_points is not None:
        start_points.shape = (start_points.shape[1], 1, start_points.shape[2])
        start_points = start_points[:,:,0:2] 

    # Clean out aged points.
    trim_points();

    index = 0;
    indexesToDelete = [];
    if (start_points is not None):
        for point in start_points:
            print "point: " + str(point);
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
                        print "Times seen: " + str(old_point["times_seen"]) + " x: " + str(old_point["x"]) + " y: " + str(old_point["y"]);

            if not found:
                point_aging.append({"x": point[0][0], "y":point[0][1], "times_seen": 0, "when": time.time()});

            index = index + 1;

    for i in reversed(indexesToDelete):
        start_points = np.delete(start_points, i, 0);

    return start_points;

def ProcessImage():
    global frame_holder
    frame = frame_holder.copy()
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    if args.circles is not True:
        th, frame_gray = cv2.threshold(frame_gray, 230, 255, cv2.THRESH_BINARY);
    else:
        th, frame_gray = cv2.threshold(frame_gray, 200, 255, cv2.THRESH_BINARY);

    return frame_gray, frame

def FindWand():
    global old_frame,old_gray,p0,mask, line_mask, run_request
    try:
        last = time.time()
        t = threading.currentThread()
        print "Find wand..."
        while getattr(t, "do_run", True):
            now = time.time()
            if run_request:
                old_gray, old_frame = ProcessImage()
                p0 = GetPoints(old_gray)
                if p0 is not None:
                    mask = np.zeros_like(old_frame)
                    line_mask = np.zeros_like(old_gray)
                    run_request = False
                last = time.time()

            time.sleep(.3)
    except cv2.error as e:
        None
        print "Err:"
        print e
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print "*** print_exception:"
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                                  limit=2, file=sys.stdout)

def TrackWand():
        global old_frame,old_gray,p0,mask, line_mask, color, active, run_request
        print "Starting wand tracking..."
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
                        cv2.imshow("Gray", frame)

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
                            print "cv err"
                            print e
                        except:
                            print "."
                            exc_type, exc_value, exc_traceback = sys.exc_info()
                            print "*** print_exception:"
                            traceback.print_exception(exc_type, exc_value, exc_traceback,
                                                      limit=2, file=sys.stdout)
                            continue
                    else:
                        noPt = noPt + 1
                        if noPt > 10:
                            try:
                                im2, contours,hierarchy = cv2.findContours(line_mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                                cnt = contours[0]
                                x,y,w,h = cv2.boundingRect(cnt)
                                crop = line_mask[y-10:y+h+10,x-30:x+w+30]
                                result = CheckOcr(crop);
                                cv2.putText(line_mask, result, (0,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255))
                                Spell(result)
                                if line_mask is not None:
                                    cv2.imshow("Raspberry Potter", line_mask)
                                line_mask = np.zeros_like(line_mask)
                                print ""
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
                            cv2.imshow("Raspberry Potter", line_mask)
                else:
                    run_request = True
                    time.sleep(.3)

                # Now update the previous frame and previous points
                if frame_gray is not None:
                    old_gray = frame_gray.copy()
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
                #print "Type error."
                #exc_type, exc_obj, exc_tb = sys.exc_info()
                #print(exc_type, exc_tb.tb_lineno)
            except KeyboardInterrupt as e:
                raise e
            except:
                None
                print sys.exc_info()
                print "Tracking Error: %s" % e 
                print e
            key = cv2.waitKey(10)
            if key in [27, ord('Q'), ord('q')]: # exit on ESC
                cv2.destroyAllWindows()
                break

try:
    TrainOcr()
    t = Thread(target=FrameReader)
    t.do_run = True
    t.start()
    find = Thread(target=FindWand)
    find.do_run = True
    find.start()

    print "START incendio_pin ON and set switch off if video is running"
    time.sleep(2)
    TrackWand()
except KeyboardInterrupt:
    print("Shutting down...")
finally:
    t.do_run = False
    find.do_run = False
    t.join()
    find.join()
    cv2.destroyAllWindows()
    vs.stop()
    sys.exit(1)
