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
import os
import subprocess
from os.path import isfile, join, isdir
from gpiozero import LED
import sys, traceback
import math
import time
import v4l2capture
import select
import requests
from BaseHTTPServer import BaseHTTPRequestHandler,HTTPServer
from bluepy import btle
from bluepy.btle import Scanner, DefaultDelegate
import CameraLED
camera = CameraLED.CameraLED()
camera.off();

digitalLogger = LED(17)
otherpin = LED(27)

found = False
class ScanDelegate(DefaultDelegate):
    def __init__(self):
        DefaultDelegate.__init__(self)

    def handleDiscovery(self, dev, isNewDev, isNewData):
        global found;
        if isNewDev:
            print "Discovered device", dev.addr
            if (dev.addr == 'cb:22:99:ce:97:8f'):
                found = True
        elif isNewData:
            print "Received new data from", dev.addr

scanner = Scanner().withDelegate(ScanDelegate())

failures = 0;
def runScanAndSet(state):
    global found;
    found = False;
    peripheral = None;
    devices = scanner.scan(3)
    try:
        peripheral = btle.Peripheral('cb:22:99:ce:97:8f', btle.ADDR_TYPE_RANDOM)
        if (peripheral == None):
            failures += 1;
            if (failures < 10):
                runScanAndSet(state);
            else:
                failures = 0;
            return;
        failures = 0;
    except:
        failures += 1;
        if (failures < 10):
            runScanAndSet(state);
        else:
            failures = 0;
        return;

    finally:
        guid = '713d0003503e4c75ba943148f18d941e'
        characteristic = peripheral.getCharacteristics(uuid=guid)[0];
        if (state):
            turnOn(characteristic);
            turnOn(characteristic);
        if (not state):
            turnOff(characteristic);
            turnOff(characteristic);

def turnOn(characteristic):
    # Set Output
    command = bytearray(3);
    command[0] = 0x53; #S
    command[1] = 0x04;
    command[2] = 0x01;

    print str(command)
    characteristic.write(command);

    # Turn on
    command = bytearray(3);
    command[0] = 0x54; #T
    command[1] = 0x04;
    command[2] = 0x01;

    print str(command)
    characteristic.write(command);

def turnOff(characteristic):
    # Set Output
    command = bytearray(3);
    command[0] = 0x53; #S
    command[1] = 0x04;
    command[2] = 0x01;

    print str(command)
    characteristic.write(command);

    # Turn on
    command = bytearray(3);
    command[0] = 0x54; #T
    command[1] = 0x04;
    command[2] = 0x00;

    print str(command)
    characteristic.write(command);

bleState = False;
def toggleBLE():
    global bleState;
    bleState = not bleState;
    runScanAndSet(bleState);

print "Initializing point tracking"

parser = argparse.ArgumentParser(description='Cast some spells!  Recognize wand motions')
parser.add_argument('--train', help='Causes wand movement images to be stored for training selection.', action="store_true")
parser.add_argument('--setup', help='show camera view', action="store_true")

args = parser.parse_args()
print(args.train)
print(args.setup)

# This code checks to see if we should start full screen or not.
# If the file exists, we start in full screen.
f = None
try:
    f = open("/home/pi/pi_to_potter/ready.txt")
    # Do something with the file
except IOError:
    args.setup = True;
    print("File not accessible")
finally:
    if (f is not None):
        f.close()

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

# start capturing
#vs = PiVideoStream().start()
# Open the video device.
vs = v4l2capture.Video_device("/dev/video0")
vs.create_buffers(30)
vs.queue_all_buffers()
vs.start()
p0 = None #Points holder
frameMissingPoints = 0 # Current number of frames without points. (After finding a few.)

time.sleep(2.0)
run_request = True
select.select((vs,),(),())

yStart = 90;
yEnd = 170;
xStart = 110;
xEnd = 230;

image_data = vs.read_and_queue()
frame_holder = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR)
frame_holder = frame_holder[yStart:yEnd, xStart:xEnd]
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

def TrainShapes() :
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
def CheckShape(img):
    global knn, nameLookup, args, lastTrainer

    size = (20,20)
    try:
        test_gray = cv2.resize(img,size,interpolation=cv2.INTER_CUBIC)
    except:
        return "error"


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
        buf = np.frombuffer(image_data, dtype=np.uint8);

        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        frame = frame[yStart:yEnd, xStart:xEnd]

        cv2.flip(frame,1,frame)
        frame_holder = frame
        time.sleep(.03);

def Spell(spell):
    #Invoke IoT (or any other) actions here
    if (spell=="center"):
        None
    elif (spell=="circle"):
	print "Playing audio file..."
        os.system('mpg321 /home/pi/pi_to_potter/audio.mp3 &')
    elif (spell=="eight"):
        print "Togging digital logger."
        digitalLogger.toggle();
        None
    elif (spell=="left"):
        print "Toggling magic crystal."
        t = Thread(target=toggleBLE);
        t.do_run = True
        t.start()
        None
    elif (spell=="square"):
        print "Toggling 'other' pin."
        otherpin.toggle();
        None
    elif (spell=="swish"):
        None
    elif (spell=="tee"):
        None
    elif (spell=="triangle"):
        print "Toggling outlet."
        URL = "http://localhost:3000/device/t";
        r = requests.get(url = URL);
    elif (spell=="zee"):
        print "Toggling 'other' pin."
        otherpin.toggle();
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
    start_points = cv2.goodFeaturesToTrack(image, maxCorners=5, qualityLevel=0.0001, minDistance=5)

    # Clean out aged points.
    trim_points();
    return start_points;
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

kernel = np.ones((5,5),np.uint8)
def ProcessImage():
    global frame_holder
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
        print "Find wand..."
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
                    if audioProcess is not None:
                        audioProcess.kill();
                    try:
                        audioProcess = subprocess.Popen(["/usr/bin/aplay", '/home/pi/pi_to_potter/twinkle.wav']);
                    except:
                        if audioProcess is not None:
                            audioProcess.kill();
                            audioProcess = None
                else:
                    if audioProcess is not None:
                        audioProcess.kill();
                        audioProcess = None
                '''
                else:
                    cv2.imwrite("nowand/char" + str(time.time()) + ".png", old_frame);
                '''
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
        global old_frame, old_gray, p0, mask, frameMissingPoints, line_mask, color, active, run_request
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
                        cv2.imshow("frame_gray", frame_gray)
                        small = cv2.resize(frame, (120, 120), interpolation = cv2.INTER_CUBIC)
                        cv2.imshow("gray", small)
                        cv2.moveWindow("gray", 0, 0);
                        cv2.moveWindow("frame_gray", 150, 30);
                    else:
                        print "No frame."

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
                                im2, contours, hierarchy = cv2.findContours(line_mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                                if (contours is not None and len(contours) > 0):
                                    cnt = contours[0]
                                    x,y,w,h = cv2.boundingRect(cnt)
                                    crop = line_mask[y-10:y+h+10,x-30:x+w+30]
                                    result = CheckShape(crop);
                                    cv2.putText(line_mask, result, (0,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255))
                                    Spell(result)
                                    if line_mask is not None:
                                        show_line_mask = cv2.resize(line_mask, (120, 120), interpolation = cv2.INTER_CUBIC)
                                        if args.setup is not True:
                                            show_line_mask = cv2.resize(line_mask, (width, height), interpolation = cv2.INTER_CUBIC)
                                        cv2.imshow("Raspberry Potter", show_line_mask)
                                    line_mask = np.zeros_like(line_mask)
                                    print ""
                            except:
                                exc_type, exc_value, exc_traceback = sys.exc_info()
                                print "FindSpell: *** print_exception:"
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
                            cv2.moveWindow("Raspberry Potter", 0, 200);
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
                            print "Chance: " + str(frameMissingPoints);

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
                print "Type error."
                exc_type, exc_obj, exc_tb = sys.exc_info()
                print(exc_type, exc_tb.tb_lineno)
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

class myHandler(BaseHTTPRequestHandler):
    #Handler for the GET requests
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type','text/html')
        self.end_headers()
        # Send the html message
        if (self.path == "/circle"):
            Spell("circle");
        if (self.path == "/square"):
            Spell("square");
        if (self.path == "/zee"):
            Spell("zee");
        if (self.path == "/eight"):
            Spell("eight");
        if (self.path == "/triangle"):
            Spell("triangle");
        self.wfile.write("{'done':true}")
        return


def runServer():
    import SimpleHTTPServer
    import SocketServer

    PORT = 8000
    try:
	#Create a web server and define the handler to manage the
	#incoming request
	server = HTTPServer(('', PORT), myHandler)
	print 'Started httpserver on port ' , PORT

	#Wait forever for incoming htto requests
	server.serve_forever()

    except KeyboardInterrupt:
	print '^C received, shutting down the web server'
	server.socket.close()

try:
    TrainShapes()
    t = Thread(target=FrameReader)
    t.do_run = True
    t.start()
    find = Thread(target=FindWand)
    find.do_run = True
    find.start()

    server = Thread(target=runServer)
    server.do_run = True
    server.start()

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

