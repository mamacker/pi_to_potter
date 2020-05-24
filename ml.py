from os import listdir
from os.path import isfile, join, isdir
import cv2
from cv2 import *
import numpy as np

knn = None
nameLookup = {}

def nearPoints(p1, p2, dist):
    point2 = p2;
    if (p2["x"] != None):
        point2[0] = p2["x"];
        point2[1] = p2["y"];

    distance = math.sqrt( ((p1[0]-point2[0])**2)+((p1[1]-point2[1])**2) )
    print("Comparing: " + str(p1[0]) + " " + str(p1[1]) + " " + str(point2[0]) + " " + str(point2[1]) + " distance: " + str(distance));
    return distance < dist;

def TrainShapes(path_to_pictures) :
    global knn, nameLookup
    labelNames = []
    labelIndexes = []
    trainingSet = []
    numPics = 0
    dirCount = 0
    mypath = path_to_pictures + "/Pictures/"
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

    print("Training set...")
    print(trainingSet)

    print("Labels...")
    print(labelNames)

    print("Indexes...")
    print(labelIndexes)

    print("Lookup...")
    print(nameLookup)

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
    print(ret, result, neighbours, dist)
    if nameLookup[ret] is not None:
        print("Match: " + nameLookup[ret])
        return nameLookup[ret]
    else:
        return "error"

