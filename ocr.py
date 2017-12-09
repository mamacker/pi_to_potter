import numpy as np
import cv2

img = cv2.imread('digits.png')
test = cv2.imread('six.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
test_gray = cv2.cvtColor(test,cv2.COLOR_BGR2GRAY)

size = (20,20)
test_gray = cv2.resize(test_gray,size,interpolation=cv2.INTER_LINEAR)
cv2.imwrite("test_gray.png", test_gray);
# Now we split the image to 5000 cells, each 20x20 size
cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]

# Make it into a Numpy array. It size will be (50,100,20,20)
x = np.array(cells)
print "This is X"
print(x[0][0])

imgArr = np.array(test_gray).astype(np.float32)
print "This is imgArr"
print imgArr

# Now we prepare train_data and test_data.
train = x[:,:50].reshape(-1,400).astype(np.float32) # Size = (2500,400)
test = x[:,50:100].reshape(-1,400).astype(np.float32) # Size = (2500,400)
print "This is training data"
print train
print "This is train"
print(train[0], len(train[0]))

# Create labels for train and test data
k = np.arange(10)
train_labels = np.repeat(k,250)[:,np.newaxis]
test_labels = train_labels.copy()

# Initiate kNN, train the data, then test it with test data for k=1
knn = cv2.ml.KNearest_create()
knn.train(train,cv2.ml.ROW_SAMPLE, train_labels)

sample = imgArr.reshape(-1,400).astype(np.float32)
print "This is sample"
print(sample[0], len(sample[0]))

# Initiate kNN, train the data, then test it with test data for k=1
#knn = cv2.ml.KNearest_create()
#knn.train(train,cv2.ml.ROW_SAMPLE,train_labels)
ret,result,neighbours,dist = knn.findNearest(sample,k=5)

print ret
print result
print neighbours
print dist
