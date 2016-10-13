from __future__ import print_function
import numpy as np
import sys
import cv2
from timeit import default_timer as timer

from scipy.signal import fftconvolve
from scipy import misc, ndimage
from matplotlib import pyplot as plt
from skimage import feature
from skimage import color
from skimage import data
from skimage.io import imread
from pylab import imshow, show
from numba import cuda, vectorize, void, float32,float64, uint32, uint8, autojit, jit

# Local Binary Pattern function
from skimage.feature import local_binary_pattern
# To calculate a normalized histogram
from scipy.stats import itemfreq
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
import scipy as scipy
import pickle
import json

@cuda.jit(argtypes=[ uint8[:,:], uint32])
def create_fractal_kernel(image, iters):
    height, width = image.shape


    startX, startY = cuda.grid(2)
    gridX = cuda.gridDim.x * cuda.blockDim.x # stride in x
    gridY = cuda.gridDim.y * cuda.blockDim.y # stride in y

    for x in range(startX, width, 1):
        # real = xmin + x*pixel_size_x
        for y in range(startY, height, 1):
            center = image[y, x]
            code = 0
            code = (code | (1 if image[y-1][x-1] > center else 0 ) >>  7)
            code = (code | (1 if image[y-1][x] > center else 0 ) >> 6 )
            code = (code | (1 if image[y-1][x+1] > center else 0 )>> 5 )
            code = (code | (1 if image[y][x+1] > center else 0 )>> 4)
            code = (code | (1 if image[y+1][x+1] > center else 0 )>> 3 )
            code = (code | (1 if image[y+1][x] > center else 0 )>> 2)
            code = (code | (1 if image[y+1][x-1] > center else 0 )>> 1 )
            code = (code | (1 if image[y][x-1] > center else 0 )>> 0 )
            # imag = ymin + y*pixel_size_y
            # code = |(code, ( (1 if image[y+1][x+1] > center else 0 ), 3) )
            # image[y, x]  = code|((image[y-1][x-1] + image[y+1][x+1]) >> 2)
            image[y, x]  = code# image[y,x]
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def custom_lbp(image):
    height, width = image.shape
    for x in range(1, width-1):
        # real = xmin + x*pixel_size_x
        for y in range(1, height-1):
            center = image[y, x]
            code = 0
            code |=  (1 if image[y-1][x-1] > center else 0 ) <<  7
            code |=  (1 if image[y-1][x] > center else 0 ) << 6
            code |=  (1 if image[y-1][x+1] > center else 0 )<< 5
            code |=  (1 if image[y][x+1] > center else 0 )<< 4
            code |=  (1 if image[y+1][x+1] > center else 0 )<< 3
            code |=  (1 if image[y+1][x] > center else 0 )<< 2
            code |=  (1 if image[y+1][x-1] > center else 0 )<< 1
            code |=  (1 if image[y][x-1] > center else 0 )<< 0
            # imag = ymin + y*pixel_size_y
            # code = |(code, ( (1 if image[y+1][x+1] > center else 0 ), 3) )
            # image[y, x]  = code|((image[y-1][x-1] + image[y+1][x+1]) >> 2)
            image[y, x]  = code #image[ y, x]

    return image



FILENAME = "features6.json"
# Load All Featur vector to the ram

def load_allFeatures():
    featureVector = {}
    try :
        # featureVector =  pickle.load( open( FILENAME) )
        with open(FILENAME,  mode='r', encoding='utf-8') as handle:
            featureVector = json.load( handle )
    except IOError:
        with open(FILENAME, "w+") as f:
            pass
    return featureVector

featureVector = load_allFeatures()

# print(featureVector)
# print("----")
# print(featureVector[VECTOR_KEY][4])
# print(featureVector[VECTOR_KEY][4]["val"])


blockdim = (32,32) # (32, 8)
griddim =  (8, 8)#(32, 16)
iters = 50

radius = 4
# Number of points to be considered as neighbourers
no_points = 8 * radius

CurrHist = np.array([])
iterations = 60
firstTimer = True

VECTOR_KEY = "vectors1"
score = 0

face_cascade = cv2.CascadeClassifier('/home/ebadism/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/home/ebadism/opencv/data/haarcascades/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

@jit(target="cpu")
def calculateHist(gray):
    # gimage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gimage = cv2.resize(gray, (256,256)).astype(np.uint8)

    # Uniform LBP is used
    # lbp = custom_lbp(gimage)
    lbp = local_binary_pattern(gimage, no_points, radius, method='uniform')
    # Calculate the histogram
    # return lbp

    # ravel lineralize the 2-d aray to 1-d
    # itemfreq returns 2-d array sorted with bin to frequency count.
    x = itemfreq(lbp.ravel())
    # Normalize the histogram
    hist = x[:, 1]/sum(x[:, 1])
    return hist

# @vectorize([float64(float64,float64)])
def KL(a, b):
    # a = np.asarray(a, dtype=np.float)
    # b = np.asarray(b, dtype=np.float)
    c = np.sum(np.where(a != 0, a * np.log10(a / b), 0)) * 1000
    return c

def kullback_leibler_divergence(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    filt = np.logical_and(p != 0, q != 0)
    return np.sum(p[filt] * np.log2(p[filt] / q[filt]))

def histExists(hist):
    index = 0
    for i in featureVector[VECTOR_KEY]:
        b = i["val"]
        # print(b)
        score = scipy.stats.entropy(hist, b, base=None) * 1000

        if score < 25 :
            # Can return many other features
            return index, i["name"] , score
        index += 1
    return -1,"",0

def saveHist(hist):
    b = hist.tolist()
    s = len(featureVector)
    name = "newuser"+str(s)
    featureVector[VECTOR_KEY].append({"name":name,"val":b})
    return s , name, 1.0

def updateCurrent(index, x,y):
    detailList[str(index)][3] = x
    detailList[str(index)][4] = y

@jit(target="cpu")
def process(img, currsize):
    # Capture frame-by-frame
    score = 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    index = 0
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]

        if (currsize - len(faces)) != 0 :
            currsize = len(faces)
            hist = calculateHist(roi_gray)
            index, name, score = histExists(hist)
            if index > -1:
                # detailList.append([index,name,score, x,y])
                detailList[str(index)] = [index,name,score, x,y]
                print("---Found !---")
                # print([index,name,score])
                # print("score %s " % score)
            else:
                # save histogram
                # saveHist(hist)
                print("---tracking this person---")
                detailList[str(index)] = [saveHist(hist), x,y]
                # detailList.append([saveHist(hist), x,y])
                # score = cv2.compareHist(np.array(CurrHist, dtype=np.float32), np.array(hist, dtype=np.float32), CV_COMP_CHISQR)
                # score = kullback_leibler_divergence(hist, CurrHist)
                # score = KL(hist, CurrHist)
        else:
            # Update Location
            updateCurrent(index,x,y)
        index +=1
        # Comapre and Get score.thon
    # return currsize
dt = 0

global detailList
detailList = {}

global currsize
currsize = 0

while(cap.isOpened()):
# for i in range(iterations):
    # detailList.clear()
    start = timer()
    ret, img = cap.read()
    process(img, currsize)
    # CurrHist, score = process(img, CurrHist)
    dt = timer() - start
    count = 30
    for val,key in detailList.items():
        print(key)
        values = detailList[val]
        cv2.putText(img,str(values[1]), (values[3],values[4]), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
        # cv2.putText(img,"", (30,count), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
        count += 30


    # for values in detailList:
    #     # cv2.putText(img,str(values[1]), (30,count), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
    #     cv2.putText(img,str(values[1]), (values[3],values[4]), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
    #     count += 30
    # cv2.putText(img,"Score "+ str(detailList[0][1]), (detailList[0][3],detailList[0][4]), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
    # if(len(detailList) > 0):
    #     cv2.putText(img,"Score "+ str(detailList[0][0]), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print ("Processed a frame in %f s " % dt )


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


# a = np.arange(10).reshape(1,10)
# b = a.tolist()

# featureVector[VECTOR_KEY].append({"ebad1":"ali","val":b})
# print(featureVector)
# Save All Featur vector
def save_allFeatures():
    with open(FILENAME,  mode='w', encoding='utf-8') as handle:
        json.dump(featureVector, handle)

save_allFeatures()

# pickle.dump( featureVector, open( FILENAME) )




# plt.subplot(1, 2, 1)
# plt.title('CPU')
# plt.imshow(hist)
# plt.hist(hist, bins=range(255))

# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.title('GPU')
# plt.imshow(cvimage_gpu, cmap=plt.cm.gray)
# plt.axis('off')

plt.show()
