from __future__ import print_function
import numpy as np
import sys
import cv2
from timeit import default_timer as timer
import math
import cmath
from scipy.signal import fftconvolve
from scipy import misc, ndimage
from matplotlib import pyplot as plt
from skimage import feature
from skimage import color
from skimage import data
from skimage.io import imread
from pylab import imshow, show
from numba import cuda, vectorize, void,int8, float32,float64, uint32, uint8, autojit, jit
# Local Binary Pattern function
from skimage.feature import local_binary_pattern
# To calculate a normalized histogram
from scipy.stats import itemfreq
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
import scipy as scipy
import pickle
import json
from sklearn.metrics import mean_squared_error


SEARCH_INDEX = 4
n = 256
threadCount = 32
BIN_COUNT = 34

# -------Loading Feature vectors-----------
FILENAME = "features.json"
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

global vectorSize
vectors = featureVector["vectors1"]

vectorSize = len(vectors)

# Get Only FetureVectors For GPU
src1 = np.arange(vectorSize*BIN_COUNT, dtype=np.float64).reshape(vectorSize,BIN_COUNT)
n = len(src1)
for i in range(n):
    src1[i] = vectors[i]["val"]

# -------Finished Loading Feature vectors-----------


# -------CPU Version of KL Divergence ------------
def kullback_leibler_divergence(p, q):
    return np.sum(p * np.log2(p / q))

def square(A, B):
    err =  np.sum((A - B) ** 2)
    return np.sqrt(err)
    # return np.sum(p * np.log2(p / q))
print("kullback_leibler_divergence")
SumOfKL = 0.0
for i in range(0,n):
    mse = mean_squared_error(vectors[i]["val"],src1[SEARCH_INDEX])
    # kl = square(vectors[i]["val"],src1[0])
    kl = kullback_leibler_divergence(src1[SEARCH_INDEX],vectors[i]["val"])
    SumOfKL += kl
    print('kl : ' , kl,' , mse : ', mse)
print('Sum of kl ', SumOfKL)

# -------Finished CPU Version of KL Divergence ------------


@jit([void(float64[:,:], float64[:], float64[:], int8)], target='cuda')
def hist_comp(arry, hist, result, index):

    # We have N threads per block
    # And We have one block only

    x = cuda.grid(1)

    R = cuda.shared.array(9, dtype=float64)


    # No of featureVectors
    # array.shape[0] == 9*34
    A = cuda.shared.array(shape=(9,34), dtype=float64)

    # Vecture To Compair
    # hist.shape[0] == BIN_COUNT == 34 ?
    B = cuda.shared.array(34, dtype=float64)
    for i in range(BIN_COUNT):
        B[i] = hist[i]

    A[x] = arry[x]



    cuda.syncthreads()

    # Do Actual Calculations.
    # i.e: kullback_leibler_divergence
    Sum = 0.00
    for i in range(BIN_COUNT):
        a = B[i]
        b = A[x][i]
        Sum += (a * (math.log(a/b) / math.log(2.0)))

    # R Contains the KL-Divergences
    R[x] = Sum
    cuda.syncthreads()

    # These Should be Shared Variables.
    Min = cuda.shared.array(1,dtype=float32)
    mIndex = cuda.shared.array(1,dtype=int8)
    Min = 0.0000000000
    mIndex = 0

    if x == 0:
        Min = R[x]
        mIndex = x

    cuda.syncthreads()
    if R[x] <= Min:
        Min = R[x]
        mIndex = x
    cuda.syncthreads()

    if x == mIndex :
        index=mIndex


def hist_cuda_test():

    histogram_array = src1#np.zeros(vectorSize*BIN_COUNT, dtype=np.int32).reshape(vectorSize,BIN_COUNT)

    # This will be calculated from the Camera's Image processed on GPU.
    # Lets hardcode it at the moment
    histogram = src1[SEARCH_INDEX]#np.zeros(BIN_COUNT, dtype=np.float32)
    results = np.zeros(9, dtype=np.float64)
    foundIndex = -1
    # use stream to trigger async memory transfer
    cstream = cuda.stream()
    ts = timer()
    # Increase Counter to measure the Efficiency
    count = 1
    for i in range(count):
        with cstream.auto_synchronize():

            # For Histogram Compairision.
            d_histogram_array = cuda.to_device(histogram_array, stream=cstream)
            d_histogram = cuda.to_device(histogram, stream=cstream)
            d_results = cuda.to_device(results, stream=cstream)
            d_foundIndex = cuda.to_device(foundIndex, stream=cstream)

            hist_comp[1, vectorSize, cstream](d_histogram_array,d_histogram,d_results,d_foundIndex)

            d_histogram_array.copy_to_host(histogram_array, stream=cstream)
            d_histogram.copy_to_host(histogram, stream=cstream)
            d_results.copy_to_host(results, stream=cstream)
            d_foundIndex.copy_to_host(foundIndex, stream=cstream)

    te = timer()
    print('GPU Process ',count," Iterations : in ", te - ts)
    print('histogram is')
    print(results)
    print('Found Index ', foundIndex)


hist_cuda_test()
