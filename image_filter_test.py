'''
Demonstrate the significant performance difference between transferring
regular host memory and pinned (pagelocked) host memory.
'''
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
from numba import cuda, vectorize, void, int32, uint32, uint8, autojit, jit

# Local Binary Pattern function
from skimage.feature import local_binary_pattern
# To calculate a normalized histogram
from scipy.stats import itemfreq
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
import scipy as scipy
import pickle
import json

import numpy as np
from skimage import color

from numba import vectorize, int32, cuda, jit, void
import math
# I distribute kernel code in two Phaces.
# 1. ________Copy data to shared memory_______:
#
#       for 5*5 screen I am having 5 threads per block
#       Totaling 25 blocks
#       but my shared memory and will be of 8*8 size
#       Have to add apron for stensil function

# 2. _______ Apply the stencil function over the shared data.



# ------------ Correct Configuration for 256*256 window
# bpg = (8,8)
# # Each block size wopuld be 32*32
# tpb = (32,32)


# for i in range(1, 40):
#     print('i : ',i," ans :  ", 256/i)


featureVector = []
VECTOR_KEY = "vectors"
def save_allFeatures(hist):
    with open("myvect.json",  mode='w', encoding='utf-8') as handle:
        b = hist.tolist()

        featureVector.append({"name": 24, "val": b})
        json.dump(featureVector, handle)



# Configuration for 10*10 window
# n = 20
# threadCount = 10

# Configuration for 256*256 window
n = 256
threadCount = 32

# bpg = 64
# # Each block size wopuld be 32*32
# tpb = 32
# n = bpg * tpb * tpb


radius = 1
BIN_COUNT = 255

@jit([void(int32[:,:], int32[:])], target='cuda')
def thresholding(arry, hist):

    # We have 10*10 threads per block

    A = cuda.shared.array(shape=(32,32), dtype=int32)

    # H = cuda.shared.array(BIN_COUNT, dtype=int32)

    x,y = cuda.grid(2)

    ty = cuda.threadIdx.x
    tx = cuda.threadIdx.y

    A[ty,tx] = arry[x,y]


    cuda.syncthreads()

    threadCountX = A.shape[0] - 1
    threadCountY = A.shape[1] - 1
    # If within x range and y range then calculate the LBP discriptor along
    # with histogram value to specific bin

    # Other wise Ignore the Value
    if (ty > 0 and  (threadCountX-ty) > 0 ) and (tx > 0 and (threadCountY-tx) > 0):
        # You can do the Processing here. ^_^
        code = 0
        #  We need to make sure that each value is accessable to each thread
        #  TODO: make them atomic
        center = A[ty, tx]
        # Lets try averaging,
        # code += A[ty-1][x-1]
        # # cuda.syncthreads()
        # code += A[ty][tx-1]
        # # cuda.syncthreads()
        # code += A[ty+1][tx-1]
        # # cuda.syncthreads()
        # code += A[ty+1][tx]
        # # cuda.syncthreads()
        # code += A[ty+1][tx+1]
        # # cuda.syncthreads()
        # code += A[ty][tx+1]
        # # cuda.syncthreads()
        # code += A[ty-1][tx+1]
        # # cuda.syncthreads()
        # code += A[ty-1][tx-1]
        #
        # code = code / 8

        code = 0 if center > 150  else 255

        # Compiler optimization: By loop unrolling
        # turns out twice faster than rolled version for over
        # 16*16

        # cuda.syncthreads()
        # code |= (1 if A[ty-1][x-1] > center else 0 ) <<  7
        # # cuda.syncthreads()
        # code |= (1 if A[ty][tx-1] > center else 0)  << 6
        # # cuda.syncthreads()
        # code |= (1 if A[ty+1][tx-1] > center else 0 )<< 5
        # # cuda.syncthreads()
        # code |= (1 if A[ty+1][tx] > center else 0 ) << 4
        # # cuda.syncthreads()
        # code |= (1 if A[ty+1][tx+1] > center else 0 ) << 3
        # # cuda.syncthreads()
        # code |= (1 if A[ty][tx+1] > center else 0 ) << 2
        # # cuda.syncthreads()
        # code |= (1 if A[ty-1][tx+1] > center else 0 )<< 1
        # # cuda.syncthreads()
        # code |= (1 if A[ty-1][tx-1] > center else 0) << 0
        # arry[x,y] = code
        # Since atomic add; adds value to the existing value
        # Need to figure out the fraction to be added in the previous value
        code = ( code - center)

        A[ty,tx] = code

        # cuda.atomic.add(A, (ty,tx),code)
        cuda.syncthreads()

        val  = A[ty,tx]
        cuda.atomic.add(arry, (x,y),val)
        cuda.syncthreads()
        # This Atomic Operation is equivalent to  hist[code % 256] += 1
        ind = code % BIN_COUNT

        cuda.atomic.add(hist, ind, 1)




        # val = H[ind]
        # cuda.syncthreads()
        # cuda.atomic.add(hist, ind, val)
    # TODO: May be Let each block creates its local histogram and
    #  call another kernel to add it into the global memory

    # else:
    #     arry[x,y] = 255
        # cuda.atomic.add(arry, (x,y),0)
    # Total No of Image
    # if ((y > 0 and  y < 65536 ) and (x > 0 and x < 65536)):
    #     # Now Lets Merge all the histograms
    #     # Do Linear Indexing
    #     # unique block index inside a 3D block grid
    #     LinearblockId = cuda.blockIdx.x + cuda.blockIdx.y * cuda.gridDim.x #//2D
    #     # global unique thread index, block dimension uses only x-coordinate
    #     LinearthreadId = LinearblockId * cuda.blockDim.x + cuda.threadIdx.x
    #     cuda.syncthreads()
    #     # print(LinearthreadId)
    #     val = LinearthreadId % BIN_COUNT
    #
    #     histVal = H[val]
    #
    #     cuda.atomic.add(hist,val, histVal)


@jit([void(int32[:,:], int32[:])], target='cuda')
def unsharp_masking(arry, hist):

    # We have 10*10 threads per block

    A = cuda.shared.array(shape=(32,32), dtype=int32)

    # H = cuda.shared.array(BIN_COUNT, dtype=int32)

    x,y = cuda.grid(2)

    ty = cuda.threadIdx.x
    tx = cuda.threadIdx.y

    A[ty,tx] = arry[x,y]


    cuda.syncthreads()

    threadCountX = A.shape[0] - 1
    threadCountY = A.shape[1] - 1
    # If within x range and y range then calculate the LBP discriptor along
    # with histogram value to specific bin

    # Other wise Ignore the Value
    if (ty > 0 and  (threadCountX-ty) > 0 ) and (tx > 0 and (threadCountY-tx) > 0):
    #     # You can do the Processing here. ^_^
        code = 0
        #  We need to make sure that each value is accessable to each thread
        #  TODO: make them atomic
        center = A[ty, tx]
        # Lets try averaging,
        code += A[ty-1][tx-1]*-1
        # cuda.syncthreads()
        code += A[ty][tx-1]*-2
        # cuda.syncthreads()
        code += A[ty+1][tx-1]*-1
        # cuda.syncthreads()
        code += A[ty+1][tx]*-2
        # cuda.syncthreads()
        code += A[ty+1][tx+1]*-1
        # cuda.syncthreads()
        code += A[ty][tx+1]*-2
        # cuda.syncthreads()
        code += A[ty-1][tx+1]*-1
        # cuda.syncthreads()
        code += A[ty-1][tx-1]*-2

        code = code / 16

        # code = 0 if center > 150  else 255

        # Compiler optimization: By loop unrolling
        # turns out twice faster than rolled version for over
        # 16*16

        # cuda.syncthreads()
        # code |= (1 if A[ty-1][x-1] > center else 0 ) <<  7
        # # cuda.syncthreads()
        # code |= (1 if A[ty][tx-1] > center else 0)  << 6
        # # cuda.syncthreads()
        # code |= (1 if A[ty+1][tx-1] > center else 0 )<< 5
        # # cuda.syncthreads()
        # code |= (1 if A[ty+1][tx] > center else 0 ) << 4
        # # cuda.syncthreads()
        # code |= (1 if A[ty+1][tx+1] > center else 0 ) << 3
        # # cuda.syncthreads()
        # code |= (1 if A[ty][tx+1] > center else 0 ) << 2
        # # cuda.syncthreads()
        # code |= (1 if A[ty-1][tx+1] > center else 0 )<< 1
        # # cuda.syncthreads()
        # code |= (1 if A[ty-1][tx-1] > center else 0) << 0
        # arry[x,y] = code
        # Since atomic add; adds value to the existing value
        # Need to figure out the fraction to be added in the previous value
        code = ( code - center)

        A[ty,tx] = code

        # cuda.atomic.add(A, (ty,tx),code)
        cuda.syncthreads()

        val  = A[ty,tx]
        cuda.atomic.add(arry, (x,y),val)
        cuda.syncthreads()
        # This Atomic Operation is equivalent to  hist[code % 256] += 1
        ind = code % BIN_COUNT

        cuda.atomic.add(hist, ind, 1)




        # val = H[ind]
        # cuda.syncthreads()
        # cuda.atomic.add(hist, ind, val)
    # TODO: May be Let each block creates its local histogram and
    #  call another kernel to add it into the global memory

    # else:
    #     arry[x,y] = 255
        # cuda.atomic.add(arry, (x,y),0)
    # Total No of Image
    # if ((y > 0 and  y < 65536 ) and (x > 0 and x < 65536)):
    #     # Now Lets Merge all the histograms
    #     # Do Linear Indexing
    #     # unique block index inside a 3D block grid
    #     LinearblockId = cuda.blockIdx.x + cuda.blockIdx.y * cuda.gridDim.x #//2D
    #     # global unique thread index, block dimension uses only x-coordinate
    #     LinearthreadId = LinearblockId * cuda.blockDim.x + cuda.threadIdx.x
    #     cuda.syncthreads()
    #     # print(LinearthreadId)
    #     val = LinearthreadId % BIN_COUNT
    #
    #     histVal = H[val]
    #
    #     cuda.atomic.add(hist,val, histVal)


@jit([void(int32[:,:], int32[:])], target='cuda')
def lbp_texture(arry, hist):

    # We have 10*10 threads per block
    A = cuda.shared.array(shape=(32,32), dtype=int32)

    # H = cuda.shared.array(BIN_COUNT, dtype=int32)

    x,y = cuda.grid(2)

    ty = cuda.threadIdx.x
    tx = cuda.threadIdx.y

    A[ty,tx] = arry[x,y]


    cuda.syncthreads()

    threadCountX = A.shape[0] - 1
    threadCountY = A.shape[1] - 1
    # If within x range and y range then calculate the LBP discriptor along
    # with histogram value to specific bin

    # Other wise Ignore the Value
    if (ty > 0 and  (threadCountX-ty) > 0 ) and (tx > 0 and (threadCountY-tx) > 0):
    #     # You can do the Processing here. ^_^
        code = 0
        #  We need to make sure that each value is accessable to each thread
        #  TODO: make them atomic
        center = A[ty, tx]

        # Compiler optimization: By loop unrolling
        # turns out twice faster than rolled version for over
        # 16*16 window
        code |= (1 if A[ty-1][tx-1] > center else 0 ) <<  7
        code |= (1 if A[ty][tx-1] > center else 0)  << 6
        code |= (1 if A[ty+1][tx-1] > center else 0 )<< 5
        code |= (1 if A[ty+1][tx] > center else 0 ) << 4
        code |= (1 if A[ty+1][tx+1] > center else 0 ) << 3
        code |= (1 if A[ty][tx+1] > center else 0 ) << 2
        code |= (1 if A[ty-1][tx+1] > center else 0 )<< 1
        code |= (1 if A[ty-1][tx-1] > center else 0) << 0

        # Since atomic add; adds value to the existing value
        # Need to figure out the fraction to be added in the previous value
        code = ( code - center)

        A[ty,tx] = code

        # cuda.atomic.add(A, (ty,tx),code)
        cuda.syncthreads()

        val  = A[ty,tx]
        cuda.atomic.add(arry, (x,y),val)
        cuda.syncthreads()
        # This Atomic Operation is equivalent to  hist[code % 256] += 1
        ind = code % BIN_COUNT
        cuda.atomic.add(hist, ind, 1)



        # val = H[ind]
        # cuda.syncthreads()
        # cuda.atomic.add(hist, ind, val)
    # TODO: May be Let each block creates its local histogram and
    #  call another kernel to add it into the global memory
    # or add it in the end.

    # else:
    #     arry[x,y] = 255
        # cuda.atomic.add(arry, (x,y),0)
    # Total No of Image
    # if ((y > 0 and  y < 65536 ) and (x > 0 and x < 65536)):
    #     # Now Lets Merge all the histograms
    #     # Do Linear Indexing
    #     # unique block index inside a 3D block grid
    #     LinearblockId = cuda.blockIdx.x + cuda.blockIdx.y * cuda.gridDim.x #//2D
    #     # global unique thread index, block dimension uses only x-coordinate
    #     LinearthreadId = LinearblockId * cuda.blockDim.x + cuda.threadIdx.x
    #     cuda.syncthreads()
    #     # print(LinearthreadId)
    #     val = LinearthreadId % BIN_COUNT
    #
    #     histVal = H[val]
    #
    #     cuda.atomic.add(hist,val, histVal)



def lbp_cuda_test():
    # src1 = np.arange(n*n, dtype=np.int32).reshape(n,n)
    gray = cv2.cvtColor(imread('test2.jpeg'),cv2.COLOR_BGR2GRAY)
    src1 = cv2.resize(gray, (n,n)).astype(np.int32)
    plt.imshow(src1, cmap=plt.cm.gray)
    plt.axis('off')
    plt.show()

    histogram = np.zeros(BIN_COUNT, dtype=np.int32)

    # We have threadCount*threadCount per block
    threadsperblock = (threadCount,threadCount)

    blockspergrid_x = math.ceil(src1.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(src1.shape[1] / threadsperblock[1])
    # We have 2*2 blocks
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    print(threadsperblock)
    print(blockspergrid)
    # Debugging Variables
    sd=0
    dd=10
    print(src1[sd:dd,sd:dd])

    cstream = cuda.stream()  # use stream to trigger async memory transfer
    ts = timer()
    # Increase Counter to measure the Efficiency
    count = 1
    for i in range(count):
        with cstream.auto_synchronize():
            # Copies Data to the device.
            d_src1 = cuda.to_device(src1, stream=cstream)
            # Copies histogram.
            d_hist_src = cuda.to_device(histogram, stream=cstream)
            # call the kernel fucntion
            lbp_texture[blockspergrid, threadsperblock, cstream](d_src1,d_hist_src)

            d_src1.copy_to_host(src1, stream=cstream)
            d_hist_src.copy_to_host(histogram, stream=cstream)

    te = timer()
    print('GPU Process ',count," Iterations : in ", te - ts)
    print(src1[sd:dd,sd:dd])
    print('histogram is')

    # histogram = histogram / sum(histogram)
    # Calculating histogram
    print(len(histogram))
    print(histogram)
    # plt.hist(histogram,256,[0,256]); plt.show()

    # Calculating From origional Array

    # histogram = histogram.astype(np.int32)
    hist = src1.astype(np.int32)
    # hist = np.bincount(histogram.ravel(),minlength=256)
    # hist = hist / sum(hist)

    # hist = histogram#.astype(np.int64)
    # x = itemfreq(hist.ravel())
    # print(x)

    # x = itemfreq(histogram.ravel())
    # x = np.bincount(hist.ravel(),minlength=255)

    # hist = x#/sum(x)#[:, 1]/sum(x[:, 1])

    # plt.hist(hist,256,[0,256]); plt.show()
    # print(len(hist))
    # print(hist)
    # Normalizing the hist
    # print(src1)

    plt.imshow(src1, cmap=plt.cm.gray)
    plt.axis('off')
    plt.show()
    # save_allFeatures(hist)

lbp_cuda_test()

print('done')
