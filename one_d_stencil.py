'''
Demonstrate the significant performance difference between transferring
regular host memory and pinned (pagelocked) host memory.
'''
from __future__ import print_function

from timeit import default_timer as timer

import numpy as np

from numba import vectorize,int32, float32, cuda, jit, void
import math
from scipy.stats import itemfreq

src = np.arange(40 ** 2, dtype=np.float32)
dst = np.empty_like(src)
src1 = np.empty_like(src)
src2 = np.empty_like(src)


# ------------ Correct Configuration for 256*256 window
# bpg = 64
# # Each block size wopuld be 32*32
# tpb = 32
# n = bpg * tpb * tpb

global radius
radius = 1

# ------------Correct Configuration for 3*6 window

bpg = 2
# Each block size wopuld be 3*3
global tpb
tpb = 30
n = (bpg * tpb ) + radius *4
BIN_COUNT = 10

@jit([void(float32[:])], target='cuda')
def gpu_1d_stencil(A):
    # Each block of size tbp
    # radius = 1
    # tpb = 30
    sA = cuda.shared.array(32, dtype=float32)
    # sA = cuda.shared.array(shape=(1, tpb + 2 *radius), dtype=float32)
    # sA = cuda.shared.array(tpb + 2*radius, dtype=float32)

    tx = cuda.threadIdx.x
    # ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    # by = cuda.blockIdx.y
    bw = cuda.blockDim.x
    # bh = cuda.blockDim.y

    x = tx + bx * bw
    cx = tx + radius
    #  Copying all elements from memory to device shared memory
    sA[cx] = A[x]
    if(tx == bx):
        # Copying Apron Values. i.e : Outside the bound values.
        sA[cx - radius] = A[x - radius]
        sA[cx + tpb] = A[x + tpb]
    cuda.syncthreads()

    acc = 0
    # rolling over the ROI
    for i in range(-radius,radius+1,1):
        acc += sA[cx + i]
        # Fancy averaging ?
        # acc = acc/3

    if(x > 0 and x < 64):
        # adding the difference to the global memory,
        # as there already are values at specific index.
        acc = acc - sA[cx]
        cuda.atomic.add(A, x, acc)

@jit([void(float32[:],int32[:])], target='cuda')
def gpu_histogram(A, Hist):
    # Each block of size tbp
    # radius = 1 * 2  // At both ends.
    # tpb = 30

    # size = tpb + radius * 2  == 32  for each block
    totalsize = bpg* (tpb+radius*2) # for all blocks
    sA = cuda.shared.array(32, dtype=float32)

    tx = cuda.threadIdx.x
    # ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    # by = cuda.blockIdx.y
    bw = cuda.blockDim.x
    # bh = cuda.blockDim.y

    x = tx + bx * bw
    cx = tx + radius

    #  Copying all elements from memory to device shared memory
    sA[cx] = A[x]
    if(tx == bx):
        # Copying Apron Values. i.e : Outside the bound values.
        sA[cx - radius] = A[x - radius]
        sA[cx + tpb] = A[x + tpb]

    cuda.syncthreads()

    if(x > 0 and x < totalsize):
        # Calculating the bin index to save
        binIndex = sA[cx] % BIN_COUNT
        cuda.atomic.add(Hist, binIndex ,1)



def cudatest_stencil():
    src1 = np.arange(n, dtype=np.float32)

    print(src1)
    stream = cuda.stream()  # use stream to trigger async memory transfer
    ts = timer()
    # Controll the iterations
    count = 1
    for i in range(count):
        with stream.auto_synchronize():
            # ts = timer()
            d_src1 = cuda.to_device(src1, stream=stream)
            gpu_1d_stencil[bpg, tpb, stream](d_src1)
            d_src1.copy_to_host(src1, stream=stream)

    te = timer()
    print('pinned ',count," : ", te - ts)
    # mid = math.ceil(n/2)
    # print(src1[mid:n])
    print(src1[0:n])
    print(len(src1[0:n]))


def cudatest_hist():
    # src1 = np.arange(n, dtype=np.float32)
    src1 = np.random.randint(BIN_COUNT,size=n).astype(np.float32)
    histogram = np.zeros(BIN_COUNT, dtype=np.int32)

    print(src1)
    stream = cuda.stream()  # use stream to trigger async memory transfer
    ts = timer()

    # Controll the iterations
    count = 1
    for i in range(count):
        with stream.auto_synchronize():
            # ts = timer()
            d_src1 = cuda.to_device(src1, stream=stream)
            d_hist = cuda.to_device(histogram, stream=stream)
            # gpu_1d_stencil[bpg, tpb, stream](d_src1)
            gpu_histogram[bpg, tpb, stream](d_src1,d_hist)
            d_src1.copy_to_host(src1, stream=stream)
            d_hist.copy_to_host(histogram, stream=stream)

    te = timer()
    print('pinned ',count," : ", te - ts)
    print(histogram)
    # Taking histogram on origional data.
    # This histogram will contain few more frequency due to the padding we add in the orional data.
    # in kernel code.
    hist = src1.astype(np.int64)
    x = itemfreq(hist.ravel())
    hist = x#[:, 1]/sum(x[:, 1])
    print(hist)


# cudatest_stencil()
cudatest_hist()

print('done')
# assert np.allclose(dst, src)
