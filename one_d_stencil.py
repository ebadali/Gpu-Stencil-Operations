'''
Demonstrate the significant performance difference between transferring
regular host memory and pinned (pagelocked) host memory.
'''
from __future__ import print_function

from timeit import default_timer as timer

import numpy as np

from numba import vectorize,int32, float32, cuda, jit, void
import math

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


# @jit([void(float32[:,:])], target='cuda')
@jit([void(float32[:])], target='cuda')
def gpu_1d_stencil(A):

    # Each block of size tbp
    # radius = 1
    # tpb = 30
    sA = cuda.shared.array(32, dtype=float32)
    # sA = cuda.shared.array(shape=(1, tpb + 2 *radius), dtype=float32)
    # sA = cuda.shared.array(tpb + 2*radius, dtype=float32)
    #
    #
    tx = cuda.threadIdx.x
    # ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    # by = cuda.blockIdx.y
    bw = cuda.blockDim.x
    # bh = cuda.blockDim.y
    #
    x = tx + bx * bw
    cx = tx + radius
    #  Copying all elements from memory to device shared memory
    sA[cx] = A[x]
    if(tx > bx and tx < radius):
        sA[cx - radius] = A[x - radius]
        sA[cx + tpb] = A[x + tpb]
    A[x] = 0
    cuda.syncthreads()

    acc = 0
    # rolling over the ROI
    for i in range(-radius,radius+1,1):
        acc += sA[cx + i]
        # acc = acc/3

    if(x > 0 and x < 64):
        acc = acc - A[x]
        cuda.atomic.add(A, x,acc)
        # A[x] = -acc

def cudasquare():
    print(n)
    src1 = np.arange(n, dtype=np.float32)
    print(src1)
    stream = cuda.stream()  # use stream to trigger async memory transfer
    ts = timer()
    count = 1
    for i in range(count):
        with stream.auto_synchronize():
            # ts = timer()
            d_src1 = cuda.to_device(src1, stream=stream)
            d_dst = cuda.device_array_like(src1, stream=stream)
            gpu_1d_stencil[bpg, tpb, stream](d_src1)
            d_src1.copy_to_host(src1, stream=stream)
    te = timer()
    print('pinned ',count," : ", te - ts)
    mid = math.ceil(n/2)
    print(src1[0:n])
    # print(src1[mid:n])
    print(len(src1[0:n]))

cudasquare()

print('done')
# assert np.allclose(dst, src)
