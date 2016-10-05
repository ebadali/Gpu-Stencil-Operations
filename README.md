# Gpu-Stencil-Operations

This Repo Demonstrates the Methods For running Famous Image Processing Serial (CPU) operations on CUDA Powered GPU


* Stencil : It is an Important Communication pattern Among Threads within a Block of a Grid, Basically it Allows to Reads Input From Fixed Neighborhood in a single location of an Array.

This Repo Contains Stencils Operations and Demonstrates:

##### 1d Stensil Operations On Cuda in : one_d_stencil.py

Consider applying a 1D stencil to a 1D array of elements.
Each output element is the sum of input elements within a radius.
If radius is n, then each output element is the sum of n input elements:

![alt tag](https://github.com/ebadali/Gpu-Stencil-Operations/blob/master/id_conv.jpg?raw=true)



##### 2d Stensil Operations On Cuda in : image_filter_test.py 

Similarly We Create an Apron around Image Tile, So that,

* Image tile can be cached to shared memory
* Each output pixel must have access to neighboring pixels within certain radius R
* This means tiles in shared memory must be expanded with an apron that contains neighboring pixels
* Only pixels within the apron write results.The remaining threads do nothing

![alt tag](https://github.com/ebadali/Gpu-Stencil-Operations/blob/master/tileApron.png?raw=true)


##### Basic Image Processing Filters, using 2d Stensils in : image_filter_test.py

Apply some Basic operation such as Thresholding, Image Sharpening and averaging.


### Enviroment:

0. Ubunto 16.04 LTS
1. GeForce GTX 750 Ti Compute Capability 5.0 with Cuda toolkits and
2. Anaconda python 3 
3. Numba : compiler for Python 
4. Cuda Python 

###### Resources:

1. [Numba](http://numba.pydata.org/)  
2. Manuel Ujaldon [Nvidia Cuda Fellow](http://supercomputing.swin.edu.au/files/download/CUDA-basics-and-examples.pdf)

