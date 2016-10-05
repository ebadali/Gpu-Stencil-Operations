# Gpu-Stencil-Operations

* Enviroment: 
0 Ubunto 16.04 LTS
1 GeForce GTX 750 Ti/PCIe/SSE2
2 python 3
3 Numba : compiler for Python 
4 Cuda Python


This Repo Demonstrates the Methods For running Famous Serial (CPU) operations on CUDA Powered GPU


* Stencil : It is an Important Communication pattern Among Threads within a Block of a Grid, Basically it Reads Input From   Fixed Neighborhood in a single location.

This Repo Contains Stencils Operations and Demonstrates:

* 1d Stensil Operations On Cuda in : one_d_stencil.py

Consider applying a 1D stencil to a 1D array of elements.
Each output element is the sum of input elements within a radius.
If radius is 3, then each output element is the sum of 7 
input elements:
![alt tag](https://www.evl.uic.edu/sjames/cs525/images/diagram_05.jpg)



* 2d Stensil Operations On Cuda in : image_filter_test.py 

Filter coefficients can be stored in constant memory
•Image tile can be cached to shared memory
•Each output pixel must have access to neighboring pixels within certain radius R
•This means tiles in shared memory must be expanded with an apron that contains neighboring pixels
•Only pixels within the apron write results.The remaining threads do nothing




* Basic Image Processing Filters, using 2d Stensils in : image_filter_test.py
