# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 13:23:15 2022

@author: D
"""

import os
import cv2
import numpy as np
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import skimage
from skimage import io, img_as_float, color,morphology, img_as_ubyte,segmentation, measure
from scipy import signal, ndimage
import math
import inspect
matplotlib.use('TkAgg')
#%%
# Perbaikan pada fungsi visual_callback_2d
def visual_callback_2d(background, fig=None):
    if fig is None:
        fig = plt.figure(figsize=(8,8))
    fig.clf()
    ax1 = fig.add_subplot(1, 2, 1) 
    ax1.imshow(background, cmap=plt.cm.gray)

    ax2 = fig.add_subplot(1, 2, 2)
    ax_u = ax2.imshow(np.zeros_like(background), vmin=0, vmax=1, cmap=plt.cm.gray)

    def callback(levelset):
        # Membersihkan subplot (axes) tanpa menghapus figure
        ax1.clear()
        
        # Gambar ulang latar belakang
        ax1.imshow(background, cmap=plt.cm.gray)
        
        # Gambar kontur baru
        ax1.contour(levelset, [0.15], colors='c', linewidths=2)
        ax_u.set_data(levelset) # bentuk tampilan biner

        # Tampilkan plot secara interaktif
        fig.canvas.draw()
        plt.show(block=False)
        plt.pause(0.01)

    return callback
#%% Function
def img2graydouble(image):
    _,_,c = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image.astype(float)
    return image

def estructurant(radius):
    kernel = np.zeros((2*radius+1, 2*radius+1) ,np.uint8)
    y,x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    kernel[mask] = 1
    kernel[0,radius-1:kernel.shape[1]-radius+1] = 1
    kernel[kernel.shape[0]-1,radius-1:kernel.shape[1]-radius+1]= 1
    kernel[radius-1:kernel.shape[0]-radius+1,0] = 1
    kernel[radius-1:kernel.shape[0]-radius+1,kernel.shape[1]-1] = 1
    return kernel

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def initLS(image):
    height,width = image.shape
    height = float(height)
    width = float(width)
    yy , xx = np.mgrid[0:height,0:width]
    x = float(math.floor(width/2))
    y = float(math.floor(height/2))
    r = float(math.floor(min(.2*width, .2*height)))
    phi0 = (np.sqrt(((xx - x)**2 + (yy - y)**2 ))-r)
    phi0 = np.sign(phi0)*2
    return phi0

def Neumann(phi):
    nrow , ncol = phi.shape
    nrow -=1
    ncol -=1
    g = phi
    (g[0,0],g[0,ncol],g[nrow,0],g[nrow,ncol]) = (g[2,2],g[2,ncol-3],g[nrow-3,2],g[nrow-3,ncol-3])
    (g[0,1:-1],g[nrow,1:-1]) = (g[2,1:-1],g[nrow-3,1:-1])
    (g[1:-1,1],g[1:-1,ncol]) = (g[1:-1,2],g[1:-1,ncol-3])
    
    return g

def Curvature(phi):
    ny , nx = np.gradient(phi)
    absR = np.sqrt((nx**2)+(ny**2))
    absR = absR + (absR==0)*np.finfo(float).eps
    _,nxx1 = np.gradient(nx/absR)
    nyy1,_ = np.gradient(ny/absR)
    Kappa = nxx1 + nyy1
    return (Kappa , absR)

def Heaviside(phi):
    H = (1/np.pi)*np.arctan(phi)+0.5
    return H


def FittingAverage(img, phi):
    Hphi = Heaviside(phi)
    cHphi = 1-Hphi
    ca = np.sum(np.multiply(img,Hphi))
    cb = np.sum(np.multiply(img,cHphi))
    c1 = ca/(np.sum(Hphi))
    c2 = cb/(np.sum(cHphi))
    return (c1,c2)
    global local_vars
    local_vars = inspect.currentframe().f_locals
    
def Dirac(phi):
    a = 1 + phi**2
    D = (1/np.pi)
    Dir = D/a
    return Dir

def Convergence(phi,iteration=0,absR=0,teta=0.1,maxs=50, preArea= 0, preLength = 0):
    phip = np.where(phi<0, 1, 0)
    Area = np.sum(phip)
    ErrorArea = abs(Area-preArea)
    dPhi = Dirac(phi)
    Length = np.sum(absR*dPhi)
    ErrorLength = abs(Length-preLength)
    if (ErrorArea <= teta) and (ErrorLength <= teta) or (iteration==maxs):
        Converge = True
    else:
        Converge = False
    return (Converge,Area,Length,ErrorArea,ErrorLength)

def ObDetection(phi):
    g = np.where(phi<=0, 1, 0)
    se = estructurant(3)
    g1 = img_as_ubyte(g)
    opening = cv2.morphologyEx(g1, cv2.MORPH_OPEN, se)
    clearObj = segmentation.clear_border(opening)
    fillObj = ndimage.binary_fill_holes(clearObj)
    labelObj = measure.label(fillObj)
    propObj = measure.regionprops(labelObj)
    BW = propObj
    maskBW = np.zeros(image.shape, dtype=np.uint8)
    maskBW = skimage.util.img_as_int(maskBW)
    area = []
    for region in (propObj):
        area.append(region.minor_axis_length)

    for region in (propObj):
        if (region.minor_axis_length/max(area)) >= 0.7:
            # print(region.minor_axis_length)
            minr, minc, maxr, maxc = region.bbox
            cv2.rectangle(maskBW, (minc , minr ), (maxc , maxr ), (255, 255, 255), -1)
            cv2.rectangle(imgResult, (minc-20, minr-20), (maxc+20, maxr+20), (0, 255, 0), 2)
            cv2.putText(imgResult, "Area: " + str(int(region.area)), (minc, minr-30), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                                          (0, 255, 0), 1)
    
    g = fillObj
    phi = -2*maskBW+1
    init = 0
    return (phi, g, init)

#%% Global Parameter
local_vars = {}
maxs = 150
dt = 10
teta = 1
seMorf = morphology.disk(2)
print(seMorf)
kernel = matlab_style_gauss2D((5,5),1)
#%% Input
image = cv2.imread(r'5.bmp')
GT_image = io.imread((r'5.bmp'),as_gray=True)
imgResult = image.copy()
#%% RGB2GRAY
image = img2graydouble(image)
image = img_as_float(image)
image *= 255
#%% Init LS
phi = initLS(image)
#%% Initial Allocation
height,width = image.shape
g = np.zeros((height,width))
error = np.zeros((2,1))
Beta = np.zeros((1,1))
preLength = 0
preArea = 0
beta = 0
i = 1
callback = visual_callback_2d(image)
#untuk test
# phi = Neumann(phi)
# div,absR = Curvature(phi)
# c1,c2 = FittingAverage(image, phi)
# AACMR = div*absR + (1-abs(beta)) * (image - (c1+c2)/2) + beta*g*absR
# phi = phi + dt*AACMR
# phi = np.sign(phi)
# phi = signal.convolve2d(phi,kernel, mode='same')
    

# #%% Morph Regularization
# phi = np.where(phi > 0, 1, 0)
# cv2.imshow("Result", phi)
# cv2.imshow("kappa", div)
# cv2.imshow("absR", absR)
# print("neumann", phi[100,100])
# print("kappa", div[100,100])
# print("absR", absR[100,100])
# print("c1 c2", c1, c2, type(c1), type(c2))
# cv2.waitKey(0)

while i>=0:
    # Beta[i-1] = beta
    phi = Neumann(phi)
    print("iterasi", i)
    div,absR = Curvature(phi)
    c1,c2 = FittingAverage(image, phi)
    
    # #%% AACMR
    AACMR = div*absR + (1-abs(beta)) * (image - (c1+c2)/2) + beta*g*absR
    phi = phi + dt*AACMR
    
    #%% Binary Gaussian
    phi = np.sign(phi)
    phi = signal.convolve2d(phi,kernel, mode='same')
    

    #%% Morph Regularization
    phi = np.where(phi > 0, 1, 0)
    phi2 = img_as_ubyte(phi)
    
    phi = morphology.dilation((morphology.erosion(phi,seMorf)),seMorf)
    phi2 = cv2.dilate((cv2.erode(phi2,seMorf)),seMorf)
    phi = morphology.erosion((morphology.dilation(phi,seMorf)),seMorf)
    phi2 = cv2.erode(cv2.dilate(phi2,seMorf),seMorf)
    phi=phi2
    phi = (np.where(phi > 0, 1, -1)).astype(float)
    
    if beta==0:
        Converge,preArea,preLength,ErrorArea,ErrorLength=Convergence(phi, iteration=i, absR=absR, teta=teta, preArea=preArea, preLength=preLength)
        if Converge:
            print("Terdeteksi")
            phiShrink,gShrink,initialShrink = ObDetection(phi)
            g = 1-gShrink
            phi = phiShrink
            beta = 1
    else:
        callback(phi)
        phiShrink = phi
        Converge,preArea,preLength,ErrorArea,ErrorLength=Convergence(phi, iteration=i, absR=absR, teta=teta, preArea=preArea, preLength=preLength)
        if Converge:
            break
        g = 1-gShrink
        phi = phiShrink
    cv2.imshow("Ressult", phi)
    i += 1     
    # kernel = matlab_style_gauss2D((5,5),3)
    # phi = signal.convolve2d(phi,kernel, mode='same')
    
    # #%%Convergences
    # Converge,preArea,preLength,ErrorArea,ErrorLength=Convergence(phi, iteration=i, absR=absR, teta=teta, preArea=preArea, preLength=preLength)
    
    # if Converge:
    #     g = np.where(phi<=0, 1, 0)
    #     break
    # error[0,i] = ErrorArea
    # error[1,i] = ErrorLength
    

# cv2.imshow("Result", phi)
cv2.waitKey(0)
# plt.imshow(phiShrink,cmap='gray')
# plt.show()