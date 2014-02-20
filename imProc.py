# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 14:07:13 2013

@author: ondrej
"""
import sys
import numpy as np
from numpy.fft import *
import scipy.ndimage as nd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def xcorr(A,B,return_real=True):
    xc=fftshift(ifft2(fft2(A)*fft2(B).conjugate()))
    if return_real:
        return xc.real
    else:
        return xc
        
def shift(data, deltax, deltay, phase=0, return_real=None,verbose=True):
    """
    FFT-based sub-pixel image shift
    
    deltax: shift in vertical direction (first index in numpy array)
    deltay: shift in horizontal direction (second index in numpy array)
    phase: adds additional phase offset
    return_real: returns the real value of the shifted image (without residual phase due to shift)
    
    http://code.google.com/p/agpy/source/browse/trunk/AG_fft_tools/shift.py?r=536
    http://www.mathworks.com/matlabcentral/fileexchange/18401-efficient-subpixel-image-registration-by-cross-correlation/content/html/efficient_subpixel_registration.html

    Will turn NaNs into zeros
    """
    complex_data = isinstance(data.flatten()[0],(complex,np.complexfloating))
#    complex_data = False
    if return_real == None:
        if complex_data: return_real = False
        else: return_real = True
        
    if verbose:
        print "Shifting image by [%g,%g] pixels." %(deltax,deltay)    
    if np.any(np.isnan(data)):
        data = np.nan_to_num(data)
    if not complex_data:        
        nx,ny = data.shape
        Nx = np.fft.ifftshift(np.linspace(-np.fix(nx/2),np.ceil(nx/2)-1,nx))
        Ny = np.fft.ifftshift(np.linspace(-np.fix(ny/2),np.ceil(ny/2)-1,ny))
        Ny,Nx = np.meshgrid(Ny,Nx)
        gg = np.fft.ifft2( np.fft.fft2(data)* np.exp(1j*2*np.pi*(-deltax*Nx/nx-deltay*Ny/ny)))#* np.exp(-1j*phase))        
        if return_real:
            return np.real(gg)
        else:
            return gg
    else: 
        # This i to fix the weird artefacts in amplitude when shifting the complex image. 
        modulus = shift(abs(data), deltax, deltay, phase=0, return_real=True,verbose=False)
        phase = shift(np.angle(data), deltax, deltay, phase=0, return_real=True,verbose=False)    
        return modulus*np.exp(1j*phase)
        
def alignIm(A,B,align='abs', verbose = True):
    """
    Aligns the image B with respect to the image A.
    align:  'abs' - aligns aboslute values
            'phase' - aligns phase
            'both' - aligns complex number
    """
    if align == 'abs':
        xc=xcorr(abs(A),abs(B),return_real=True)
    elif align == 'phase':
        xc=xcorr(np.angle(A),np.angle(B),return_real=True)
    elif align == 'both':
        xc=xcorr(A,B,return_real=True)

    mx,my = np.unravel_index(np.argmax(xc),xc.shape)    
    YY,XX = np.meshgrid(np.arange(xc.shape[0]),np.arange(xc.shape[1])) # x is vertical (first) direction!
    rpix = 20
    mask = ((XX-mx)**2+(YY-my)**2) < rpix**2
    maxCoord=nd.measurements.center_of_mass(xc.real*mask)    
    if verbose: 
        print "Maximum in cross-correlation estimated @ ", maxCoord
        plt.figure(); plt.imshow(xc); plt.scatter(maxCoord[1],maxCoord[0],marker='x',c='k')
#    maxCoord=np.unravel_index(np.argmax(xc),xc.shape)
    shiftX = -(xc.shape[0]/2.-maxCoord[0])
    shiftY = -(xc.shape[1]/2.-maxCoord[1])
    if np.iscomplexobj(B):
        C=shift(B,shiftX,shiftY,return_abs=False, verbose = verbose)
    else:
        C=shift(B,shiftX,shiftY, verbose = verbose)
    return C, (shiftX, shiftY)

def padimg(A,shape,value=0):
    B = value*np.ones(shape).astype(A.dtype)
    s = (B.shape[0]/2-A.shape[0]/2, B.shape[1]/2-A.shape[1]/2)
    e = (s[0] + A.shape[0], s[1] + A.shape[1])
    B[s[0]:e[0],s[1]:e[1]] = A
    return B

def maxxc(delta,A,B):
    sys.stdout.write('.')
#    cplx = any(np.iscomplexobj(B))
    return -(xcorr(A,shift(B,delta[0],delta[1],verbose=0,return_abs=False))).real.max()
    
def alignImIterative(A,B,xtol=1e-10):
    if not A.shape == B.shape:
        B = padimg(B,A.shape)
    print "Initial shift from the position of the cross-correlation maximum:"
    _,delta0 = alignIm(A,B,align='both')
    res = minimize(maxxc, delta0, args=(A,B,), method='nelder-mead',options={'xtol': xtol, 'disp': True})
#    res = minimize(maxxc, res.x, args=(A,B,), method='powell',options={'xtol': 1e-18, 'disp': True})    
    print "Refined shift:"
#    cplx = any(np.iscomplexobj(B))
    C = shift(B,res.x[0],res.x[1], return_abs=False)    
    return C, res.x
    
def normalize(A,type='sum'):
    A = A.astype(np.float64)
    if 'stretch' in type.lower():
        A -= A.min()        
    if 'sum' in type.lower():
        B = A / A.sum()
    if 'max' in type.lower():
        B = A / A.max()
    if 'normal' in type.lower():
        print "Scaling to zero mean and unit variance."
        B = (A-A.mean())/np.sqrt(A.var())
    print "Output: sum = %g, max = %g, min = %g"%(B.sum(),B.max(), B.min())
    return B

def bary(d):
  # barycenter
  return nd.measurements.center_of_mass(d)

def bary2D(d):
    # barycenter of 2d sum porjection
    return nd.measurements.center_of_mass(d.sum(0))
    
def rebin(a, shape):
    # Bins array a by summing the pixels.
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).sum(-1).sum(1)

def tukeywin(window_length, alpha=0.5):
    '''The Tukey window, also known as the tapered cosine window, can be regarded as a cosine lobe of width \alpha * N / 2
    that is convolved with a rectangle window of width (1 - \alpha / 2). At \alpha = 1 it becomes rectangular, and
    at \alpha = 0 it becomes a Hann window.
 
    We use the same reference as MATLAB to provide the same results in case users compare a MATLAB output to this function
    output
 
    Reference
    ---------
 
http://www.mathworks.com/access/helpdesk/help/toolbox/signal/tukeywin.html
 
    '''
    # Special cases
    if alpha <= 0:
        return np.ones(window_length) #rectangular window
    elif alpha >= 1:
        return np.hanning(window_length)
 
    # Normal case
    x = np.linspace(0, 1, window_length)
    w = np.ones(x.shape)
 
    # first condition 0 <= x < alpha/2
    first_condition = x<alpha/2
    w[first_condition] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[first_condition] - alpha/2) ))
 
    # second condition already taken care of
 
    # third condition 1 - alpha / 2 <= x <= 1
    third_condition = x>=(1 - alpha/2)
    w[third_condition] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[third_condition] - 1 + alpha/2))) 
 
    return w
    
def tukeywin2D(window_size, alpha=0.5):
    w1 = tukeywin(window_size[0],alpha)
    w2 = tukeywin(window_size[0],alpha)    
    return np.outer(w1,w2)