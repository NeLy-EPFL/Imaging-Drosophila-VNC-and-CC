#!/usr/bin/env python3
"""
Regression module
=================

This module provides functions to find the pixel-wise linear regression weights for annotated behavior videos.

The method is describe in:
Chen, Hermans, et al.
"Imaging neural activity in the ventral nerve cord of behaving adult Drosophila"
Nature Communications 2018

Dependencies
------------
python3     (tested with 3.6.5)
numpy       (tested with 1.14.3 and 1.14.4)
matplotlib  (tested with 2.2.2)
sklearn     (tested with 0.19.1)
skimage     (tested with 0.14.0)

Copyright (C) 2018 F. Aymanns, florian.aymanns@epfl.ch

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from skimage import io
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
import math
import sys
import os.path

def HeatIm(image, out_path=None, colormap='jet'):
    """
    This function turn a gray scale image into a color map.

    Parameters
    ----------
    image : 2D numpy array
        Gray scale image that is converted.
    out_path : string, optional
        Path used to save output.
        If not specified, no output is not saved.
    colormap : string, default='jet'
        Name of the matplotlib color map used.

    Returns
    -------
    3D numpy array
        RGB image in form of a numpy array.
        Third dimension encodes RGB.
    """
    cmap = plt.get_cmap(colormap)
    color_img = cmap(image,bytes=True)
    color_img = np.delete(color_img, 3, -1)
    
    if out_path is not None:
        extension = os.path.splitext(out_path)[1]
        if extension == '.eps':
            fig = plt.figure(facecolor='black')
            ax = plt.subplot(1, 1, 1)
            ax.spines['bottom'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            MinPixVal = min(image.flatten())
            MaxPixVal = max(image.flatten())
            pim = plt.imshow(image, cmap=colormap, interpolation='nearest', vmin=MinPixVal, vmax=MaxPixVal)
            cbar = plt.colorbar(fraction=0.035, pad=0.03)
            cbar.ax.tick_params(axis = 'y', which ='major', width=0, colors='white', labelsize=20)
            plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor='none', transparent=True, bbox_inches='tight', pad_inches=0.05) 
            plt.close(fig)
        else:
            if not out_path is None:
                io.imsave(out_path, color_img.astype(np.uint8))
    return color_img

def crop_stack(stack, x_min, x_max, y_min, y_max, *args):
    """
    Function to crop 2D and 3D stack.

    This function crops 2D and 3D stack in the form of numpy arrays.
    It includes the specified *_min and *_max values.
    The first dimension is always assumed to be time.

    Parameters
    ----------
    stack : np.array
        Image to be cropped.
    x_min : int
        Smallest x pixel index included.
    x_max : int
        Biggest x pixel index included.
    y_min : int
        Smallest y pixel index included.
    y_max : int
        Biggest x pixel index included.
    z_min : int, optional
        Smallest z pixel index included.
    z_max : int, optional
        Biggest z pixel index included.

    Returns
    _______
    cropped : np.array
        Cropped stack

    Raises
    ______
    ValueError
        Raised if input stack does not have three or four dimension.

    Examples
    --------
    >>> from skimage import io
    >>> import nely_suite as ns
    >>> GCamP_img = io.imread('GC6s.tif')
    >>> x_min, x_max, y_min, y_max = ns.find_registered_region(GCamP_img);
    >>> GCamP_img = ns.crop_stack(GCamP_img, x_min, x_max, y_min, y_max)
    """
    if stack.ndim!=3 and stack.ndim!=4:
        raise ValueError('Can only corp stacks with three or four dimension. Input has {} dimension.'.format(stack.ndim))

    if stack.ndim == 3:
        cropped = stack[:, x_min:x_max+1, y_min:y_max+1]
    elif stack.ndim == 4:
        cropped = stack[:, x_min:x_max+1, y_min:y_max+1, args[0]:args[1]]
    return cropped

def find_registered_region(img):
    """
    Find region without border artefact from registration.

    Warping images at the end of the registration might leave some areas at the edges that are not defined.
    These values are generally assigned NaN values. This function finds the largest centered rectangle that
    does not include pixels with NaN values.

    Parameters
    ----------
    img : np.array
        Image in which the region should be found. Can be 2D or 3D.
        First dimension is considered time if array has three dimensions.

    Returns
    -------
    x_min : int
        Column index of left rectangle side. 
    x_max : int
        Column index of right rectangle side. 
    y_min : int
        Row index of upper rectangle side. 
    y_max : int
        Row index of lower rectangle side.

    Raises
    ------
    ValueError
        Raised if input array has more than three dimension.

    Examples
    --------
    >>> from skimage import io
    >>> import nely_suite as ns
    >>> GCamP_img = io.imread('GC6s.tif')
    >>> x_min, x_max, y_min, y_max = ns.find_registered_region(GCamP_img);
    >>> GCamP_img = ns.crop_stack(GCamP_img, x_min, x_max, y_min, y_max)
    """
    if img.ndim > 3:
        raise ValueError('Input array has too many dimension! Maximum is 3!')
    bools = np.isnan(img)
    if bools.ndim == 3:
        bools = np.any(bools,0)
    updated = True
    center_x = math.floor(bools.shape[0]/2)
    center_y = math.floor(bools.shape[1]/2)
    assert not bools[center_x,center_y], 'Center is not registered!'
    x_min = center_x
    x_max = center_x
    y_min = center_y
    y_max = center_y

    #Note: +1 for all the max is necessary because : operator excludes last element
    while updated:
        updated=False
        if not np.any(bools[x_min-1,y_min:y_max+1]):
            x_min = x_min-1
            updated = True
        if not np.any(bools[x_max+1,y_min:y_max+1]):
            x_max = x_max+1
            updated = True
        if not np.any(bools[x_min:x_max+1,y_min-1]):
            y_min = y_min-1
            updated = True
        if not np.any(bools[x_min:x_max+1,y_max+1]):
            y_max = y_max+1
            updated = True
    return x_min, x_max, y_min, y_max

def avg_frames(stack, indices=None):
    """
    This function calculates the average of a list of frames.

    First dimension of stack is assumed to be time.

    Parameters
    ----------
    stack : np.array
        Stack containing frames for averaging.
    indices : list of integers, optional
        Indices of frames used for average.
        If not give, the whole stack is used.

    Returns
    -------
    avg_img : np.array
        Average image.

    Examples
    --------
    >>> from skimage import io
    >>> import numpy as np
    >>> import nely_suite as ns
    >>> GCamP_img = io.imread('GC6s.tif')
    >>> avg_img = ns.avg_frames(GCamP_img,np.range(10))
    """
    if indices is None:
        sumImg = np.sum(stack,1)
    else:
        sumImg = sum(stack[indices]) 
    avgImg=sumImg/len(indices)
    return  avgImg 

def dff(img, baseline):
    """
    This function calculates the change in fluorescence change in percent.

    First dimension of stack is assumed to be time.

    Parameters
    ----------
    img : np.array
        Single image or stack of images.
    baseline : np.array
        Baseline used. If 1D, it isused as list of indices (c.f. avg_frames).
        Otherwise, must have the same dimension as the image(s) given in img.

    Returns
    -------
    dff_img : np.array
        dff Image.

    Examples
    --------
    >>> from skimage import io
    >>> import numpy as np
    >>> import nely_suite as ns
    >>> GCamP_img = io.imread('GC6s.tif')
    >>> avg_img = ns.avg_frames(GCamP_img, np.range(10))
    >>> dff_img = ns.dff(GCamP_img, avg_img)

    >>> from skimage import io
    >>> import numpy as np
    >>> import nely_suite as ns
    >>> GCamP_img = io.imread('GC6s.tif')
    >>> dff_img = ns.dff(GCamP_img, np.arange(10))
    """
    if baseline.ndim == 1:
        baseline = avg_frames(img, baseline)
    dff_img = (img - baseline) / baseline *100
    return dff_img

def build_regressor(behavior_sequence,times):
    """
    This functions constructs a regressor based on a behaviour sequence.
    It uses the half life time reported in Chen 2018 for GCamp6s.

    Parameters
    ----------
    behavior_sequence : 1D numpy array
        Sequence with ones and zeros that indicates when a certain behaviour is observed
    times : 1D numpy array
        Array with corresponding times for behavior sequence (in seconds).

    Returns
    -------
    regressor : 1D numpy array of same size as behavior_sequence
        Predicted Ca response.
    """
    t = times[:100]
    tau = 1.1448/math.log(2)
    ca_func = np.exp(-t/tau)
    regressor = np.convolve(ca_func,behavior_sequence,'same')
    regressor = np.expand_dims(regressor,axis=1)
    return regressor

def linear_regression(img,regressor):
    """
    This function calculated the pixel-wise linear-regression weights and R^2 for a given image stack and regressor.

    Parameters
    ----------
    img : 3D numpy array
        Stack of images. First dimension encodes time.
    regressor : 1D numpy array
        Regressor of same length as first dimension of img.

    Returns
    -------
    weights : 2D numpy array
        Weights for each pixel.
    R2 : 2D numpy array
        R^2 for each pixel.
    """
    img[np.where(np.isnan(img))]=0

    weights = np.zeros((img.shape[1],img.shape[2]))
    R2 = np.zeros((img.shape[1],img.shape[2]))
    
    mean_values = np.mean(img,axis=0)
    variance = np.var(img,axis=0)
    
    reg = linear_model.LinearRegression()
    
    for i in range(img.shape[1]):
        print('{} out of {}'.format(i,img.shape[1]))
        for j in range(img.shape[2]):
            reg.fit(regressor,img[:,i,j])
            weights[i,j]  = reg.coef_[0]
            R2[i,j]  = np.sum(reg.predict(regressor)-mean_values[i,j])/variance[i,j]
    return weights, R2

if __name__=='__main__':
    test_bool = (input('Should the test data be used? (Y/n): ') or 'y')
    if test_bool == 'Y' or test_bool == 'y':
        GCamP_path = 'test_data/GC6s.tif'
        start_frame_baseline = 0 
        stop_frame_baseline =  10 
        behavior_idx_path = 'test_data/behavior_idx.txt'
        fluorescence_idx_path = 'test_data/fluorescence_idx.txt'
        times_path = 'test_data/times.txt'
        walking_seq_path = 'test_data/walking_seq.txt'
        grooming_seq_path = 'test_data/grooming_seq.txt'
    elif test_bool == 'N' or test_bool == 'n':
        GCamP_path =            os.path.expanduser(os.path.expandvars(input('Path to GCamP image stack (e.g. ./test_data/GC6s.tif):')))
        start_frame_baseline =  int(input('Start frame for baseline (e.g. 0):'))
        stop_frame_baseline =   int(input('Stop frame for baseline (e.g. 10):'))
        behavior_idx_path =     os.path.expanduser(os.path.expandvars(input('Path to behavior_idx file (e.g. ./test_data/behavior_idx.txt):')))
        fluorescence_idx_path = os.path.expanduser(os.path.expandvars(input('Path to fluorescence_idx file (e.g. ./test_data/fluorescence_idx.txt):')))
        times_path =            os.path.expanduser(os.path.expandvars(input('Path to times file (e.g. ./test_data/times.txt):')))
        walking_seq_path =      os.path.expanduser(os.path.expandvars(input('Path to walking_seq file (e.g. ./test_data/walking_seq.txt):')))
        grooming_seq_path =     os.path.expanduser(os.path.expandvars(input('Path to grooming_seq file (e.g. ./test_data/grooming_seq.txt):')))
    else:
        print("You answered: "+test_bool+" but answer can only be 'y' or 'n'.")
        raise Exception

    img = io.imread(GCamP_path)
    x_min,x_max,y_min,y_max = find_registered_region(img)
    img = crop_stack(img,x_min,x_max,y_min,y_max)
    baseline_img = avg_frames(img,range(start_frame_baseline,stop_frame_baseline))
    dFF = dff(img,baseline_img)
    
    behavior_idx = np.genfromtxt(behavior_idx_path).astype(np.int)
    fluorescence_idx = np.genfromtxt(fluorescence_idx_path).astype(np.int)
    times = np.genfromtxt(times_path)
    walking_seq = np.genfromtxt(walking_seq_path)[behavior_idx]
    grooming_seq = np.genfromtxt(grooming_seq_path)[behavior_idx]
    dFF = dFF[fluorescence_idx]

    walking_regressor = build_regressor(walking_seq,times)
    grooming_regressor = build_regressor(grooming_seq,times)
   
    walking_weights, walking_R2 = linear_regression(dFF,walking_regressor)
    grooming_weights, grooming_R2 = linear_regression(dFF,grooming_regressor)

    walking_weights = walking_weights/max(walking_weights.flatten())
    grooming_weights = grooming_weights/max(grooming_weights.flatten())

    HeatIm(walking_weights,  'walking_weights_heat.eps')
    HeatIm(grooming_weights, 'grooming_weights_heat.eps')
    
    pathname = os.path.dirname(sys.argv[0])
    pathname = os.path.abspath(pathname)
    print('Saved heat maps in '+pathname)
