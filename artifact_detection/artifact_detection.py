#!/usr/bin/env python3
"""
Divergence/Artifact module
==========================

This module provides functions to compute the divergence of vector fields generated with our registration algorithm.
Furthermore it calculates the average number of artifacts per frame based on the divergence.

The method is describe in:
Chen, Hermans, et al.
"Imaging neural activity in the ventral nerve cord of behaving adult Drosophila"
Nature Communications 2018

Dependencies
------------
python3     (tested with 3.6.5)
numpy       (tested with 1.14.3 and 1.14.4)
skimage     (tested with 0.14.0)
matplotlib  (tested with 2.2.2)
OpenCV      (tested with 3.4.1)

Copyright (C) 2018 F. Aymanns, florian.aymanns@epfl.ch

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
import numpy as np
from skimage import io
import cv2
import os
import os.path

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

def compute_divergence(path,out=None,roi=None):
    """
    This functions computes the divergence for a set of vector fields.

    Parameters
    ----------
    path : string
        Path to folder with input data. Should contain: tdTom.tif and
        vector fields in files wx_frameXXX.txt, wy_frameXXX.txt.
        All input files can be generated using the registration algorithm
        provided in this repository.
    out : string, optional
        Name of output directory. If not specified, no output is save.
    roi : list of 4 integers
        Can be used to specify a ROI. Format: [x_min,x_max,y_min,y_max]
    """
    path = os.path.expanduser(os.path.expandvars(path))
    images = io.imread(path+'/tdTom.tif')
    n_frames = images.shape[0]
    if roi is None:
        roi_y_min=0
        roi_y_max=images.shape[1]
        roi_x_min=0
        roi_x_max=images.shape[2]
    else:
        roi_y_min=roi[2]
        roi_y_max=roi[3]
        roi_x_min=roi[0]
        roi_x_max=roi[1]
        images = crop_stack(images,roi_y_min,roi_y_max-1,roi_x_min,roi_x_max-1)
    frobenius_output = np.zeros((n_frames,images.shape[1],images.shape[2]))
    divergence_output = np.zeros((n_frames,images.shape[1],images.shape[2]))
    vectors = np.zeros((n_frames,320,320,3))
    x = np.linspace(1,roi_x_max-roi_x_min,roi_x_max-roi_x_min)
    y = np.linspace(1,roi_y_max-roi_y_min,roi_y_max-roi_y_min)
    for i in range(n_frames-1):
        print('Frame {}'.format(i+1))
        wx = np.genfromtxt(path+'/wx_frame{}.dat'.format(i+1),dtype=np.float,delimiter=',')[roi_y_min:roi_y_max,roi_x_min:roi_x_max]
        wy = np.genfromtxt(path+'/wy_frame{}.dat'.format(i+1),dtype=np.float,delimiter=',')[roi_y_min:roi_y_max,roi_x_min:roi_x_max]
        grad_wx = np.gradient(wx)
        grad_wy = np.gradient(wy)
        divergence = grad_wx[1]+grad_wy[0]
        frobenius = np.sqrt(np.square(grad_wx[0])+np.square(grad_wx[1])+np.square(grad_wy[0])+np.square(grad_wy[1]))
        frobenius_output[i+1,:,:]=frobenius
        divergence_output[i+1,:,:]=divergence
        fig = Figure(figsize=(4, 4), dpi=80)
        canvas = FigureCanvas(fig)
        ax = fig.gca()
        ax.streamplot(x,y,wx,wy)
        ax.axis('off')
        canvas.draw()       # draw the canvas, cache the renderer
        vectors[i,:,:,:] = np.reshape(np.fromstring(canvas.tostring_rgb(), dtype='uint8'),(320,320,3))
    if out is not None:
        out = os.path.expanduser(os.path.expandvars(out))
        os.makedirs(out,exist_ok=True)
        io.imsave(out+'/roi.tif',images.astype(np.float32))
        io.imsave(out+'/frobenius.tif',frobenius_output.astype(np.float32))
        io.imsave(out+'/vectors.tif',vectors.astype(np.uint8))
        io.imsave(out+'/divergence.tif',divergence_output.astype(np.float32))
    return divergence_output

if __name__=='__main__':
    path = input('Directory with input files (e.g. ./test_data):')
    out = (input('Output directory (can be empty or e.g. output):') or None)
    div = compute_divergence(path,out)
    num_of_artifacts = 0  
    mask = np.where(div<-1.2)
    z = np.zeros(div.shape,dtype=np.uint8)
    z[mask] = 255
    for frame in range(len(z)):
        output = cv2.connectedComponentsWithStats(z[frame],8,cv2.CV_32S)
        stats = output[2]
        num_of_artifacts += sum((stats[1:,cv2.CC_STAT_AREA]>20))
    print('Found an average of {} artifacts per frame.'.format(num_of_artifacts/len(div)))
