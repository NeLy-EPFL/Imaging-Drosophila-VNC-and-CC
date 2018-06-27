#!/usr/bin/env python3
"""
Annotation module
=================

This module provides functions to facilitate/semi-automate the annotation process
of behavior videos of *Drosophila Melanogaster*.

The method is describe in: Chen, Hermans, et al. "Imaging neural activity in the ventral nerve cord of behaving adult Drosophila", Nature Communications 2018

NOTE: The regions selection can be quite slow if a laptop touchpad is used. I recommend to use a proper mouse.

Dependencies
------------
python3     (tested with 3.6.5)
numpy       (tested with 1.14.3 and 1.14.4)
OpenCV      (tested with 3.4.1)
matplotlib  (tested with 2.2.2)
ffmpeg      (tested with 3.4.2 and 4.0)

Copyright (C) 2018 F. Aymanns, florian.aymanns@epfl.ch

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import numpy as np
import math
import cv2
import glob
import os.path
from matplotlib import pyplot as plt
from subprocess import call
import sys

regions = []

def get_regions():
    """
    This function returns a list of regions set for the annotation.
    The format of one region is [x_min,x_max,y_min,y_max].

    Returns
    -------
    regions : list of list
        Has the following format: [[x_min,x_max,y_min,y_max],[[x_min,x_max,y_min,y_max],...]
    """
    global regions
    return regions

def set_regions(new_regions):
    """
    This function can set the regions used for the annotation.

    Parameters
    ----------
    new_regions : list of lists
        New regions in the following format: [[x_min,x_max,y_min,y_max],[x_min,x_max,y_min,y_max],...]
    """
    global regions
    regions = new_regions

class NoRegionsSpecified(Exception):
    """Raised if no Regions are specified for saving."""
    pass

def save_regions(path, regions=None):
    """
    This function saves the regions to file.

    Parameters
    ----------
    path : string
        String specifying the path under which the files is saved.
    regions : list of lists, optional
        Regions in the following format: [[x_min,x_max,y_min,y_max],[x_min,x_max,y_min,y_max],...].
        If not given, the regions selected for the annotation are used.

    Raises
    ------
    NoRegionsSpecified
        If no regions were passed to function by parameter and no regions were previously set.
    """
    path = os.path.expanduser(os.path.expandvars(path))
    if regions is not None:
        np.savetxt(path, regions)
    else:
        if len(globals()['regions']) >= 1:
            raise NoRegionsSpecified('No regions given as parameter and no regions were previously set.')
        np.savetxt(globals()['regions'])

def load_regions(path):
    """
    This functions loads regions form file.

    Parameters
    ----------
    path : string
        Path to file from which regions are loaded.
    """
    path = os.path.expanduser(os.path.expandvars(path))
    global regions
    regions = np.genfromtxt(path)
    return regions

def region_selection_callback(event, x, y, flags, param):
    """
    This is the callback function used for region selection using the mouse.
    Is not intended for use outside of this module.

    To select ROIs use the function :func:`~nely_suite.annotation.region_selection` of this module.
    """
    global regions
    if event==cv2.EVENT_LBUTTONDOWN:
        regions.append([x,-1,y,-1])
    elif event==cv2.EVENT_LBUTTONUP:
        pass
        if x > regions[-1][0]:
            regions[-1][1] = int(x)
        else:
            tmp = regions[-1][0]
            regions[-1][0] = int(x)
            regions[-1][1] = tmp
        if y > regions[-1][2]:
            regions[-1][3] = int(y)
        else:
            tmp = regions[-1][2]
            regions[-1][2] = int(y)
            regions[-1][3] = tmp
        cv2.rectangle(param['image'], (regions[-1][0], regions[-1][2]), (regions[-1][1], regions[-1][3]), (255, 0, 0), 1)
        cv2.imshow(param['window_name'],param['image'])
        #TODO find function that forces window to update
        len_reg = len(regions)
        if len_reg==1:
            print('Region selected for walking: ',regions[-1])
            print("Select region for grooming in front of the fly's head!")
        elif len_reg==2:
            print('Region selected for grooming: ',regions[-1])
            print("Press 'a' to accept the selected regions. Press 'r' to remove the last selection.")
        elif len_reg>2:
            print('Too many regions were selected. {} selected but 2 were expected.'.format(len_reg))

def region_selection(img, default_regions=[[180, 475, 220, 300],[480, 540, 160, 300]]):
    """
    This function can be used to select ROIs on an image for annotation.
    Regions are selected by click and release on the displayed image.
    Pressing 'r' resets the region selected last.
    Pressing 'a' accepts the selected regions.
    `NOTE:` Walking region has to be selected first.

    Parameters
    ----------
    img : 2D np.array
        Image on which the regions are selected.
    default_regions : list of lists
        Regions used if 'a' is pressed before any regions were selected.
        Format is: [[x_min,x_max,y_min,y_max],[[x_min,x_max,y_min,y_max],...]

    Returns
    -------
    regions : list of lists
        Selected regions in format [[x_min,x_max,y_min,y_max],[[x_min,x_max,y_min,y_max],...]
    """
    global regions

    img_clone = img.copy()

    cv2.namedWindow('ROI selection')
    cv2.moveWindow('ROI selection', 0, 0)
    cv2.setMouseCallback('ROI selection', region_selection_callback, {'window_name':'ROI selection', 'image':img})
    
    print("Press 'a' if you want to use the default regions.\nOtherwise, select a region for walking positioned over the fly's hind and middle legs. (Click, drag, and release to select a regions.) ")

    while True:
        assert (cv2.getWindowProperty("ROI selection",0)>=0), 'Window closed before regions selection was accepted!'
        cv2.imshow('ROI selection',img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            #reset selected positions
            img = img_clone.copy()
            regions = regions[:-1]
            for r in regions:
                cv2.rectangle(img, (r[0], r[2]), (r[1], r[3]), (255, 0, 0), 1)
            cv2.imshow('ROI selection', img) 
            cv2.setMouseCallback('ROI selection', region_selection_callback, {'window_name':'ROI selection', 'image':img})

        if key == ord('a'):
            #accept selected positions
            print(len(regions))
            if len(regions) == 0:
                regions.append(default_regions[0])
                regions.append(default_regions[1])
            elif len(regions) == 1:
                print('Select a region in front of the flies head for grooming!\n')
            elif len(regions) > 2:
                print("\n\nExactly two regions should be selected!\nThe first one for walking over the hind and middle legs and the second one for grooming in front of the flies head!\nYou can remove the last regions by pressing the 'r' key.")
            else:
                break
    cv2.destroyAllWindows()
    return regions

def hysteresis_filter(seq, n=5):
    """
    This function implements a hysteresis filter for boolean sequences.
    The state in the sequence only changes if n consecutive element are in a different state.

    Parameters
    ----------
    seq : 1D np.array of type boolean
        Sequence to be filtered.
    n : int, default=5
        Length of hysteresis memory.

    Returns
    -------
    seq : 1D np.array of type boolean
        Filtered sequence.
    """
    seq = seq.astype(np.bool)
    state = seq[0]
    start_of_state = 0
    memory = 0
    for i in range(len(seq)):
        if state != seq[i]:
            memory += 1
        elif memory < n:
            memory = 0
            continue
        if memory == n:
            seq[start_of_state:i-n+1]=state
            start_of_state = i-n+1
            state = not state
            memory = 0
    seq[start_of_state:]=state
    return seq


def plot_histogram(values, threshold=None, nbins=100, save_path=''):
    """
    Plot histogram with threshold line.

    Parameters
    ----------
    values : 1D array like
        List of values used for histogram.
    threshold : float, optional
        Position of vertical threshold line.
    nbins : int, default=100
        Number of bins in histogram.
    save_path : string, optional
        Location where plot is saved.
    """
    plt.figure()
    plt.hist(values, bins=nbins)
    if threshold is not None:
        plt.axvline(x=threshold)
    if save_path != '':
        plt.savefig(save_path)
    plt.show()

class NoFilesError(Exception):
    """Exception raised if no image files are found."""
    pass

def find_image_files(path):
    """
    Returns list of image files of a specific format in the path specified.

    Takes a UNIX style path (~ and * work) and returns list of paths to files.
    If the last character is not '*' and the input path does not end with a file extension,
    the first extension, for which a file exist, of the following list is used.
    '.png','.jpg','.jpeg','.jpe','.jp2','.pbm','.pgm','.ppm','.sr','.ras','.tiff','.tif','.bmp','.dib','.webp'

    Parameters
    ----------
    path : string
        Path to directory which is searched.

    Returns
    -------
    paths : list of strings
        List of paths to files.

    Raises
    ------
    NoFilesError
        Raised if no image files could be found in the specified directory.
    """
    path = os.path.expanduser(os.path.expandvars(path))
    path, file_extension = os.path.splitext(path)
    possible_extensions = ['.png','.jpg','.jpeg','.jpe','.jp2','.pbm','.pgm','.ppm','.sr','.ras','.tiff','.tif','.bmp','.dib','.webp']
    if file_extension=='' and path[-1] != '*':
        all_files = glob.glob(os.path.join(path, '*'))
        for p in all_files:
            p_path, p_extension = os.path.splitext(p);
            if p_extension in possible_extensions:
                file_extension = '*' + p_extension
                break
        if file_extension=='':
            raise NoFilesError('No image files of supported formats were found under the input path!')
        
    paths = sorted(glob.glob(os.path.join(path, file_extension)))
    return paths

def show_frame(paths, index, show_walking_ROI=False, show_grooming_ROI=False):
    """
    Displays a frame of the behavior video unprocessed and the binary for motion detection.

    Parameters
    ----------
    paths : list of strings
        Paths to images of behavior video.
    index : int
        Index of the frame that should be displayed.
    show_walking_ROI : boolean, default=False
        Display walking region in extra window.
    show_grooming_ROI : boolean, default=False
        Display grooming region in extra window.
    """
    previous_img = cv2.imread(paths[index], cv2.IMREAD_GRAYSCALE)
    ret, previous_img = cv2.threshold(previous_img, 132, 255, cv2.THRESH_BINARY)
    p = paths[index+1]
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    ret, thresh_img = cv2.threshold(img, 132, 255, cv2.THRESH_BINARY)
    diff_img = abs(thresh_img - previous_img)
    median_blurred_diff_img = cv2.medianBlur(diff_img, 5)
    cv2.imshow('Frame {}'.format(index),np.hstack((img, median_blurred_diff_img)))
    cv2.moveWindow('Frame {}'.format(index), 0, 0)
    if show_walking_ROI:
        cv2.imshow('Walking regions', median_blurred_diff_img[regions[-2][2]:regions[-2][3], regions[-2][0]:regions[-2][1]])
        cv2.moveWindow('Walking regions', 0, img.shape[0])
    if show_grooming_ROI:    
        cv2.imshow('Grooming regions', median_blurred_diff_img[regions[-1][2]:regions[-1][3], regions[-1][0]:regions[-1][1]])
        cv2.moveWindow('Grooming regions', regions[-2][1]-regions[-2][0], img.shape[0])

def annotate(paths,
            hysteresis_memory_walking=5,
            hysteresis_memory_grooming=5,
            threshold_walking=None,
            threshold_grooming=None,
            verbosity=0,
            save_raw=False):
    """
    This function produces annotation sequences of walking and grooming for behavior videos.

    Parameters
    ----------
    paths : list of strings
        Paths to frames of behavior video.
    hysteresis_memory_walking : int, default=5
        Length for :func:`~nely_suite.annotation.hysteresis_filter` applied to walking sequence.
    hysteresis_memory_grooming : int, default=5
        Length for :func:`~nely_suite.annotation.hysteresis_filter` applied to grooming sequence.
    threshold_walking : float, optional
        Threshold for walking region in number of pixels. Used to detect motion. For details see paper.
        If not specified, the Otsu threshold is used.
    threshold_grooming : float, optional
        Threshold for grooming region in number of pixels. Used to detect motion. For details see paper.
        If not specified, the Otsu threshold is used.
    verbosity : int, default=0
        Changes amount of output.
    save_raw : boolean, default=False
        Saves unfiltered sequences as text files.

    Returns
    -------
    walking_seq : 1d np.array of type boolean
        Sequence indicating grooming for each frame.
    grooming_seq : 1d np.array of type boolean
        Sequence indicating grooming for each frame.

    Examples
    --------
    >>>paths = find_image_files(path)
    >>>img = cv2.imread(paths[0],cv2.IMREAD_GRAYSCALE)
    >>>region_selection(img)
    >>>[walking_seq, grooming_seq] = annotate(paths, hysteresis_memory=5, save_raw=True)
    """

    global regions
    n_frames = len(paths)

    previous_img = cv2.imread(paths[0], cv2.IMREAD_GRAYSCALE)
    ret, previous_img = cv2.threshold(previous_img, 132, 255, cv2.THRESH_BINARY)
    nonzero_count_walking = np.zeros(n_frames-1)
    nonzero_count_grooming = np.zeros(n_frames-1)
    i = 0
    for p in paths[1:]:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        ret, thresh_img = cv2.threshold(img, 132, 255, cv2.THRESH_BINARY)
        diff_img = abs(thresh_img - previous_img)
        median_blurred_diff_img = cv2.medianBlur(diff_img, 5)
        nonzero_count_walking[i] = np.count_nonzero(median_blurred_diff_img[regions[-2][2]:regions[-2][3], regions[-2][0]:regions[-2][1]])
        nonzero_count_grooming[i] = np.count_nonzero(median_blurred_diff_img[regions[-1][2]:regions[-1][3], regions[-1][0]:regions[-1][1]])
        previous_img = thresh_img
        i += 1

    ret_walking, thresh_walking = cv2.threshold( (nonzero_count_walking/max(nonzero_count_walking)*255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret_grooming, thresh_grooming = cv2.threshold((nonzero_count_grooming/max(nonzero_count_walking)*255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    if threshold_walking is None:
        threshold_walking = ret_walking / 255 * max(nonzero_count_walking)
        print('Using threshold_walking = {}'.format(threshold_walking))
    if threshold_grooming is None:
        threshold_grooming = ret_grooming / 255 * max(nonzero_count_grooming)
        print('Using threshold_grooming = {}'.format(threshold_grooming))

    if verbosity>0:
        plot_histogram(nonzero_count_walking, threshold_walking, save_path='histogram_walking.pdf')
        plot_histogram(nonzero_count_grooming, threshold_grooming, save_path='histogram_grooming.pdf')
        #show_frame(path,np.argmax(nonzero_count_walking), True, True)
        #cv2.waitKey(0)
        #show_frame(path,np.argmax(nonzero_count_grooming), True, True)
        #cv2.waitKey(0)

    walking_seq  = (nonzero_count_walking > threshold_walking)
    grooming_seq = (nonzero_count_grooming > threshold_grooming)
    if save_raw:
        np.savetxt('raw_walking.txt', walking_seq)
        np.savetxt('raw_grooming.txt', grooming_seq)

    grooming_seq[walking_seq] = False
    walking_seq = hysteresis_filter(walking_seq, hysteresis_memory_walking)
    grooming_seq[walking_seq] = False
    grooming_seq = hysteresis_filter(grooming_seq, hysteresis_memory_grooming)
    grooming_seq[walking_seq] = False
    
    return walking_seq, grooming_seq

def make_annotated_video(paths, walking_seq, grooming_seq, save_path, draw_ROIs=False):
    """
    Produces video with 'W' and 'G' in upper left corner to indicate walking and grooming,
    respectively.

    Parameters
    ----------
    paths : list of strings
        Paths to frames of behavior video.
    walking_seq : 1D np.array of type boolean
        Sequence indicating walking.
    grooming_seq : 1D np.array of type boolean
        Sequence indicating grooming.
    save_path : string
        Location where video file is saved.
    draw_ROIs : boolean, default=False
        Draw rectangles for ROIs on video or not.
    """
    if not os.path.exists('output'):
        os.makedirs('output')
    for i in range( len(paths)-1 ):
        img = cv2.imread(paths[i+1], cv2.IMREAD_GRAYSCALE)
        if draw_ROIs:
            for j in range( len(regions) ):
                cv2.rectangle(img, (regions[j][0], regions[j][2]), (regions[j][1], regions[j][3]), (255, 0, 0), 1)
        if walking_seq[i]:
                cv2.putText(img, 'Walking', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        elif grooming_seq[i]:
                cv2.putText(img, 'Grooming', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        #cv2.putText(img, str(i), (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        cv2.imwrite('output/frame{}.png'.format(i), img)
    call(['ffmpeg', '-y', '-r', '30', '-i', 'output/frame%d.png', save_path])


if __name__=="__main__":
    data_dir = input("Please enter path to directory with behavior frames (e.g. './test_data'): ")
    
    paths = find_image_files(data_dir)
    img = cv2.imread(paths[0])
    regions = region_selection(img,default_regions=[[199, 389, 232, 288],[510, 530, 130, 265]])
    [walking_seq, grooming_seq] = annotate(paths,
                                           hysteresis_memory_walking=8,
                                           hysteresis_memory_grooming=10,
                                           threshold_walking=400,
                                           threshold_grooming=4.5,
                                           save_raw=False)
    pathname = os.path.dirname(sys.argv[0])
    pathname = os.path.abspath(pathname)
    np.savetxt('walking_seq.txt',walking_seq)
    np.savetxt('grooming_seq.txt',grooming_seq)
    make_annotated_video(paths, walking_seq, grooming_seq, 'annotated_video.mp4', True)
    print('\n\nSaved annotation sequences and annotated video in '+pathname)
