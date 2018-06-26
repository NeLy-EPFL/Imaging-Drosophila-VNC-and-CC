Regression module
=================

Description
-----------
This module provides functions to find the pixel-wise linear regression weights for annotated behavior videos.

The method is describe in:
Chen, Hermans, et al.
"Imaging neural activity in the ventral nerve cord of behaving adult Drosophila"
Nature Communications 2018

Usage
-----
The module file is executable on UNIX systems such as Mac OS and Linux.
To run it simply type `./regression.py` in the terminal after navigating to this directory using `cd`.
If this does not work, you can run the module file with python3, e.g. from terminal using the command `python3 regression.py`.

Input
-----
GCamP images: Tiff stack of GCamP channel.
start frame for baseline: The first frame that is used for the computation of the baseline image for the dFF image.
stop frame for baseline: The last frame that is used for the computation of the baseline image for the dFF image.
behavior_idx: File containing an array with the behaviour frame corresponding to the values stored in the times file.
fluorescence_idx: File containing an array with the fluorescence frame corresponding to the values stored in the times file.
times: Time points. Should be valid for all idx and seq inputs.
walking_seq: File containing an array of booleans indicating if the fly walked at the corresponding time point.
grooming_seq: File containing an array of booleans indicating if the fly groomed at the corresponding time point.

*NOTE*: The frame rate of the behavior cameras is normally significantly larger than the fluorescence images. Therefore,
the fluorescence data is upsampled. The behavior sequences can be obtained using the annotation module provided in this repository.
The fluorescence image provided as test data is a downsampled version of the image shown in the paper.

Dependencies
------------
python3     (tested with 3.6.5)
numpy       (tested with 1.14.3 and 1.14.4)
matplotlib  (tested with 2.2.2)
sklearn     (tested with 0.19.1)
skimage     (tested with 0.14.0)

License
-------

Copyright (C) 2018 F. Aymanns, florian.aymanns@epfl.ch

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
