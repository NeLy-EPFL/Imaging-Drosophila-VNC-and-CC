Semi-automated annotation
=========================

Description
-----------
This module can be used to annotated behavior frames of *Drosophila Melanogaster*.

The method is describe in:
Chen, Hermans, et al.
"Imaging neural activity in the ventral nerve cord of behaving adult Drosophila"
Nature Communications 2018

*NOTE*: The regions selection can be quite slow if a laptop touchpad is used. I recommend to use a proper mouse.

Usage
-----
The module file is executable on UNIX systems such as Mac OS and Linux.
To run it simply type `./annotation.py` in the terminal after navigating to this directory using `cd`.
If this does not work, you can run the module file with python3, e.g. from terminal using the command `python3 annotation.py`.

Dependencies
------------
python3     (tested with 3.6.5)
numpy       (tested with 1.14.3 and 1.14.4)
OpenCV      (tested with 3.4.1)
matplotlib  (tested with 2.2.2)
ffmpeg      (tested with 3.4.2 and 4.0)

License
-------
Copyright (C) 2018 F. Aymanns, florian.aymanns@epfl.ch

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
