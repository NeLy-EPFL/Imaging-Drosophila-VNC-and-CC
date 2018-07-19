Description
-----------

This directory contains the source code of the motion compensation algorithm described in the article “Imaging neural activity in the ventral nerve cord of behaving adult Drosophila” by C. L. Chen, L. Hermans, M. C. Viswanathan, D. Fortun, M. Unser, A. Cammarato, M. H. Dickinson and P. Ramdya. The code is only for scientific or personal use.

How to use
----------

* Install the MIJ library (http://bigwww.epfl.ch/sage/soft/mij/): copy the files “ij.jar” and “mij.jar” in the java folder of MATLAB (should be <path-to-Matlab>/java/jar).
* Compile the deep matching code using the following commands in directory:
	cd code/external/deepmatching_1.2.2_c++_mac/ or cd code/external/deepmatching_1.2.2_c++_linux/ depending on your OS
	make clean all
* Edit “setPath.m”: set the paths of the MIJ library
* Run “setPath.m”
* Edit “test.m”, Section “Input - Output”
* Run “test.m”
