CC=g++

OS_NAME=$(shell uname -s)
ifeq ($(OS_NAME),Linux) 
#old#  LAPACKLDFLAGS=/usr/lib64/atlas/libsatlas.so   # single-threaded blas
  LAPACKLDFLAGS=/usr/lib/atlas-base/libatlas.so /usr/lib/libblas/libblas.so
  #LAPACKLDFLAGS=/usr/lib64/atlas/libtatlas.so  # multi-threaded blas
  #BLAS_THREADING=-D MULTITHREADED_BLAS # remove this if wrong
endif
ifeq ($(OS_NAME),Darwin)  # Mac OS X
  LAPACKLDFLAGS=-framework Accelerate # for OS X
endif
LAPACKCFLAGS=-Dinteger=int $(BLAS_THREADING)
#old#
#STATICLAPACKLDFLAGS=-fPIC -Wall -g -fopenmp -static -static-libstdc++ /home/lear/douze/tmp/jpeg-6b/libjpeg.a /usr/lib64/libpng.a /usr/lib64/libz.a /usr/lib64/libblas.a /usr/lib/gcc/x86_64-redhat-linux/4.9.2/libgfortran.a /usr/lib/gcc/x86_64-redhat-linux/4.9.2/libquadmath.a # statically linked version

STATICLAPACKLDFLAGS=-fPIC -Wall -g -fopenmp -static -static-libstdc++ /home/lear/douze/tmp/jpeg-6b/libjpeg.a /usr/lib/x86_64-linux-gnu/libpng.a /usr/lib/x86_64-linux-gnu/libz.a /usr/lib/libblas/libblas.a /usr/lib/gcc/x86_64-linux-gnu/4.8/libgfortran.a /usr/lib/gcc/x86_64-linux-gnu/4.8/libquadmath.a # statically linked version

#old#
#CFLAGS= -fPIC -Wall -g -std=c++11 $(LAPACKCFLAGS) -fopenmp -DUSE_OPENMP -O3
CFLAGS= -fPIC -Wall -g -std=c++11 $(LAPACKCFLAGS)  -O3
LDFLAGS=-fPIC -Wall -g -ljpeg -lpng  
CPYTHONFLAGS=-I/usr/include/python2.7

SOURCES := $(shell find . -name '*.cpp' ! -name 'deepmatching_matlab.cpp')
OBJ := $(SOURCES:%.cpp=%.o)
HEADERS := $(shell find . -name '*.h')


all: deepmatching 

.cpp.o:  %.cpp %.h
	$(CC) $(CFLAGS) -c $+ -o $@

deepmatching:  $(OBJ)
	$(CC) -o $@ $(LDFLAGS) $(LAPACKLDFLAGS) $^

deepmatching-static: $(HEADERS) $(OBJ)
	$(CC) $^ $(STATICLAPACKLDFLAGS) -o $@

python: $(HEADERS) $(OBJ)
#	swig -python $(CPYTHONFLAGS) deepmatching.i # not necessary, only do if you have swig compiler
	g++ $(CFLAGS) -c deepmatching_wrap.c $(CPYTHONFLAGS)
	g++ -shared $(LDFLAGS) $(LAPACKLDFLAGS) deepmatching_wrap.o $(OBJ) -o _deepmatching.so $(LIBFLAGS) 

clean:
	rm -f $(OBJ) deepmatching *~ *.pyc .gdb_history deepmatching_wrap.o _deepmatching.so deepmatching.mex???

