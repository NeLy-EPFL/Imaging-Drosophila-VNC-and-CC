# -*- coding: utf-8 -*-
"""

Goal:
    This script allows the user to verify that the ROIs were correctly selected from the automatic ROI procedure on all or on a range of frames.
    
Method:
    The script opens previously selected frames with the ROIs drawn on them and presents the frames one by one to the user.
    The user says whether the ROIs are correctly selected or not and, if not, all the objects that can be detected within the frame using the OpenCV library (and the procedure
    previously described in the automatic detection script) are presented to the user for ROI selection. If the detection is still incorrect or if the shape of the ROI 
    detected is not the good one, the user can select the frame to be stored and come back to this frame with the manual drawing script later (P3).
    The DR/R*100 are then recalculated once all the new ROIs have been selected and stored in a text file. The corrected frames with the new contours drawn on them are also 
    stored. 
     
"""

from skimage import io
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os, os.path
import sys
import shutil
from PIL import Image, ImageDraw
import time

dataDir = 'YOUR_PATH_TO_EXPERIMENT_FOLDER'

outDir= dataDir + '/output/'

outDirFrameNo_Usrselected = outDir + 'Frame#_UsrSelected/'
outDirROI_UsrCorrected = outDir + 'ROI_UsrCorrected/'
outDirGC6_UsrCorrected = outDir + 'GC6_UsrCorrected/'
outDircropROI_UsrCorrected = outDir + 'cropROI_UsrCorrected/'

outDirROI_auto = outDir + 'ROI_auto/'
outDirGC6_auto = outDir + 'GC6_auto/'
outDircropROI_auto = outDir + 'cropROI_auto/'

imgStackFileNameG = 'GC6s.tif'
imgStackFileNameR = 'tdTom.tif'
imgStackFileNameRGB = 'RGB.tif'
imgStackFileNameRGB1 = 'RGB1.tif'
imgStackFileNameRGB2 = 'RGB2.tif'
imgStackFileNameRGBtemp = 'RGBtemp.tif'
imgStackFileNameRGBNan = 'RGBNan.tif'

#%% COPY RGB file for different purposes
shutil.copyfile(dataDir + '/registered/' + imgStackFileNameRGB,dataDir + '/registered/' + imgStackFileNameRGB1) 
shutil.copyfile(dataDir + '/registered/' + imgStackFileNameRGB,dataDir + '/registered/' + imgStackFileNameRGB2) 
shutil.copyfile(dataDir + '/registered/' + imgStackFileNameRGB,dataDir + '/registered/' + imgStackFileNameRGBtemp) 
shutil.copyfile(dataDir + '/registered/' + imgStackFileNameRGB,dataDir + '/registered/' + imgStackFileNameRGBNan) 

#%% IMPORT imaging data
gcamp_chan = io.imread(dataDir + '/registered/' + imgStackFileNameG)
tdTom_chan = io.imread(dataDir + '/registered/' + imgStackFileNameR)
RGB_chan = io.imread(dataDir + '/registered/' + imgStackFileNameRGB)
RGB1_chan = io.imread(dataDir + '/registered/' + imgStackFileNameRGB1)
RGB2_chan = io.imread(dataDir + '/registered/' + imgStackFileNameRGB2)
RGBtemp_chan = io.imread(dataDir + '/registered/' + imgStackFileNameRGBtemp)
RGBtemp_nan = io.imread(dataDir + '/registered/' + imgStackFileNameRGBNan)

if (not os.path.exists(outDirGC6_UsrCorrected))==True:
	os.makedirs(outDirGC6_UsrCorrected)

del imgStackFileNameG, imgStackFileNameR, imgStackFileNameRGB


L_GCamP6_abs_UI=[]
R_GCamP6_abs_UI=[]
L_tdtom_abs_UI=[]
R_tdtom_abs_UI=[]
start = 0
stop = len(tdTom_chan)
blurVal = 6
blurVal_BiFltr=120
blurVal_med=5
counter_array=0
#erode val = value for the erosion applied via the kernel - bigger value then ROI will be smaller as more erosion etc. 
erodeVal = 1
sizeExcl = 2
cropLength=35
bin_s=2.5
sample_rate_hz=2.418032787 
WrongDetection = []
frames_is_correct = []
frames_for_fix = []
k_array=[]
count = 0


def UIFramesAlreadyChecked():
	"""
	This function checks if the script was already run and paused. If yes, the user will be forced to finish the paused selection.
	"""
	Yes = ["Y","y"]
	No = ["N","n"]

	finished1 = False
	while not finished1:
	    if os.path.exists(outDirFrameNo_Usrselected + "FrameToFix_Pause" + '.txt'):
	        print('\033[1m' + "Frames partially reviewed, you will be asked to correct the frames range that was left after you paused the script. If you want to correct another range of frames, you first have to finish the paused selection and to run the script again once this selection is completed.") 
	        print '\033[0m'
	        finished1 = True
	        return 1

	    elif os.path.exists(outDirFrameNo_Usrselected):
	        return 0
	    else:
	        os.makedirs(outDirFrameNo_Usrselected) 
	        return 0

def OpenOldData():
	"""
	This function opens the range of frames to manually correct if the script was run and paused.
	"""
	frameToFix = []

	readframes_for_fix = open(outDirFrameNo_Usrselected +'FrameToFix_Pause.txt')
	string_tofix=(readframes_for_fix.read()).split("\n")

	for i in range(0,len(string_tofix)-1): #The last element is blank. 
	    frames_for_fix.append(int(string_tofix[i]))

	frameRange1 = frames_for_fix[-1]
	frameRange0 = frames_for_fix[-2]
	k = frames_for_fix[-3]
	count = frames_for_fix[-4]
	frames_for_fix.pop()
	frames_for_fix.pop()
	frames_for_fix.pop()
	frames_for_fix.pop()

	frameRange = []
	frameRange.append(frameRange0)
	frameRange.append(frameRange1)

	if  os.path.exists(outDirFrameNo_Usrselected +'frames_are_correct_Pause.txt'):
	    correctFrames = open(outDirFrameNo_Usrselected +'frames_are_correct_Pause.txt')
	    string_correctFrames=(correctFrames.read()).split("\n")

	    for i in range(0,len(string_correctFrames)-1): 
	        frames_is_correct.append(int(string_correctFrames[i]))

	for i in range(frameRange[0],frameRange[-1]+1):
		frameToFix.append(i)

	return k, frameToFix, count

def GoOnData(k,FrameRange,count):
	"""
	This function is called only if the script was run and paused and presents the number of frames that are left to correct. If the user does not
	want to follow his correction, the script will close.
	"""
	yesKeyin = ["y","Y"]
	noKeyin =  ["n","N"]

	finished2 = False
	while not finished2:
	    print("*** You stopped on frame: ",k, "in the frame range :",FrameRange[0],FrameRange[-1],". There are ",len(FrameRange)-count,"frames to correct***") 
	    frameKeyin=raw_input("Do you want to go on with this frame Selection ? (Press Y for yes and N for no, if no, script will end)\n->") 
	    newFrameKeyin = unicode(frameKeyin,'utf-8') 
	    try:
	        finished2 = True
	        if newFrameKeyin in yesKeyin:
	            FrameRange[0]=k
	        elif newFrameKeyin in noKeyin:
	                exit(0)
	                return 0
	        else:
	            print("Incorrect format of input, please try again!\n")
	            finished2 = False
	    except IndexError:
	        print("Wrong format")

	return FrameRange


def OpenPausedValues(L_GC,R_GC,L_TT,R_TT,WrongDetection):
	"""
	This function opens the fluorescence values if the script was paused.
	"""
	LGCTemp = []
	LTTTemp = []
	RGCTemp = []
	RTTTemp = []
	WrongDetTemp = []
	WrongDetPart = []
	WrongDetAllList = []

	if os.path.exists(outDirGC6_UsrCorrected +'L_GC_Temp.txt')==True: 
	    L_GC_Temp = open(outDirGC6_UsrCorrected +'L_GC_Temp.txt')
	    
	    L_GC_Temp_split=(L_GC_Temp.read()).split("\n")

	    for i in range(0,len(L_GC_Temp_split)-1): 
	        LGCTemp.append(float(L_GC_Temp_split[i]))

	if os.path.exists(outDirGC6_UsrCorrected +'L_TT_Temp.txt')==True: 
	    L_TT_Temp = open(outDirGC6_UsrCorrected +'L_TT_Temp.txt')
	    
	    L_TT_Temp_split=(L_TT_Temp.read()).split("\n")

	    for i in range(0,len(L_TT_Temp_split)-1):
	        LTTTemp.append(float(L_TT_Temp_split[i]))

	if os.path.exists(outDirGC6_UsrCorrected +'R_GC_Temp.txt')==True: 
	    R_GC_Temp = open(outDirGC6_UsrCorrected +'R_GC_Temp.txt')
	    
	    R_GC_Temp_split=(R_GC_Temp.read()).split("\n")

	    for i in range(0,len(R_GC_Temp_split)-1):
	        RGCTemp.append(float(R_GC_Temp_split[i]))

	if os.path.exists(outDirGC6_UsrCorrected +'R_TT_Temp.txt')==True: 
	    R_TT_Temp = open(outDirGC6_UsrCorrected +'R_TT_Temp.txt')
	    
	    R_TT_Temp_split=(R_TT_Temp.read()).split("\n")

	    for i in range(0,len(R_TT_Temp_split)-1):
	        RTTTemp.append(float(R_TT_Temp_split[i]))

	if os.path.exists(outDirGC6_UsrCorrected +'WrongDetection_Temp.txt')==True: 
	    WD_Temp = open(outDirGC6_UsrCorrected +'WrongDetection_Temp.txt')
	    
	    WD_Temp_split=(WD_Temp.read()).split("\n")

	    for i in range(0,len(WD_Temp_split)-1):
	        WrongDetTemp.append(WD_Temp_split[i])

	if (not WrongDetTemp) == False:
		print ("Opening paused wrong detection frames number")
		for i in range(len(WrongDetTemp)):
			WrongDetAllList.append(int(WrongDetTemp[i]))

	if (not LGCTemp) == True:
		print ("Opening auto fluorescence values")
		return L_GC,R_GC,L_TT,R_TT,WrongDetection

	else : 
		print ("Opening paused fluorescence values")
		return LGCTemp, RGCTemp, LTTTemp, RTTTemp, WrongDetAllList

def checkWrongDetExist(WrongDetection):
	"""
	This function opens the txt file that stores the frame number to be given to script P3 for manual drawing of ROI.
	"""
	if os.path.exists(outDirGC6_UsrCorrected +'wrongROIShape.txt')==True:
		WrongDetPrev = []
		WrongDetPrevAllList = []
		WD_Prev = open(outDirGC6_UsrCorrected +'wrongROIShape.txt')
		WD_Prev_split=(WD_Prev.read()).split("\n")
		for i in range(0,len(WD_Prev_split)-1):
			WrongDetPrev.append(WD_Prev_split[i])
		if (not WrongDetPrev) == False:
			for i in range(len(WrongDetPrev)):
				if int(WrongDetPrev[i]) not in WrongDetection:
					WrongDetection.append(int(WrongDetPrev[i]))

	return

def OpenOldAbsValues():
	"""
	This function loads the fluorescence means of every ROI that were calculated by the autoselection script. It returns 4 lists containing the 
	fluorescence means per channel for Left and Right ROI of GCaMP6s and tdTom channels.
	"""
	if os.path.exists(outDirGC6_UsrCorrected+'L_GC_orig_Usr.txt')==True:
	    print(">>>Reading GC_orig_Usr...")
	    if os.path.exists(outDirGC6_UsrCorrected+'L_GC_orig_Usr.txt')==True: 
	        fileGC_L = open(outDirGC6_UsrCorrected+'L_GC_orig_Usr.txt')
	    if os.path.exists(outDirGC6_UsrCorrected+'R_GC_orig_Usr.txt')==True: 
	        fileGC_R = open(outDirGC6_UsrCorrected+'R_GC_orig_Usr.txt')
	    if os.path.exists(outDirGC6_UsrCorrected+'L_tdtom_orig_Usr.txt')==True: 
	        fileTT_L = open(outDirGC6_UsrCorrected+'L_tdtom_orig_Usr.txt')
	    if os.path.exists(outDirGC6_UsrCorrected+'R_tdtom_orig_Usr.txt')==True: 
	        fileTT_R = open(outDirGC6_UsrCorrected+'R_tdtom_orig_Usr.txt')        
	    
	else:
	    print(">>>Reading GC_orig_auto...")
	    if os.path.exists(outDirGC6_auto+'L_GC_orig.txt')==True: 
	        fileGC_L = open(outDirGC6_auto+'L_GC_orig.txt')
	    if os.path.exists(outDirGC6_auto+'R_GC_orig.txt')==True: 
	        fileGC_R = open(outDirGC6_auto+'R_GC_orig.txt')
	    if os.path.exists(outDirGC6_auto+'L_tdtom_orig.txt')==True: 
	        fileTT_L = open(outDirGC6_auto+'L_tdtom_orig.txt')
	    if os.path.exists(outDirGC6_auto+'R_tdtom_orig.txt')==True: 
	        fileTT_R = open(outDirGC6_auto+'R_tdtom_orig.txt')


	L_GCamP6_abs_correct=[]
	R_GCamP6_abs_correct=[]
	L_tdtom_abs_correct=[]
	R_tdtom_abs_correct=[]

	string_readGC_L=(fileGC_L.read()).split("\n")
	string_readGC_R=(fileGC_R.read()).split("\n")
	string_readtdtom_L=(fileTT_L.read()).split("\n")
	string_readtdtom_R=(fileTT_R.read()).split("\n")

	fileGC_L.close()
	fileGC_R.close()
	fileTT_L.close()
	fileTT_R.close()

	for i in range(0,len(string_readGC_L)-1):
	    L_GCamP6_abs_correct.append(float(string_readGC_L[i]))
	    R_GCamP6_abs_correct.append(float(string_readGC_R[i]))
	for i in range(0,len(string_readtdtom_L)-1):
	    L_tdtom_abs_correct.append(float(string_readtdtom_L[i]))
	    R_tdtom_abs_correct.append(float(string_readtdtom_R[i]))

	print ("open len",len(L_GCamP6_abs_correct)   ) 

	return L_GCamP6_abs_correct,R_GCamP6_abs_correct,L_tdtom_abs_correct,R_tdtom_abs_correct

def GetFramesNumberChecked():
	"""
	This function checks if the total number of frames in the ROI_auto directory matches the number of frame in the initial .tif file. 
	If yes, the function returns the total number of frames, if not, the program is closed.
	"""
	print(">>>Import auto-selected images...")
	i_ROI_auto=0
	for k in range(0,len(tdTom_chan)):
	    try: 
	        imgROI_temp = io.imread(outDirROI_auto + "%04d" % k + '.png')
	        i_ROI_auto+=1
	        frameLastNumber=k
	    except FileNotFoundError:
	        if k==len(tdTom_chan):
	            break
	        else:
	            continue

	if k!=len(tdTom_chan)-1:
	    print("The amount of ROI_auto images doesn't match the one of images stack. This program cannot proceed...Please check ROI_auto images...")
	    sys.exit(0)

	return frameLastNumber

def GetFrameRange(LastNumber):
	"""
	This function manages the user "interface". The user can enter the range of frames he wants to review and the function will translate this range
	to a list containing two elements (the first and last frame number to review). 
	"""
	frames="0"
	frameRange=[]
	frameToFix = []

	finished = False

	while not finished:
	    print("***",LastNumber, " images for review in total (from", 0,"-",len(tdTom_chan),").***") 
	    frameKeyin=raw_input("Keyin the range of frame for correction (eg.,'25-66','0-350' or 'all')\n->") 
	    newFrameKeyin = unicode(frameKeyin,'utf-8') 

	    try:
	        finished = True
	        if frameKeyin=="all":
	            frameRange=[0,LastNumber]
	        
	        elif newFrameKeyin[0].isnumeric():
	            for i in range(0,len(newFrameKeyin)):
	                if newFrameKeyin[i].isnumeric():
	                    frames=frames+newFrameKeyin[i]
	                    if i==len(newFrameKeyin)-1:
	                        frameRange.append(int(frames))                     
	                elif newFrameKeyin[i]=="-":
	                    frameRange.append(int(frames))
	                    frames="0"                
	            if frameRange[-1]<frameRange[0]:
	                finished = False
	                del frameRange[:]
	                frames = "0"
	                print("The last value must be larger than the first one. Please keyin again!\n")
	            
	            else:
	                if len(frameRange)>2:
	                    frames = "0"
	                    del frameRange[:]
	                    finished = False
	                    print("Input cannot contain more than 3 values. Please keyin again!\n")
	                    
	                else:
	                    if frameRange[-1]>LastNumber:
	                        frames = "0"
	                        del frameRange[:]
	                        finished = False
	                        print("The input exceed the range of images. Please keyin again.\n")
	        else:
	            print("Incorrect format of input, please try again!\n")
	            finished = False

	    except IndexError:
	        print("Please keyin the range in correct format!")

	if len(frameRange)==1:
	    frameRange.append(frameRange[0])


	for i in range(frameRange[0],frameRange[-1]+1):
		frameToFix.append(i)

	return frameToFix


def createFold(tdTom_chan):
	"""
	This function creates the folder that will contain the fluorescence images with validated ROIs.
	"""
	b = 0		
	if os.path.exists(outDircropROI_UsrCorrected)==False:
		print ("copying files from auto")
		os.makedirs(outDircropROI_UsrCorrected)
		os.makedirs(outDirROI_UsrCorrected)
		b = 1

		for i in range(len(tdTom_chan)):
			try:
				shutil.copyfile(outDirROI_auto + "%04d" % i + '.png', outDirROI_UsrCorrected +  "%04d" % i + '.png') 
				shutil.copyfile(outDircropROI_auto + "%04d" % i + '.png', outDircropROI_UsrCorrected +  "%04d" % i + '.png')
			except FileNotFoundError:
				print("Error on the file")

	return b

def ImageToReview(k,b):
	"""
	This function opens the image to review and shows the image to the user. The user then needs to select if the ROIs are well detected or not.
	This function returns the string that the user entered.
	"""
	sayNotoWrongROI=["n","N"]
	sayBacktoGoback=["b","B"]
	sayPause = ["p","P"]
	sayAll = ["A","a"]
	sayOK = ['']

	if b==0:
	    try:
	        imgROI = io.imread(outDirROI_UsrCorrected + "%04d" % k + '.png')
	        print ("frame from ROI Usr Corrected")
	        k_array.append(k)
	    except :
	    	print ("File not found except 1")
	    	print (outDirROI_UsrCorrected + "%04d" % k + '.png')


	else :
		try:
			print ("frame from ROI auto")
			imgROI = io.imread(outDirROI_auto + "%04d" % k + '.png')
			k_array.append(k)
		except :
			print ("File not found except 2")
			print (outDirROI_auto + "%04d" % k + '.png')

	print ("image in review - frame number ", k)
	plt.imshow(imgROI)
	plt.pause(0.5)

	finished5 = False

	while not finished5:

		keyin=raw_input("----------------------------------\nIf wrong, keyin 'n' or 'N'.\nIf correct, press 'Enter'.\nBack to previous img, press 'b' or 'B'.\nPause and come back later, press 'p' or 'P'.\n----------------------------------\n->")
		
		if (keyin in sayNotoWrongROI) == True:
			finished5 = True
		elif (keyin in sayBacktoGoback) == True :
			finished5 = True
		elif (keyin in sayPause) == True :
			finished5 = True
		elif (keyin in sayAll) == True :
			finished5 = True
		elif (keyin in sayOK) == True :
			finished5 = True
		else : 
			print ("keyin",keyin)
			print ("Incorrect format of input, please key in again")
			finished5 = False

	plt.close()

	return keyin


def ManageUI(keyin,k,frameRange,count):
	"""
	This function receives the input from the user and manages the output -> it will store the frame as correct or not depending on the user input but it can 
	also go back to the previous frame if the user wants to correct his choice.
	"""
	BoolTo3 = 0
	sayNotoWrongROI=["n","N"]
	sayBacktoGoback=["b","B"]
	sayPause = ["p","P"]
	sayAll = ["A","a"]
	frames_for_fix = frameRange[:]

	if keyin in sayPause:
	    
  		frames_is_correct.sort()

  		frames_for_fix.append(count)
  		frames_for_fix.append(k)
  		frames_for_fix.append(frameRange[0])
  		frames_for_fix.append(frameRange[-1])

  		np.savetxt(outDirFrameNo_Usrselected + "FrameToFix_Pause" + '.txt', frames_for_fix, delimiter=' ', newline=os.linesep, fmt="%s")

		exit(0)

	elif keyin in sayNotoWrongROI:
		BoolTo3 = 1	
	        
	elif keyin in sayBacktoGoback: 
	    if len(k_array) > 1:
	    	k = k_array[-2]-1
	    	k_array.pop()
	    	k_array.pop()
	    	count -=2
	        
	    else:
	        print("This is the frame chosen to start. It cannot be moved back further.")
	        k = k_array[0]-1
	        k_array.pop()
	        count -=1
	    
	else: 
		BoolTo3 = 0


	return k, BoolTo3, count


def ThresholdingImage(ChannelAnalysis,k,blurVal,erodeval):
	"""
	This function converts the image to an 8 bit image, then extends the colour range (and actually augment 
	image contrast to better detect the ROI). We substract the baseline and then scale the max value 
	of the colour to 255(max poss value on 8 bit). Afterwards, we apply a blur (or gaussian, median, bilateral,..) filter 
	to the image before applying the Otsu Treshold. The blur filter is used to smooth the image before applying
	the threshold. Once the image is thresholded we use an erosion function to avoid very small ROI to be detected
	& to avoid larger ROI size detection due to the blur filter. If you want to increase the ROI size detection
	you should decreade the erodeval. The function returns the eroded image.
	"""
	img = (tdTom_chan[k,:,:]+gcamp_chan[k,:,:])/2
	if ChannelAnalysis == 1:
	    img = tdTom_chan[k,:,:]
	img2 = cv2.convertScaleAbs(img, alpha=(255.0/65535.0))
	minVal = np.min(img2) 
	img3 = img2 - minVal
	maxVal = np.max(img3) 
	img4 = np.floor(255/maxVal) * img3
	img5 = img4.astype(np.uint8)

	#% THRESHOLD image
	blur = cv2.blur(img5,(blurVal,blurVal),0)   
	#    blur = cv2.GaussianBlur(img5,(blurVal,blurVal),0)
	#    blur = cv2.medianBlur(img5,blurVal_med)
	#    blur = cv2.bilateralFilter(img5,1,blurVal_BiFltr,blurVal_BiFltr)
	#    blur = cv2.medianBlur(blur1 ,blurVal_med)
	#    img6 = cv2.adaptiveThreshold(img5,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
	ret, thresh = cv2.threshold(blur,1,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	kernel = np.ones((erodeVal,erodeVal), np.uint8)
	img_erosion = cv2.erode(thresh, kernel, iterations=1)

	return img_erosion


def DrawAllContours(img,erodeI,imgRGBtemp):
	"""
	This function receives the eroded image and detects all the ROIs in the frame. The contours of all ROIs are detected and are added
	to the image with the index of the contours. The function returns the image containing the contours, the contours list, the number 
	of objects detected on the frame (ret) and markers which is an array representing the image with all the object detected in it.
	"""
	ret, markers = cv2.connectedComponents(erodeIm)
	allroisC = np.zeros(np.shape(img), dtype=bool) 

	for j in range(1, ret+1): 
	    test = markers == j 
	    if np.sum(test) >= sizeExcl:
	        allroisC = allroisC + test
	thresh2C = np.array(allroisC,np.uint8) 

	_, ContoursD, _= cv2.findContours(thresh2C,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  

	imgtemp = imgRGBtemp
	countingN=0
	cv2.drawContours(imgtemp,ContoursD,-1,(255,0,0),1) 
	ContNewList = []
	wrongList = []
	for c in ContoursD :         
	    if countingN<=len(ContoursD)-1: 
	        M1 = cv2.moments(c)
	        if M1['m00'] == 0.0:
	            cx = countingN
	            cy = countingN
	            wrongList.append(countingN)
	        else :
	            cx = float(M1['m10']/M1['m00'])
	            cy = float(M1['m01']/M1['m00'])
	            cv2.putText(imgtemp, str(countingN), (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
	        
	        countingN=countingN+1

	return imgtemp, ContoursD, ret, markers, wrongList

def AskROI(img,k,lenCont,WrongDetection,wrongContList):
	"""
	This function presents the image containing all the detected ROIs to the user and the user can then select which are the good ROI.
	"""
	testList = []
	for j in range(lenCont):
		if j not in wrongContList:
			testList.append(str(j))
	testList.append("Nd")
	testList.append("nd")
	testList.append("ND")
	testList.append("nD")

	NanValDet = []
	NanValDet.append("Nd")
	NanValDet.append("nd")
	NanValDet.append("ND")
	NanValDet.append("nD")

	plt.imshow(img)
	plt.pause(0.2)
	                       
	finished1 = False
	while not finished1:    
	    finished1 = True       
	    print('which is L ROI?')
	    KeyinAnsL=raw_input('key in number (if wrong detection, key the number followed by "+X". If ROI not on frame, enter nd)->')
	    print('which is R ROI ?')
	    KeyinAnsR=raw_input('key in number (if wrong detection, key the number followed by "+X". If ROI not on frame, enter nd)->')
	    
	    LSplit = KeyinAnsL.split("+")
	    RSplit = KeyinAnsR.split("+")
	    if (LSplit[0] not in testList) == True:
	        print("Incorrect input, please try again!\n")
	        finished1 = False
	    if (RSplit[0] not in testList) == True:
	        print("Incorrect input, please try again!\n")
	        finished1 = False

	boolLN = 0
	for i in range(len(LSplit)):
		if LSplit[i] in NanValDet:
			KeyinAnsL = 0
			boolLN = 1

	boolRN = 0
	for i in range(len(RSplit)):
		if RSplit[i] in NanValDet:
			KeyinAnsR = 0
			boolRN = 1

	WrongDet1 = 'X'
	WrongDet2 = 'x'

	CurrentL =[]
	CurrentR = []


	if WrongDet1 in LSplit:
	    CurrentL = [k, "L"]
	    if (k in WrongDetection) == False:
	    	WrongDetection.append(k)
	    KeyinAnsL = LSplit[0]

	if WrongDet2 in LSplit:
	    CurrentL = [k,"L"]
	    if (k in WrongDetection) == False:
	    	WrongDetection.append(k)
	    KeyinAnsL = LSplit[0]

	if (WrongDet1 or WrongDet2) not in LSplit:
		if k in WrongDetection:
			WrongDetection.remove(k)

	if WrongDet1 in RSplit:
	    CurrentR = [k,"R"]
	    if (k in WrongDetection) == False:
	    	WrongDetection.append(k)
	    KeyinAnsR = RSplit[0]

	if WrongDet2 in RSplit:
	    CurrentR = [k,"R"]
	    if (k in WrongDetection) == False:
	    	WrongDetection.append(k)
	    KeyinAnsR = RSplit[0]

	if (WrongDet1 or WrongDet2) not in RSplit:
		if k in WrongDetection:
			WrongDetection.remove(k)


	arrayLR=[int(KeyinAnsL), int(KeyinAnsR)] 
	plt.close()

	return arrayLR, WrongDetection, boolLN, boolRN

def findingContoursNew(contoursD,ret,markers,Indexlist,imgRGB1,imgRGB2,imgRGB):
	"""
	This function calculates the centroid of each contour and draws the ones selected as left and right contours on the 3 different images
	that will be stored later. 
	"""
	ContList = []

	for l in range(len(Indexlist)):
	    Cont = contoursD[Indexlist[l]]
	    ContList.append(Cont)

	cv2.drawContours(imgRGB,ContList,-1,(255,255,255),1)

	cv2.drawContours(imgRGB1,ContList[0],-1,(255,255,255),1)
	ML = cv2.moments(ContList[0])
	cxL = float(ML['m10']/ML['m00'])
	cyL = float(ML['m01']/ML['m00'])                        
	 
	cv2.drawContours(imgRGB2,ContList[1],-1,(255,255,255),1)
	MR = cv2.moments(ContList[1])
	cxR = float(MR['m10']/MR['m00'])
	cyR = float(MR['m01']/MR['m00'])  

	return ML, MR, cxL, cxR, cyL, cyR, ContList, imgRGB

def CroppingImgNew(imgRGB1,imgRGB2,cxL,cxR,cyL,cyR):
	"""
	This function adds black edges on imgRGB1 and imgRGB2 before cropping around the ROI. 
	"""
	black_L=np.zeros((cropLength,256,3),np.int8) 
	black_C=np.zeros((2*cropLength+192,cropLength,3),np.int8) 

	imgRGB1temp=imgRGB1
	imgRGB1_1=np.row_stack((black_L,imgRGB1temp))
	imgRGB1_2=np.row_stack((imgRGB1_1,black_L))    
	imgRGB1_3=np.column_stack((black_C,imgRGB1_2))
	imgRGB1_edge=np.column_stack((-1*imgRGB1_3,black_C))


	imgRGB2temp=imgRGB2
	imgRGB2_1=np.row_stack((black_L,imgRGB2temp))
	imgRGB2_2=np.row_stack((imgRGB2_1,black_L))   
	imgRGB2_3=np.column_stack((black_C,imgRGB2_2))
	imgRGB2_edge=np.column_stack((-1*imgRGB2_3,black_C))   

	del imgRGB1temp, imgRGB1_1, imgRGB1_2, imgRGB1_3, imgRGB2temp, imgRGB2_1, imgRGB2_2, imgRGB2_3,

	cxL_Edge=int(cxL)+cropLength
	cyL_Edge=int(cyL)+cropLength
	cxR_Edge=int(cxR)+cropLength
	cyR_Edge=int(cyR)+cropLength


	imgCropL=imgRGB1_edge[cyL_Edge-cropLength:cyL_Edge+cropLength, cxL_Edge-cropLength:cxL_Edge+cropLength]
	imgCropR=imgRGB2_edge[cyR_Edge-cropLength:cyR_Edge+cropLength, cxR_Edge-cropLength:cxR_Edge+cropLength]            

	return imgCropL, imgCropR

def GetAbsFluoValues(imgGC,imgTdTom,contours,listG6,listTT,k,boolNan):
	"""
	This function calculates the fluorescence mean of the ROI selected and replace the old value by the new one in the list containing
	all the fluorescence means. 
	"""
	if boolNan == 1:
		listG6[k] = np.nan
		listTT[k] = np.nan
	else : 
		mask = np.zeros((imgGC.shape))
		cv2.fillConvexPoly(mask, contours, 1)
		mask = mask.astype(np.bool)
		masked_imgGC = imgGC*mask
		masked_imgTdTom = imgTdTom*mask
		fluorGC = masked_imgGC[masked_imgGC.nonzero()].mean()
		fluorTdTom = masked_imgTdTom[masked_imgGC.nonzero()].mean()
		listG6[k]=fluorGC
		listTT[k]=fluorTdTom

	return listG6,listTT


def CreateAndStoreImages(imgCropR,imgCropL,imgRGB):
	"""
	This function creates the combined image with the global RGB and the 2 different ROI images and saves them in the output directory
	"""
	ShapeCL = np.shape(imgCropL)
	ShapeCR = np.shape(imgCropR)
	ShapeRGB = np.shape(imgRGB)

	BDL = []
	TempLN = ShapeRGB[0]+ShapeCL[0]+1
	BDL.append(TempLN)
	BDL.append(ShapeRGB[1])
	BDL.append(ShapeRGB[2])
	imSize = tuple(BDL)

	newIm = np.zeros(imSize)
	newIm[0:ShapeRGB[0],0:ShapeRGB[1],:]=imgRGB
	newIm[ShapeRGB[0],:,:]=255
	newIm[ShapeRGB[0]+1:ShapeRGB[0]+1+ShapeCL[0],0:ShapeCL[1],:]=-imgCropL
	newIm[ShapeRGB[0]+1:ShapeRGB[0]+1+ShapeCR[0],ShapeRGB[1]/2:ShapeRGB[1]/2+ShapeCR[1],:]=-imgCropR
	newIm[ShapeRGB[0]+1:ShapeRGB[0]+1+ShapeCR[0],ShapeRGB[1]/2-1,:]=255

	imNewA = Image.fromarray(newIm.astype('uint8'))
	d = ImageDraw.Draw(imNewA)
	dLY = ShapeRGB[0]+1+ShapeCL[0]/2
	dLX = ShapeRGB[1]/2-ShapeCL[1]/2
	dRY = ShapeRGB[0]+1+ShapeCL[0]/2
	dRX = ShapeRGB[1]-ShapeCL[1]/2
	d.text((dLX,dLY),"L",fill=(255,255,255))
	d.text((dRX,dRY),"R",fill=(255,255,255))

	imNewA.save(outDirROI_UsrCorrected + "%04d" % k + '.png')

	CropSize = []
	CropColumns = ShapeCR[1]+3+ShapeCL[1]
	CropLines = ShapeCR[0] + 10
	CropSize.append(CropLines)
	CropSize.append(CropColumns)
	CropSize.append(ShapeCR[2])
	CropTup = tuple(CropSize)

	newImCrop = np.zeros(CropTup)
	newImCrop[0:ShapeCL[0],0:ShapeCL[1],:] = -imgCropL
	newImCrop[0:ShapeCR[0],ShapeCL[1]+3:,:] = -imgCropR

	imNewCrop = Image.fromarray(newImCrop.astype('uint8'))
	C = ImageDraw.Draw(imNewCrop)
	CLY = ShapeCL[0]
	CLX = ShapeCL[1]/2
	CRY = ShapeCR[0]
	CRX = ShapeCL[1]+ShapeCR[1]/2
	C.text((CLX,CLY),"L",fill=(255,255,255))
	C.text((CRX,CRY),"R",fill=(255,255,255))

	imNewCrop.save(outDircropROI_UsrCorrected + "%04d" % k + '.png')

	return

def CreateAndStoreTempFluoValues(L_GC, L_TT, R_GC, R_TT, WrongDetection):
	"""
	This function saves the temporary fluorescence values in a text file.
	"""
	WrongDetection.sort()

	np.savetxt(outDirGC6_UsrCorrected + "L_GC_Temp" + '.txt', L_GC, delimiter=' ', newline=os.linesep, fmt="%s") 

	np.savetxt(outDirGC6_UsrCorrected + "L_TT_Temp" + '.txt', L_TT, delimiter=' ', newline=os.linesep, fmt="%s")

	np.savetxt(outDirGC6_UsrCorrected + "R_GC_Temp" + '.txt', R_GC, delimiter=' ', newline=os.linesep, fmt="%s")

	np.savetxt(outDirGC6_UsrCorrected + "R_TT_Temp" + '.txt', R_TT, delimiter=' ', newline=os.linesep, fmt="%s")

	np.savetxt(outDirGC6_UsrCorrected + "WrongDetection_Temp" + '.txt', WrongDetection, delimiter=' ', newline=os.linesep, fmt="%s")

	return


def GraphValues(GFluoAbs,TFluoAbs,name):
	"""
    This function receives the mean fluorescence value per ROI and returns the normalised values to plot on final graph. Normalised fluorescence values are equal to : 
    the mean fluorescence value minus the baseline value divided by the baseline value. In the case of DFF, only the GCaMP6S signal is used while for DRR, the ratio of 
    green to red channel is used. The function also stores text files containing the fluorescent values in the output directory
    """
	samples=int(bin_s*sample_rate_hz)

	binned_G6=[]
	G6_Norm=[]
	binned_GC_TTom=[]
	binned_TTom=[]
	TT_Norm=[]

	for a in range(0,len(GFluoAbs)-samples):
	    mean_bin=np.nanmean(GFluoAbs[a:a+samples]) 
	    mean_bin_TT = np.nanmean(TFluoAbs[a:a+samples])         
	    binned_G6.append(mean_bin)
	    binned_GC_TTom.append(mean_bin/mean_bin_TT)

	baseline_GC=min(binned_G6)
	baseline_GC_TT=min(binned_GC_TTom)
	index_min_GC_TT=binned_GC_TTom.index(min(binned_GC_TTom))

	for b in range(0,len(GFluoAbs)):
	    G6_Norm.append(((GFluoAbs[b]-baseline_GC)/baseline_GC)*100)

	for b in range(0,len(TFluoAbs)):
	    TT_Norm.append(((GFluoAbs[b]/TFluoAbs[b])-baseline_GC_TT)/baseline_GC_TT*100)


	np.savetxt(outDirGC6_UsrCorrected + name + "_GC_dF_Usr" + '.txt', G6_Norm, delimiter=' ', newline=os.linesep, fmt="%s") 

	np.savetxt(outDirGC6_UsrCorrected + name +"_GC_tdtom_norm_Usr" + '.txt', TT_Norm, delimiter=' ', newline=os.linesep, fmt="%s")

	np.savetxt(outDirGC6_UsrCorrected + name + "_GC_orig_Usr" + '.txt', GFluoAbs, delimiter=' ', newline=os.linesep, fmt="%s")

	np.savetxt(outDirGC6_UsrCorrected + name + "_tdtom_orig_Usr" + '.txt', TFluoAbs, delimiter=' ', newline=os.linesep, fmt="%s")


	return G6_Norm, TT_Norm

def storeWrongRoiDetection():
	"""
	This function stores the frames that need to undergo the manual drawing script (P3) in a text file.
	"""
    np.savetxt(outDirGC6_UsrCorrected + "wrongROIShape" + '.txt', WrongDetection, delimiter=' ', newline=os.linesep, fmt="%s")

    return

def drawOneContour(ContoursD,ret,markers,NewROIList,imgRGB,name):
	"""
	This function draws only one contour if one ROI could be detected instead of two.
	"""
	ContList = []

	if name == 'R':
		ContList.append(ContoursD[NewROIList[1]])

	if name == 'L':
		ContList.append(ContoursD[NewROIList[0]])

	cv2.drawContours(imgRGB,ContList,-1,(255,255,255),1)

	return imgRGB

# *************** MAIN *****************

print dataDir
UIselect = UIFramesAlreadyChecked()
L_GC,R_GC,L_TT,R_TT = OpenOldAbsValues()
if UIselect ==1:
	k, FrameRange, count = OpenOldData()
	FrameRange = GoOnData(k,FrameRange, count)
	L_GC,R_GC,L_TT,R_TT, WrongDetection = OpenPausedValues(L_GC,R_GC,L_TT,R_TT,WrongDetection)
else : 
	LastNum = GetFramesNumberChecked()
	FrameRange = GetFrameRange(LastNum)
	k=FrameRange[0]
b = createFold(tdTom_chan)
checkWrongDetExist(WrongDetection)
StoredFrameToFix = len(FrameRange)
newSavingList = FrameRange[:]

while k in range(FrameRange[0],FrameRange[-1]+1):

	print ("There are", StoredFrameToFix-count , "frames to fix")
	keyInUI = ImageToReview(k,b)
	k, BoolTo3, count = ManageUI(keyInUI,k,FrameRange,count)

	if BoolTo3 == 1 :
		imgRGBtemp = RGBtemp_chan[k,:,:,:]
		img = gcamp_chan[k,:,:]
		erodeIm = ThresholdingImage(0,k,blurVal,erodeVal)
		imgTemp, ContoursD, ret, markers, wrongContoursList= DrawAllContours(img,erodeIm,imgRGBtemp)
		NewROIList, WrongDetection, boolLN, boolRN = AskROI(imgTemp,k,len(ContoursD),WrongDetection,wrongContoursList)
		ML, MR, cxL, cxR, cyL, cyR, ContList, imgRGB = findingContoursNew(ContoursD,ret,markers,NewROIList,RGB1_chan[k,:,:,:],RGB2_chan[k,:,:,:],RGB_chan[k,:,:,:])
		imgCropL, imgCropR = CroppingImgNew(RGB1_chan[k,:,:,:],RGB2_chan[k,:,:,:],cxL,cxR,cyL,cyR)
		L_GC, L_TT = GetAbsFluoValues(gcamp_chan[k,:,:],tdTom_chan[k,:,:],ContList[0],L_GC,L_TT,k,boolLN)
		R_GC, R_TT = GetAbsFluoValues(gcamp_chan[k,:,:],tdTom_chan[k,:,:],ContList[1],R_GC,R_TT,k,boolRN)
		shapeRGB = np.shape(imgRGB)
		if boolLN == 1 : 
			blackImL = np.zeros(np.shape(imgCropL))
			imgCropL = blackImL
			imgRGBtemp = RGBtemp_nan[k,:,:,:]
			imgRGB = drawOneContour(ContoursD,ret,markers,NewROIList,imgRGBtemp,'R')
		if boolRN == 1 :
			blackImR = np.zeros(np.shape(imgCropR))
			imgCropR = blackImR
			imgRGBtemp = RGBtemp_nan[k,:,:,:]
			imgRGB = drawOneContour(ContoursD,ret,markers,NewROIList,imgRGBtemp,'L')

		CreateAndStoreImages(imgCropR,imgCropL,imgRGB)
		CreateAndStoreTempFluoValues(L_GC, L_TT, R_GC, R_TT,WrongDetection)
	k+=1
	count += 1

	if k==FrameRange[-1]+1:      
	    break

G6_Norm_L, TT_Norm_L = GraphValues(L_GC,L_TT,"L")
G6_Norm_R, TT_Norm_R = GraphValues(R_GC,R_TT,"R")

if not WrongDetection :
    print ("no wrong ROI shape detection")
else : 
    storeWrongRoiDetection()

if UIselect==1:
	os.remove(outDirFrameNo_Usrselected + "FrameToFix_Pause" + '.txt')

if os.path.exists(outDirGC6_UsrCorrected +'L_GC_Temp.txt')==True:
	os.remove(outDirGC6_UsrCorrected +'L_GC_Temp.txt')
	os.remove(outDirGC6_UsrCorrected +'L_TT_Temp.txt')
	os.remove(outDirGC6_UsrCorrected +'R_GC_Temp.txt')
	os.remove(outDirGC6_UsrCorrected +'R_TT_Temp.txt')
	os.remove(outDirGC6_UsrCorrected +'WrongDetection_Temp.txt')
	

os.remove(dataDir + '/registered/' + imgStackFileNameRGB1)
os.remove(dataDir + '/registered/' + imgStackFileNameRGB2)
os.remove(dataDir + '/registered/' + imgStackFileNameRGBtemp)
os.remove(dataDir + '/registered/' + imgStackFileNameRGBNan)


