# -*- coding: utf-8 -*-
"""

Goal:
    This script uses the data from the dictionary created in script P4 and creates all the frames for the final movie.
    
Method:
    The data stored in the dictionary are opened.
    The behavior frames and fluorescence frames are opened and the start and stop indexes of behavior frames are found.
    Puff data is thresholded based on its mode.
    The video frames are created with myFunc() and ready to be used to create the final video with the ffmpeg command.

Note :
	The name of the neuron is written on the top left of the movie frames. If MDN, set boolN as 1, if MAN set boolN as 2, if A1 set boolN as 3.
     
"""

import os, os.path
import cPickle as pickle
from skimage import io
from PIL import Image
Image.Image.tostring = Image.Image.tobytes
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import sys
import re
import numpy as np
from multiprocessing import Pool
import time
from scipy import stats
import matplotlib
import math

dataDir = 'YOUR_PATH_TO_EXPERIMENT_FOLDER'
print dataDir

vidDir = dataDir + '/behavior_imgs'
outDir= dataDir + '/output/'
outDircropROI_UsrCorrected = outDir + 'cropROI_UsrCorrected/'
outDircropROI_auto = outDir + 'cropROI_auto/'
imgStackFileNameR = 'tdTom.tif'
tdTom_stack = io.imread(dataDir + '/registered/' + imgStackFileNameR)

outFigureDirPuff = outDir + 'FramesForVideo/'
if not os.path.exists(outFigureDirPuff):
    os.makedirs(outFigureDirPuff)

Cam_fudge = 1
convFactor = 2*math.pi*5
anglFactor = 360.0

#boolN = 1
#boolN = 2
boolN = 3

def OpenDicData():
	"""
	This function opens the dictionary created in P4 and returns all the data stored in it.
	"""
	if os.path.exists(outDir+"DicDataAnalysisPAB.p"):
		DicData = pickle.load( open( outDir + "DicDataAnalysisPAB.p", "rb" ) )
		frameCntr = DicData['frameCntr']
		L_DR = DicData['L_DR']
		R_DR = DicData['R_DR']
		velForw = DicData['velForw']
		velSide = DicData['velSide']
		velTurn = DicData['velTurn']
		timeSec = DicData['timeSec']
		cam_systime = DicData['cam_systime']
		AddValues = DicData['AddValues']
		puffSampled = DicData['puffSampled']
		vidStartSysTime = AddValues[0]
		vidStopSysTime = AddValues[1]
		stepOnIdx = AddValues[2]

	else :
		print ("File not found - Data not analysed yet - please go to 4th part of data analysis")
		sys.exit(0) 

	return frameCntr, L_DR, R_DR, velForw, velSide, velTurn, timeSec, cam_systime, puffSampled, vidStartSysTime, vidStopSysTime, stepOnIdx

def getBehaviorImg():
	"""
	This function opens the paths to the behavior frames and stores them in a list.
	"""
	listFiles = os.listdir(vidDir)
	vidList = [None] * (len(listFiles))
	for f in os.listdir(vidDir):
	    if f.endswith(".jpg"):
	        frNum = int(re.split(".jpg",re.split("frame",f)[-1])[0])
	        vidList[frNum]=os.path.join(vidDir,f)
	if vidList[-1]== None:
		vidList.pop()

	return vidList

def getCropImg():
	"""
	This function opens the fluorescence frames including the ROIs contours.
	"""
	print(">>>Checking if the images are pre-processed appropriately for making the video...")
	cropROI_stack=[]
	cropROI_stack_temp=[]

	if os.path.exists(outDircropROI_UsrCorrected):
	    for k in range(0,len(tdTom_stack)):
	        try:
	            cropROI_chan_temp = io.imread(outDircropROI_UsrCorrected + "%04d" % k + '.png')
	        except FileNotFoundError:
	            print("!!!The ROI selection have not finished yet. Please finish the ROI selection!!!")
	            sys.exit(0)
	        cropROI_stack_temp.append(cropROI_chan_temp[:70,:,:])
	            
	else:
	    for k in range(0,len(tdTom_stack)):
	        try:
	            cropROI_chan_temp = io.imread(outDircropROI_auto + "%04d" % k + '.png')
	        except FileNotFoundError:
	            print("!!!The ROI selection have not finished yet. Please finish the ROI selection!!!")
	            sys.exit(0)
	        cropROI_stack_temp.append(cropROI_chan_temp[:70,:,:])

	cropROI_stack=np.array(cropROI_stack_temp)

	return cropROI_stack

def defFlowFrameRange(timeSec):
	"""
	This function returns the flow frame range which is the number of datapoint within 10 seconds of videos.
	"""
	xaxisRange=10
	flowFrameRange = xaxisRange*len(timeSec)/(timeSec[-1]-timeSec[0])

	return xaxisRange, flowFrameRange

def vidStartAndStop(Cam_systime, vidStartSysTime, vidStopSysTime):
	"""
	This function finds the start and stop indexes of the behavior frames based on the start and stop time of the video.
	"""
	if vidStartSysTime==Cam_systime[0]:
	    startVidIdx=0
	else:     
	    for i in range(0,len(Cam_systime)):
	        if vidStartSysTime-Cam_systime[i]<0:
	            startVidIdx=i-1 
	            break
     
	if vidStopSysTime==Cam_systime[-1]:
	    stopVidIdx=len(Cam_systime)-1
	else:     
	    for i in range(0,len(Cam_systime)):
	        if vidStopSysTime-Cam_systime[i]<0:
	            stopVidIdx=i-1
	            break

	return startVidIdx, stopVidIdx

def getYAxisMinMax(OpFlow):
	"""
	This function returns the max and min of the optic flow y axis.
	"""
	ylimMin = np.nanmin(OpFlow)
	ylimMax = np.nanmax(OpFlow)
	if abs(ylimMin)<=abs(ylimMax): 
	    ylimMin=-ylimMax
	else:
	    ylimMax=-ylimMin

	return ylimMin, ylimMax

def getYAxisMinMaxGC(LGC,RGC):
	"""
	This function returns the max and min of the fluorescence DR/R*100 y axis.
	"""
	TotGCTT1=max(LGC) 
	TotGCTT2 = max(RGC)
	if TotGCTT1 > TotGCTT2:
		ylimMaxGC=TotGCTT1
	else :
		ylimMaxGC = TotGCTT2
	TotMinGCTT1 = min(LGC)
	TotMinGCTT2 = min(RGC)
	if TotMinGCTT1 < TotMinGCTT2:
		ylimMinGC = TotMinGCTT1
	else : 
		ylimMinGC = TotMinGCTT2

	if 0<ylimMaxGC<50:
		ylimMaxGC = 50
	elif 50<ylimMaxGC<100:
		ylimMaxGC = 100
	elif 100<ylimMaxGC<150:
		ylimMaxGC = 150
	elif 150<ylimMaxGC<200:
		ylimMaxGC = 200
	elif 200<ylimMaxGC<250:
		ylimMaxGC = 250
	elif 250<ylimMaxGC<300:
		ylimMaxGC = 300
	elif 300<ylimMaxGC<350:
		ylimMaxGC = 350
	elif 350<ylimMaxGC<400:
		ylimMaxGC = 400
	elif 400<ylimMaxGC<450:
		ylimMaxGC = 450

	if -100<ylimMinGC<-50:
		ylimMinGC = -100
	elif -50<ylimMinGC<0:
		ylimMinGC = -50
	else : 
		ylimMinGC = 0

	return ylimMinGC, ylimMaxGC


def getXAxisAndRatio(vidStartSysTime, vidStopSysTime, timeSec,cam_systime,startVidIdx,stopVidIdx):
	"""
	This function returns the min and max values of the x axis and the ratio of the number of points in the global time list 
	to the number of points in the behavior time list.
	"""
	vidStopTime = vidStopSysTime - vidStartSysTime
	xlimMin=0
	xlimMax=xlimMin+vidStopTime 
	interval=60
	Ratio_HDseriestoCam=float(len(timeSec))/float(len(cam_systime[startVidIdx:stopVidIdx]))

	return xlimMin, xlimMax, Ratio_HDseriestoCam

def getAlignedData(velForw,velSide,velTurn,flowFrameRange,timeSec,startIdx_HD,stopIdx_HD,xaxisRange,L_GCamP6_tdtom_norm_HD,R_GCamP6_tdtom_norm_HD):
	"""
	This function selects the 10 seconds of data that are going to be plotted on one video frame.
	"""
 	#The begining of data for video
    if len(velForw[0:stopIdx_HD]) < flowFrameRange:
        opflowDataForw = velForw[0:stopIdx_HD]
        opflowDataSide = velSide[0:stopIdx_HD]
        opflowDataTurn = velTurn[0:stopIdx_HD]
        
        GC_Data_L = L_GCamP6_tdtom_norm_HD[0:stopIdx_HD]
        GC_Data_R = R_GCamP6_tdtom_norm_HD[0:stopIdx_HD]
        
        xax = timeSec[0:stopIdx_HD]        
        xaxislimit=[timeSec[startIdx_HD]-xaxisRange/2,timeSec[startIdx_HD]+xaxisRange/2]

    #The end of data for video   
    elif  len(velForw[startIdx_HD:-1]) < flowFrameRange/2:
        opflowDataForw = velForw[startIdx_HD-(stopIdx_HD-startIdx_HD):-1]
        opflowDataSide = velSide[startIdx_HD-(stopIdx_HD-startIdx_HD):-1]
        opflowDataTurn = velTurn[startIdx_HD-(stopIdx_HD-startIdx_HD):-1]
        
        GC_Data_L = L_GCamP6_tdtom_norm_HD[startIdx_HD-(stopIdx_HD-startIdx_HD):-1]
        GC_Data_R = R_GCamP6_tdtom_norm_HD[startIdx_HD-(stopIdx_HD-startIdx_HD):-1]
        
        xax = timeSec[startIdx_HD-(stopIdx_HD-startIdx_HD):-1]
        xaxislimit=[timeSec[startIdx_HD]-xaxisRange/2,timeSec[startIdx_HD]+xaxisRange/2]
    
    else:
        opflowDataForw = velForw[startIdx_HD-(stopIdx_HD-startIdx_HD):stopIdx_HD]
        opflowDataSide = velSide[startIdx_HD-(stopIdx_HD-startIdx_HD):stopIdx_HD]
        opflowDataTurn = velTurn[startIdx_HD-(stopIdx_HD-startIdx_HD):stopIdx_HD]
        
        GC_Data_L = L_GCamP6_tdtom_norm_HD[startIdx_HD-(stopIdx_HD-startIdx_HD):stopIdx_HD]
        GC_Data_R = R_GCamP6_tdtom_norm_HD[startIdx_HD-(stopIdx_HD-startIdx_HD):stopIdx_HD]
        
        xax = timeSec[startIdx_HD-(stopIdx_HD-startIdx_HD):stopIdx_HD]
        xaxislimit=[timeSec[startIdx_HD]-xaxisRange/2,timeSec[startIdx_HD]+xaxisRange/2]

    return opflowDataForw, opflowDataSide, opflowDataTurn, GC_Data_L, GC_Data_R, xax, xaxislimit

def managePuffData(puffSampled,timeSec,stepOnIdx):
	"""
	This function finds the mode of the puff data and substracts it to all the puff data in order to threshold it in the next function.
	"""
	puffTempBegIdx = puffSampled[stepOnIdx:len(timeSec)+stepOnIdx]
	puffSampledRound = [ round(elem, 1) for elem in puffTempBegIdx ] 
	modePuffSampled = stats.mode(puffSampledRound)
	puffDiff = [elem - modePuffSampled[0][0] for elem in puffSampledRound]

	return puffDiff

def ceiledPuff(puffDiff):
	"""
	This function thresholds the puff data to determine whether a puff was given to the fly or not.
	"""
	puffListCopy = puffDiff[:]
	EventIdx = []
	for i in range(len(puffDiff)):
		if puffDiff[i] > 0.1:
			EventIdx.append(i)
	if (not(EventIdx)==True)==True:
		puffListCopy = puffListCopy
	else :
		if EventIdx[0] != 0:
			EventIdx.insert(0,0)
		breakEventIdx = []
		for i in range(len(EventIdx)-1):
			if EventIdx[i+1] != EventIdx[i]+1:
				if len(breakEventIdx)>1:
					if breakEventIdx[-1]!=EventIdx[i]:
						breakEventIdx.append(EventIdx[i])
					if breakEventIdx[-1]!=EventIdx[i+1]:
						breakEventIdx.append(EventIdx[i+1])
				else:
					breakEventIdx.append(EventIdx[i])
					breakEventIdx.append(EventIdx[i+1])
		
		if breakEventIdx[-1]!=EventIdx[-1]:
			breakEventIdx.append(EventIdx[-1])
		if breakEventIdx[-1]!=len(puffDiff):
			breakEventIdx.append(len(puffDiff))

		for i in range(len(breakEventIdx)-1):
			maxVal = max(puffDiff[breakEventIdx[i]:breakEventIdx[i+1]])
			if maxVal > 4.8:
				for j in range(breakEventIdx[i],breakEventIdx[i+1]):
					puffListCopy[j] = 4.9
			elif (3.2 < maxVal < 4.8) == True:
				for j in range(breakEventIdx[i],breakEventIdx[i+1]):
					puffListCopy[j] = 3.3
			elif (1.7 < maxVal < 3.2) == True:
				for j in range(breakEventIdx[i],breakEventIdx[i+1]):
					puffListCopy[j] = 1.8
			elif (0.11 < maxVal < 1.7) == True:
				for j in range(breakEventIdx[i],breakEventIdx[i+1]):
					puffListCopy[j] = 0.2
			else :
				for j in range(breakEventIdx[i],breakEventIdx[i+1]):
					puffListCopy[j] = 0

	return puffListCopy

# **************** MAIN ***************

def myFunc(vidIdx):

	fig = plt.figure(facecolor='black')
	fig.subplots_adjust(hspace=0.02)

	# Image behavior
	ax1 = plt.subplot2grid((500,600), (201,0), colspan=301, rowspan=299)
	ax1.yaxis.label.set_color('white')
	plt.ylabel("vidIdx=",fontsize=10)

	# Image crop ROI
	ax3 = plt.subplot2grid((500,600), (0,0), colspan=301, rowspan=151)
	ax3.xaxis.label.set_color('white')
	ax3.yaxis.label.set_color('white')
	ax3.set_xticks([])
	ax3.set_yticks([])
	ax3.get_xaxis().set_label_coords(0.85,-0.05)
	if boolN == 1 : 
		ax3.text(0, 0, 'MDN', fontsize=12,color='white')
	elif boolN == 2 : 
		ax3.text(0, 0, 'dMAN', fontsize=12,color='white')
	elif boolN == 3 : 
		ax3.text(0, 0, 'A1', fontsize=12,color='white')
	ax3.set_xlabel('            Left                              Right', fontsize=10,horizontalalignment='right')
	ax3.text(23, 86, 'neuron', fontsize=10,color='white')
	ax3.text(101, 86, 'neuron', fontsize=10,color='white')

	# VForward
	ax4 = plt.subplot2grid((500,600), (205,421), colspan=179, rowspan=90)
	ax4.spines['bottom'].set_visible(False)
	ax4.spines['top'].set_visible(False)
	ax4.spines['right'].set_visible(False)
	ax4.spines['left'].set_color('white')
	ax4.get_xaxis().set_visible(False)
	ax4.yaxis.label.set_color('white')
	ax4.tick_params(axis='x', colors='white')
	ax4.tick_params(axis='y', colors='white')    
	ax4.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
	ax4.axhline(0, linestyle='dashed',color='gray',linewidth=1.5)
	ax4.set_ylim(ylimMinAP, ylimMaxAP)
	ax4.tick_params(axis='y', labelsize=10)
	ax4.set_ylabel(unicode('(mm s$^{-1}$)'),fontsize=7,rotation=90,horizontalalignment='center')
	tempListCur4 = [ylimMinAP,0,ylimMaxAP]
	newYAxis4 = np.array(tempListCur4)
	ax4.yaxis.set_ticks(newYAxis4)

	vidImage = Image.open(vidList[vidIdx-Cam_fudge])
	startIdx_HD = int((vidIdx-startVidIdx)*Ratio_HDseriestoCam) 
	stopIdx_HD = int(startIdx_HD + flowFrameRange/2)
	imgFrame = int(np.floor(frameCntr[stepOnIdx+startIdx_HD]/3))-1
	opflowDataForw, opflowDataSide, opflowDataTurn, GC_Data_L, GC_Data_R, xax, xaxislimit = getAlignedData(velForw,velSide,velTurn,flowFrameRange,timeSec,startIdx_HD,stopIdx_HD,xAxisRange,L_DR,R_DR)

	# Image behavior
	normA = matplotlib.colors.Normalize(vmin=15.0,vmax=255.0)
	ax1.get_xaxis().set_visible(False)
	ax1.get_yaxis().set_visible(True)
	ax1.tick_params(axis=u'both', which=u'both',length=0)
	ax1.imshow(vidImage, cmap = 'gray', origin='upper',norm=normA)
	ax1.set_ylabel("vidIdx="+str(vidIdx-Cam_fudge),fontsize=10)
	ax1.get_yaxis().set_label_coords(-0.1,0.5)
	ax1.text(62, 505, ("%.2f" % timeSec[startIdx_HD] + 's'), ha="center", va="center", rotation=0,size=10, color='white')
	if puffThresh[startIdx_HD] == 4.9:
		ax1.text(570, -30, 'Puff', color='white',fontweight='bold',bbox=dict(facecolor='tomato', edgecolor='tomato', boxstyle="square,pad=0.2"))
	if puffThresh[startIdx_HD] == 3.3:
		ax1.text(570, -30, 'Puff', color='white',fontweight='bold',bbox=dict(facecolor='red', edgecolor='red', boxstyle="square,pad=0.2"))
	if puffThresh[startIdx_HD] == 1.8:
		ax1.text(570, -30, 'Puff', color='white',bbox=dict(facecolor='firebrick', edgecolor='firebrick', boxstyle="square,pad=0.2"))
	if puffThresh[startIdx_HD] == 0.2:
		ax1.text(570, -30, 'Puff', color='white',bbox=dict(facecolor='darkred', edgecolor='darkred', boxstyle="square,pad=0.2"))

	# Image crop ROI
	ax3.imshow(cropROI_stack[imgFrame,:,:,:], cmap = 'gray', origin='upper')
	ax3.set_ylabel("imgframe="+str(imgFrame),fontsize=10)
	ax3.get_yaxis().set_label_coords(-0.13,0.5)

	# VForward
	ax4.axvline(timeSec[startIdx_HD], linestyle='dashed',color='white',linewidth=1.5)
	ax4.plot(xax,opflowDataForw, label = "AP", color='r',linewidth=1.4)
	ax4.set_xlim(xaxislimit[0], xaxislimit[1])
	ax4.get_yaxis().set_label_coords(-0.3,0.51)

	ax7 = plt.subplot2grid((500,600), (208,355), colspan=1, rowspan=90)
	ax7.spines['bottom'].set_visible(False)
	ax7.spines['left'].set_visible(False)
	ax7.spines['right'].set_visible(False)
	ax7.get_yaxis().set_label_coords(-5,0.21)
	plt.ylabel(r'$\mathrm{\mathsf{v_{forward}}}$',fontsize=15,color='r',rotation=90,horizontalalignment='left')
	ax7.set_xticks([])
	ax7.set_yticks([])

	# VSide
	ax5 = plt.subplot2grid((500,600), (313,421), colspan=179, rowspan=90)
	ax5.spines['bottom'].set_visible(False)
	ax5.spines['top'].set_visible(False)
	ax5.spines['right'].set_visible(False)
	ax5.spines['left'].set_color('white')
	ax5.get_xaxis().set_visible(False)
	ax5.yaxis.label.set_color('white')
	ax5.get_yaxis().set_label_coords(-0.3,0.51)
	ax5.tick_params(axis='x', colors='white')
	ax5.tick_params(axis='y', colors='white') 
	ax5.tick_params(axis='y', labelsize=10)   
	ax5.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
	tempListCur = [ylimMinML,0,ylimMaxML]
	newYAxis = np.array(tempListCur)
	ax5.yaxis.set_ticks(newYAxis)
	plt.axhline(0, linestyle='dashed',color='gray',linewidth=1.5)
	plt.axvline(timeSec[startIdx_HD], linestyle='dashed',color='white',linewidth=1.5)
	plt.plot(xax,opflowDataSide, label = "ML", color='c',linewidth=1.4)
	plt.ylim(ylimMinML, ylimMaxML)
	plt.xlim(xaxislimit[0], xaxislimit[1])
	plt.ylabel(unicode('(mm s$^{-1}$)'), fontsize=7,rotation=90,horizontalalignment='center')

	ax8 = plt.subplot2grid((500,600), (315,365), colspan=1, rowspan=90)
	ax8.spines['bottom'].set_visible(False)
	ax8.spines['left'].set_visible(False)
	ax8.spines['right'].set_visible(False)
	ax8.get_yaxis().set_label_coords(-16,0.32)
	plt.ylabel(r'$\mathrm{\mathsf{v_{side}}}$', fontsize=15,color='c',rotation=90,horizontalalignment='left')
	ax8.set_xticks([])
	ax8.set_yticks([])

	# VRotation
	ax6 = plt.subplot2grid((500,600), (420,421), colspan=179, rowspan=90)
	ax6.spines['bottom'].set_color('white')
	ax6.spines['top'].set_visible(False)
	ax6.spines['right'].set_visible(False)
	ax6.spines['left'].set_color('white')
	ax6.xaxis.label.set_color('white')
	ax6.yaxis.label.set_color('white')
	ax6.get_yaxis().set_label_coords(-0.3,0.51)
	ax6.tick_params(axis='x', colors='white', top='off')
	ax6.tick_params(axis='y', colors='white')  
	ax6.tick_params(axis='y', labelsize=10)
	ax6.yaxis.set_major_formatter(FormatStrFormatter('%0.1f')) 
	tempListCur6 = [ylimMinYaw,0,ylimMaxYaw]
	newYAxis6 = np.array(tempListCur6)
	ax6.yaxis.set_ticks(newYAxis6)
	plt.axhline(0, linestyle='dashed',color='gray',linewidth=1.5)
	plt.axvline(timeSec[startIdx_HD], linestyle='dashed',color='white',linewidth=1.5)
	plt.plot(xax,opflowDataTurn, label = "Yaw", color='g',linewidth=1.4)
	plt.ylim(ylimMinYaw, ylimMaxYaw) #set limit of Y axis
	plt.xlim(xaxislimit[0], xaxislimit[1])
	ax6.tick_params(axis='x', labelsize=10)
	plt.ylabel(unicode('(deg. s$^{-1}$)'), fontsize=7,rotation=90,horizontalalignment='center')
	plt.xlabel(unicode('Time (s)'),fontsize=10)

	ax9 = plt.subplot2grid((500,600), (420,345), colspan=1, rowspan=90)
	ax9.spines['bottom'].set_visible(False)
	ax9.spines['left'].set_visible(False)
	ax9.spines['right'].set_visible(False)
	ax9.get_yaxis().set_label_coords(8,0.12)
	plt.ylabel(r'$\mathrm{\mathsf{v_{rotation}}}$', fontsize=15,color='g',rotation=90,horizontalalignment='left')
	ax9.set_xticks([])
	ax9.set_yticks([])

	# Left Fluorescence
	ax10 = plt.subplot2grid((500,600), (102,421), colspan=179, rowspan=90)
	ax10.spines['bottom'].set_color('white')
	ax10.spines['top'].set_visible(False)
	ax10.spines['right'].set_visible(False)
	ax10.spines['left'].set_color('white')
	ax10.xaxis.label.set_color('white')
	ax10.get_xaxis().set_visible(False)
	ax10.spines['bottom'].set_visible(False)
	ax10.yaxis.label.set_color('white')
	ax10.yaxis.set_label_coords(-0.4,0.12)
	ax10.tick_params(axis='x', colors='white')
	ax10.tick_params(axis='y', colors='white')    
	ax10.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
	ax10.tick_params(axis='y', labelsize=10)
	currentYax10 = np.arange(0,ylimMaxGC,100)
	tempListCur10 = list(currentYax10)
	if ylimMaxGC%100==0:
		tempListCur10.append(ylimMaxGC)
	newYAxis10 = np.array(tempListCur10)
	ax10.yaxis.set_ticks(newYAxis10)
	plt.axhline(0, linestyle='dashed',color='gray',linewidth=1.5)
	plt.axvline(timeSec[startIdx_HD], linestyle='dashed',color='white',linewidth=1.5)
	plt.plot(xax, GC_Data_L, label = "L", color='y',linewidth=1.4)
	plt.ylim(ylimMinGC, ylimMaxGC) 
	plt.xlim(xaxislimit[0], xaxislimit[1])
	plt.ylabel(unicode('Left\n$\Delta$R/R\n(%)'), fontsize=10,rotation=0,horizontalalignment='center')

	# Right Fluorescence
	ax12 = plt.subplot2grid((500,600), (0,421), colspan=179, rowspan=90)
	ax12.spines['bottom'].set_color('white')
	ax12.spines['top'].set_visible(False)
	ax12.spines['right'].set_visible(False)
	ax12.spines['left'].set_color('white')
	ax12.xaxis.label.set_color('white')
	ax12.get_xaxis().set_visible(False)
	ax12.spines['bottom'].set_visible(False)
	ax12.yaxis.label.set_color('white')
	ax12.yaxis.set_label_coords(-0.4,0.12)
	ax12.tick_params(axis='x', colors='white')
	ax12.tick_params(axis='y', colors='white')    
	ax12.yaxis.set_major_formatter(FormatStrFormatter('%.0f')) 
	ax12.tick_params(axis='y', labelsize=10)
	ax12.yaxis.set_ticks(newYAxis10)
	plt.axhline(0, linestyle='dashed',color='gray',linewidth=1.5)
	plt.axvline(timeSec[startIdx_HD], linestyle='dashed',color='white',linewidth=1.5)
	plt.plot(xax, GC_Data_R, label = "R", color='y',linewidth=1.4)
	plt.ylim(ylimMinGC, ylimMaxGC)
	plt.xlim(xaxislimit[0], xaxislimit[1])
	plt.ylabel(unicode('Right\n$\Delta$R/R\n(%)'), fontsize=10,rotation=0,horizontalalignment='center')

	plt.savefig(unicode(outFigureDirPuff + "VidFrame" + "%04d" % vidIdx + '.png'), facecolor=fig.get_facecolor(), edgecolor='none', transparent=True) 
	
	plt.close(fig)

if __name__ == '__main__':
	frameCntr, L_DR, R_DR, velForw, velSide, velTurn, timeSec, cam_systime, puffSampled, vidStartSysTime, vidStopSysTime, stepOnIdx = OpenDicData()
	velForw = velForw *convFactor
	velSide = velSide *convFactor
	velTurn = velTurn *anglFactor
	vidList = getBehaviorImg()
	cropROI_stack = getCropImg()
	puffDiff = managePuffData(puffSampled,timeSec,stepOnIdx)
	puffThresh = ceiledPuff(puffDiff)
	xAxisRange, flowFrameRange = defFlowFrameRange(timeSec)
	startVidIdx, stopVidIdx = vidStartAndStop(cam_systime, vidStartSysTime, vidStopSysTime)

	ylimMinAP, ylimMaxAP = getYAxisMinMax(velForw)
	ylimMinML, ylimMaxML = getYAxisMinMax(velSide)
	ylimMinYaw, ylimMaxYaw = getYAxisMinMax(velTurn)
	ylimMinGC, ylimMaxGC = getYAxisMinMaxGC(L_DR,R_DR)
	xlimMin, xlimMax, Ratio_HDseriestoCam = getXAxisAndRatio(vidStartSysTime, vidStopSysTime, timeSec,cam_systime,startVidIdx,stopVidIdx)
	ylimMinGC = -5

	pool = Pool()

	t1 = time.time()
	if Cam_fudge < 0 :
		pool.map(myFunc, range(startVidIdx,stopVidIdx+Cam_fudge))
	else :
		pool.map(myFunc, range(startVidIdx,stopVidIdx))
	
	t2 = time.time()
	print(t2-t1)

