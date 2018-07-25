# -*- coding: utf-8 -*-
"""

Goal:
    This script finds 3 relevant events linked to neuronal activity for 3 flies and presents them on a summary movie.
    
Method:
    The data stored in the dictionnary are opened.
    9 events are selected for 3 flies within all the experiments and the behavior frames for those 9 events are loaded.
    Frames to create the summary movie with the 9 events aligned are created. 

Note :
	This script was written for A1 left neuron related events, A1 right neuron related events, MDN and MAN. 
	The user needs to set the SelectChoice value to 1 to compute the movie for A1 neuron, to 2 for MAN and to 3 for MDN.
	If A1 neurons are selected, the user needs to set the LeftChoice value to 1 for events related to left neuron or to set LeftChoice to 0 
	for events related to right neuron. 
         
"""

import os, os.path
import sys
import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import pandas as pd
import itertools
import re
from PIL import Image
Image.Image.tostring = Image.Image.tobytes
from multiprocessing import Pool
from matplotlib.ticker import FormatStrFormatter
import matplotlib


#TO SET 
LeftChoice = 1
#LeftChoice = 0
#SelectChoice = 1
#SelectChoice = 2
SelectChoice = 3

#Paths to store the final movie.
outFigDir = 'YOUR_PATH_TO'
if SelectChoice == 1:
	if LeftChoice == 1:
		outDirTest = outFigDir+'Movie-A1-Left/'
	if LeftChoice == 0:
		outDirTest = outFigDir+'Movie-A1-Right/'
elif SelectChoice == 2:
	outDirTest = outFigDir+'Movie-MAN/'
elif SelectChoice == 3:
	outDirTest = outFigDir+'Movie-MDN/'

#Global variable
window = 10
windowVideo = 3
timeWindowA1 = 0.125
windowOver = 2
MinOverB = 1
MinOverA = 2

if SelectChoice == 1 : 

	Percentile = 90

	CamFudgeA13 = [-16,-1,17,10,-5,4,27,8,11,16] 
	CamFudgeA14 = [14,9,14,12,0,11,10,18]
	CamFudgeA15 = [7,19,17,20,27,12,17,9,3]
	CamFudge = []
	CamFudge.append(CamFudgeA13)
	CamFudge.append(CamFudgeA14)
	CamFudge.append(CamFudgeA15)

	ListDirA13 = []
	ListDirA13.append("YOUR_PATH_TO/A1-3/fly1_S47_tdTom-GC6s_fly3_004")
	ListDirA13.append("YOUR_PATH_TO/A1-3/fly1_S47_tdTom-GC6s_fly3_005")
	ListDirA13.append("YOUR_PATH_TO/A1-3/fly1_S47_tdTom-GC6s_fly3_006")
	ListDirA13.append("YOUR_PATH_TO/A1-3/fly1_S47_tdTom-GC6s_fly3_008")
	ListDirA13.append("YOUR_PATH_TO/A1-3/fly1_S47_tdTom-GC6s_fly3_009")
	ListDirA13.append("YOUR_PATH_TO/A1-3/fly1_S47_tdTom-GC6s_fly3_010")
	ListDirA13.append("YOUR_PATH_TO/A1-3/fly1_S47_tdTom-GC6s_fly3_011")
	ListDirA13.append("YOUR_PATH_TO/A1-3/fly1_S47_tdTom-GC6s_fly3_012")
	ListDirA13.append("YOUR_PATH_TO/A1-3/fly1_S47_tdTom-GC6s_fly3_013")
	ListDirA13.append("YOUR_PATH_TO/A1-3/fly1_S47_tdTom-GC6s_fly3_014")

	ListDirA14 = []
	ListDirA14.append("YOUR_PATH_TO/A1-4/imaging_S47_fly1_data_001")
	ListDirA14.append("YOUR_PATH_TO/A1-4/imaging_S47_fly1_data_002")
	ListDirA14.append("YOUR_PATH_TO/A1-4/imaging_S47_fly1_data_003")
	ListDirA14.append("YOUR_PATH_TO/A1-4/imaging_S47_fly1_data_004")
	ListDirA14.append("YOUR_PATH_TO/A1-4/imaging_S47_fly1_data_005")
	ListDirA14.append("YOUR_PATH_TO/A1-4/imaging_S47_fly1_data_006")
	ListDirA14.append("YOUR_PATH_TO/A1-4/imaging_S47_fly1_data_007")
	ListDirA14.append("YOUR_PATH_TO/A1-4/imaging_S47_fly1_data_009")

	ListDirA15 = []
	ListDirA15.append("YOUR_PATH_TO/A1-5/imaging_S47_fly3_data_002")
	ListDirA15.append("YOUR_PATH_TO/A1-5/imaging_S47_fly3_data_004")
	ListDirA15.append("YOUR_PATH_TO/A1-5/imaging_S47_fly3_data_006")
	ListDirA15.append("YOUR_PATH_TO/A1-5/imaging_S47_fly3_data_007")
	ListDirA15.append("YOUR_PATH_TO/A1-5/imaging_S47_fly3_data_008")
	ListDirA15.append("YOUR_PATH_TO/A1-5/imaging_S47_fly3_data_009")
	ListDirA15.append("YOUR_PATH_TO/A1-5/imaging_S47_fly3_data_011")
	ListDirA15.append("YOUR_PATH_TO/A1-5/imaging_S47_fly3_data_012")
	ListDirA15.append("YOUR_PATH_TO/A1-5/imaging_S47_fly3_data_016")

	ListDir = []
	ListDir.append(ListDirA13)
	ListDir.append(ListDirA14)
	ListDir.append(ListDirA15)

elif SelectChoice == 2 : 

	Percentile = 97.5

	CamFudgeMAN2 = [20,11,18,15,17,19,13,15,6,15]
	CamFudgeMAN4 = [4,10,10,3,5,5,-2,0,38,35]
	CamFudgeMAN5 = [12,-6,4,12,31]
	CamFudge = []
	CamFudge.append(CamFudgeMAN2)
	CamFudge.append(CamFudgeMAN4)
	CamFudge.append(CamFudgeMAN5)

	ListDirMAN2 = []
	ListDirMAN2.append("YOUR_PATH_TO/MAN-2/image_MAN1_tdTom-GC6s_fly2_014")
	ListDirMAN2.append("YOUR_PATH_TO/MAN-2/image_MAN1_tdTom-GC6s_fly2_015")
	ListDirMAN2.append("YOUR_PATH_TO/MAN-2/image_MAN1_tdTom-GC6s_fly2_016")
	ListDirMAN2.append("YOUR_PATH_TO/MAN-2/image_MAN1_tdTom-GC6s_fly2_017")
	ListDirMAN2.append("YOUR_PATH_TO/MAN-2/image_MAN1_tdTom-GC6s_fly2_018")
	ListDirMAN2.append("YOUR_PATH_TO/MAN-2/image_MAN1_tdTom-GC6s_fly2_019")
	ListDirMAN2.append("YOUR_PATH_TO/MAN-2/image_MAN1_tdTom-GC6s_fly2_020")
	ListDirMAN2.append("YOUR_PATH_TO/MAN-2/image_MAN1_tdTom-GC6s_fly2_021")
	ListDirMAN2.append("YOUR_PATH_TO/MAN-2/image_MAN1_tdTom-GC6s_fly2_022")
	ListDirMAN2.append("YOUR_PATH_TO/MAN-2/image_MAN1_tdTom-GC6s_fly2_023")

	ListDirMAN4 = []
	ListDirMAN4.append("YOUR_PATH_TO/MAN-4/imaging_MAN1_fly3_003")
	ListDirMAN4.append("YOUR_PATH_TO/MAN-4/imaging_MAN1_fly3_004")
	ListDirMAN4.append("YOUR_PATH_TO/MAN-4/imaging_MAN1_fly3_005")
	ListDirMAN4.append("YOUR_PATH_TO/MAN-4/imaging_MAN1_fly3_006")
	ListDirMAN4.append("YOUR_PATH_TO/MAN-4/imaging_MAN1_fly3_007")
	ListDirMAN4.append("YOUR_PATH_TO/MAN-4/imaging_MAN1_fly3_008")
	ListDirMAN4.append("YOUR_PATH_TO/MAN-4/imaging_MAN1_fly3_009")
	ListDirMAN4.append("YOUR_PATH_TO/MAN-4/imaging_MAN1_fly3_010")
	ListDirMAN4.append("YOUR_PATH_TO/MAN-4/imaging_MAN1_fly3_011")
	ListDirMAN4.append("YOUR_PATH_TO/MAN-4/imaging_MAN1_fly3_013")

	ListDirMAN5 = []
	ListDirMAN5.append("YOUR_PATH_TO/MAN-5/imaging_MAN1_fly4_009")
	ListDirMAN5.append("YOUR_PATH_TO/MAN-5/imaging_MAN1_fly4_010")
	ListDirMAN5.append("YOUR_PATH_TO/MAN-5/imaging_MAN1_fly4_011")
	ListDirMAN5.append("YOUR_PATH_TO/MAN-5/imaging_MAN1_fly4_012")
	ListDirMAN5.append("YOUR_PATH_TO/MAN-5/imaging_MAN1_fly4_013")

	ListDir = []
	ListDir.append(ListDirMAN2)
	ListDir.append(ListDirMAN4)
	ListDir.append(ListDirMAN5)

elif SelectChoice == 3 : 

	Percentile = 97.5

	CamFudgeMDN1 = [11,4,24,10,17,12,7,12,20,12]
	CamFudgeMDN2 = [10,17,7,5,3,17,16,17,13,20]
	CamFudgeMDN3 = [17,11,10,17,6,16,14,3,-3,8]
	CamFudge = []
	CamFudge.append(CamFudgeMDN1)
	CamFudge.append(CamFudgeMDN2)
	CamFudge.append(CamFudgeMDN3)

	ListDirMDN1 = []
	ListDirMDN1.append("YOUR_PATH_TO/MDN-1/image_MDN1_tdTom-GC6s_003")
	ListDirMDN1.append("YOUR_PATH_TO/MDN-1/image_MDN1_tdTom-GC6s_005")
	ListDirMDN1.append("YOUR_PATH_TO/MDN-1/image_MDN1_tdTom-GC6s_007")
	ListDirMDN1.append("YOUR_PATH_TO/MDN-1/image_MDN1_tdTom-GC6s_008")
	ListDirMDN1.append("YOUR_PATH_TO/MDN-1/image_MDN1_tdTom-GC6s_009")
	ListDirMDN1.append("YOUR_PATH_TO/MDN-1/image_MDN1_tdTom-GC6s_010")
	ListDirMDN1.append("YOUR_PATH_TO/MDN-1/image_MDN1_tdTom-GC6s_011")
	ListDirMDN1.append("YOUR_PATH_TO/MDN-1/image_MDN1_tdTom-GC6s_012")
	ListDirMDN1.append("YOUR_PATH_TO/MDN-1/image_MDN1_tdTom-GC6s_013")
	ListDirMDN1.append("YOUR_PATH_TO/MDN-1/image_MDN1_tdTom-GC6s_014")

	ListDirMDN2 = []
	ListDirMDN2.append("YOUR_PATH_TO/MDN-2/170515_MDN1_tdTom-GC6s_fly1_006")
	ListDirMDN2.append("YOUR_PATH_TO/MDN-2/170515_MDN1_tdTom-GC6s_fly1_008")
	ListDirMDN2.append("YOUR_PATH_TO/MDN-2/170515_MDN1_tdTom-GC6s_fly1_009")
	ListDirMDN2.append("YOUR_PATH_TO/MDN-2/170515_MDN1_tdTom-GC6s_fly1_010")
	ListDirMDN2.append("YOUR_PATH_TO/MDN-2/170515_MDN1_tdTom-GC6s_fly1_011")
	ListDirMDN2.append("YOUR_PATH_TO/MDN-2/170515_MDN1_tdTom-GC6s_fly1_012")
	ListDirMDN2.append("YOUR_PATH_TO/MDN-2/170515_MDN1_tdTom-GC6s_fly1_013")
	ListDirMDN2.append("YOUR_PATH_TO/MDN-2/170515_MDN1_tdTom-GC6s_fly1_014")
	ListDirMDN2.append("YOUR_PATH_TO/MDN-2/170515_MDN1_tdTom-GC6s_fly1_015")
	ListDirMDN2.append("YOUR_PATH_TO/MDN-2/170515_MDN1_tdTom-GC6s_fly1_016")

	ListDirMDN3 = []
	ListDirMDN3.append("YOUR_PATH_TO/MDN-3/imaging_MDN1_tdTom-GC6s_fly2_006")
	ListDirMDN3.append("YOUR_PATH_TO/MDN-3/imaging_MDN1_tdTom-GC6s_fly2_007")
	ListDirMDN3.append("YOUR_PATH_TO/MDN-3/imaging_MDN1_tdTom-GC6s_fly2_008")
	ListDirMDN3.append("YOUR_PATH_TO/MDN-3/imaging_MDN1_tdTom-GC6s_fly2_009")
	ListDirMDN3.append("YOUR_PATH_TO/MDN-3/imaging_MDN1_tdTom-GC6s_fly2_010")
	ListDirMDN3.append("YOUR_PATH_TO/MDN-3/imaging_MDN1_tdTom-GC6s_fly2_011")
	ListDirMDN3.append("YOUR_PATH_TO/MDN-3/imaging_MDN1_tdTom-GC6s_fly2_012")
	ListDirMDN3.append("YOUR_PATH_TO/MDN-3/imaging_MDN1_tdTom-GC6s_fly2_013")
	ListDirMDN3.append("YOUR_PATH_TO/MDN-3/imaging_MDN1_tdTom-GC6s_fly2_014")
	ListDirMDN3.append("YOUR_PATH_TO/MDN-3/imaging_MDN1_tdTom-GC6s_fly2_016")

	ListDir = []
	ListDir.append(ListDirMDN1)
	ListDir.append(ListDirMDN2)
	ListDir.append(ListDirMDN3)


if not os.path.exists(outDirTest):
    os.makedirs(outDirTest)

def OpenDicData(outDir):
	"""
	This function opens the DR/R%, optic flow data and starting and stopping point of the video in P4 stored in the dictionnary. 
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

		vidStopTime = vidStopSysTime - vidStartSysTime

	else :
		print ("File not found - Data not analysed yet - please go to 4th part of data analysis")
		sys.exit(0) 

	return L_DR, R_DR, velForw, velSide, velTurn, timeSec, vidStopTime, vidStartSysTime, vidStopSysTime, cam_systime


def getDeriv(L_DR,timeSec):
	"""
	This function computes the first order derivative of the DR/R% trace. 
	"""
	derivL = []
	interv = 1
	for i in range(interv,len(L_DR)):
		y1 = L_DR[i-1]
		y2 = L_DR[i]
		x1 = timeSec[i-1]
		x2 = timeSec[i]
		deriv = (y2-y1)/(x2-x1)
		derivL.append(deriv)
	derivResL = np.zeros(len(derivL)+interv)
	derivResL[0:interv]=np.nan
	derivResL[interv:]=derivL[:]
	return derivResL, interv

def getIndxThresh(derivTot,thresh):
	"""
	This function finds the indexes of derivative values that are crossing the threshold values.
	"""
	IndxThreshTot = []
	for deriv in derivTot:
		IndxThresh = []
		for i in range(len(deriv)):
			if deriv[i] > thresh:
				if deriv[i+1]>thresh:
					if deriv[i-1]<thresh:
						IndxThresh.append(i)
		IndxThreshTot.append(IndxThresh)
	return IndxThreshTot

def getEventIdx(idxThreshTot,derivGlob,timeFactGlob,window,timeSecAll):
	"""
	This function receives the indexes detected from getIndxThresh(). It then finds the indexes of the 
	0 crossing of the derivative that happened before the derivative crossed the threshold value.
	"""
	eventIdxTotBefore = []
	eventIdxTotAfter = []

	for idxThresh,deriv,timeFact,timeSec in itertools.izip(idxThreshTot,derivGlob,timeFactGlob,timeSecAll):
		timeFrame = int(window*timeFact)
		timeFrameEnd = int(len(timeSec)-timeFrame)
		timeFrame2Sec = int(2*timeFact)
		eventIdx = []
		eventIdxAfter = []
		for j in idxThresh:
			for i in range(len(deriv[:j])):
				if deriv[j-i] < 0:
					if deriv[j-i+1]>0:
						if ((j-i+1) in eventIdx) == False:
							if (j-i+1)>timeFrame:
								if (j-i+1)<timeFrameEnd:
									eventIdx.append(j-i+1)
						break
		for s in eventIdx:
			for k in range(len(deriv[s:])):
				if deriv[s+k] > 0:
					if deriv[s+k+1]<0:
						if k<timeFrame2Sec:
							if ((s+k) in eventIdxAfter) == False:
								if (s+k)>timeFrame:
									eventIdxAfter.append(s+k)
						else :
							newK = s + timeFrame2Sec -1
							eventIdxAfter.append(newK)
						break

		eventIdxTotBefore.append(eventIdx)
		eventIdxTotAfter.append(eventIdxAfter)
	
	return eventIdxTotBefore, eventIdxTotAfter

def findStartEventIdxA1(PosLeftList,eventIdxTotBeforeL):
	"""
	This function returns the idx of the events detected.
	"""
	TotIdxList = []
	for i in range(len(PosLeftList)):
		IdxListExp = []
		if len(PosLeftList[i])!=0:
			for j in range(len(PosLeftList[i])):
				IdxListExp.append(eventIdxTotBeforeL[i][PosLeftList[i][j]])
		TotIdxList.append(IdxListExp)

	return TotIdxList

def vidStartAndStop(Cam_systime, vidStartSysTime, vidStopSysTime):
	"""
	This function returns the start and stop indexes of the summary movie computed in P5.
	"""
	if vidStartSysTime==Cam_systime[0]:
	    startVidIdx=0
	else:     
	    for i in range(0,len(Cam_systime)):
	        if vidStartSysTime-Cam_systime[i]<0:
	            startVidIdx=i-1-1 
	            break
     
	if vidStopSysTime==Cam_systime[-1]:
	    stopVidIdx=len(Cam_systime)-1-1
	else:     
	    for i in range(0,len(Cam_systime)):
	        if vidStopSysTime-Cam_systime[i]<0:
	            stopVidIdx=i-1-1
	            break

	return startVidIdx, stopVidIdx

def FindBehaviourIdx(TotIdxListStartEvent,FramePerSecondBehaviour,timeFactGlob,AllVidStartIdx,CamFudge):
	"""
	This function receives the event detected indexes and returns the corresponding indexes of the behavior frames that will be used as the 0 second time point in the final video.  
	"""
	BehaviourIdxList = []
	for i in range(len(TotIdxListStartEvent)):
		if len(TotIdxListStartEvent[i])!=0:
			timeFact = timeFactGlob[i]
			timeFactBehaviour = FramePerSecondBehaviour[i]
			for j in TotIdxListStartEvent[i]:
				SecToMatch = j/timeFact
				IdxBehav = int(SecToMatch*timeFactBehaviour) + AllVidStartIdx[i] - CamFudge[i]
				BehaviourIdxList.append([i,IdxBehav])

	return BehaviourIdxList

def getBehaviorImg(vidDir):
	"""
	This function imports the paths of the behavior frames that will be shown in the final movie.
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


def A1EventSeparated(eventIdxTotL,eventIdxTotR,timeFactGlob):
	"""
	This function receives the events detected for left and right A1 neurons and separates them within left neuron only related events, right neuron only related events 
	and left and right neurons events happening within a time window of 0.25 second. 
	"""
	LFromLAREventGlob = []
	RFromLAREventGlob = []
	LEventOnlyGlob = []
	REventOnlyGlob = []
	for i in range(len(eventIdxTotL)):
		LFromLAREventExp = []
		RFromLAREventExp = []
		timeFactExp = int(timeWindowA1*timeFactGlob[i])
		for j in eventIdxTotL[i]:
			IdxListToMatch = list(np.arange(j-timeFactExp,j+timeFactExp,1))
			LARDetected = list(set(eventIdxTotR[i]).intersection(set(IdxListToMatch)))
			if len(LARDetected)!=0:
				LFromLAREventExp.append(j)
				RFromLAREventExp+=LARDetected
		LEventExp = list(set(eventIdxTotL[i])-set(LFromLAREventExp))
		REventExp = list(set(eventIdxTotR[i])-set(RFromLAREventExp))
		LEventOnlyGlob.append(LEventExp)
		REventOnlyGlob.append(REventExp)
		LFromLAREventGlob.append(LFromLAREventExp)
		RFromLAREventGlob.append(RFromLAREventExp)

	return LFromLAREventGlob, RFromLAREventGlob ,LEventOnlyGlob, REventOnlyGlob 

def getEventAfter(LEventOnlyGlob,eventIdxTotBeforeL,eventIdxTotAfterL):
	"""
	This function sorts the "after events detection". After events are the time point related to the first 0 crossing of the derivative after the event time point occured.
	"""
	EventAfterSortedList = []
	for i in range(len(LEventOnlyGlob)):
		AfterListExp = []
		for j in LEventOnlyGlob[i]:
			idxEventBefore = eventIdxTotBeforeL[i].index(j)
			AfterListExp.append(eventIdxTotAfterL[i][idxEventBefore])
		EventAfterSortedList.append(AfterListExp)

	return EventAfterSortedList

def getOpTrace(TotIdxListStartEvent,VelForwGlob,timeFactGlob):
	"""
	This function returns one of the selected optic flow trace within a 3 seconds time window around the event selected.
	"""
	APExp = []
	for i in range(len(TotIdxListStartEvent)):
		if len(TotIdxListStartEvent[i])!=0:
			timeFact = int(timeFactGlob[i])
			for j in TotIdxListStartEvent[i]:
				APTempList = list(VelForwGlob[i][j-timeFact:j+2*timeFact])
				APExp.append(APTempList)

	return APExp

def aligneAPTraces(APTraceGlob):
	"""
	This function detects the longer optic flow traces of the 9 events and interpolate the other optic flow traces to this length.
	"""
	APTrace = []
	APTraceStore = []
	allLen = []
	for i in range(len(APTraceGlob)):
		for j in APTraceGlob[i]:
			APTraceStore.append(j)
			allLen.append(len(j))
	maxLen = max(allLen)

	for i in range(len(APTraceStore)):
		LinspaceTemp = np.linspace(0,len(APTraceStore[i])-1,len(APTraceStore[i]))
		LinspaceGoodLength=np.linspace(0,len(APTraceStore[i])-1,maxLen)
		InterpolateList=np.interp(LinspaceGoodLength, LinspaceTemp, APTraceStore[i]) 
		APTrace.append(InterpolateList)

	TimeTrace = np.linspace(-1,2,maxLen)

	return TimeTrace, APTrace

def PlotOpTraces(TimeAPTrace,APAllList,nameLab):
	"""
	This function plots the selected optic flow trace of the 9 events selected in the final movie.
	"""
	fig = plt.figure()

	ax1=plt.subplot(3,3,1)
	plt.plot(TimeAPTrace,APAllList[0])
	plt.ylabel(nameLab,fontsize=10)
	ax1.set_xticklabels([])

	ax2=plt.subplot(3,3,2)
	plt.plot(TimeAPTrace,APAllList[3])
	ax2.set_xticklabels([])

	ax3=plt.subplot(3,3,3)
	plt.plot(TimeAPTrace,APAllList[6])
	ax3.set_xticklabels([])

	ax4=plt.subplot(3,3,4)
	plt.plot(TimeAPTrace,APAllList[1])
	plt.ylabel(nameLab,fontsize=10)
	ax4.set_xticklabels([])

	ax5=plt.subplot(3,3,5)
	plt.plot(TimeAPTrace,APAllList[4])
	ax5.set_xticklabels([])

	ax6=plt.subplot(3,3,6)
	plt.plot(TimeAPTrace,APAllList[7])
	ax6.set_xticklabels([])

	ax7=plt.subplot(3,3,7)
	plt.plot(TimeAPTrace,APAllList[2])
	plt.ylabel(nameLab,fontsize=10)
	plt.xlabel("Time(s)",fontsize=10)

	ax8=plt.subplot(3,3,8)
	plt.plot(TimeAPTrace,APAllList[5])
	plt.xlabel("Time(s)",fontsize=10)

	ax9=plt.subplot(3,3,9)
	plt.plot(TimeAPTrace,APAllList[8])
	plt.xlabel("Time(s)",fontsize=10)

	plt.savefig(unicode(outDirTest + 'opFlow.png'), facecolor=fig.get_facecolor(), edgecolor='none', transparent=True) 
	plt.close(fig)

	return

def findPeakAPA1L(eventIdxTotBeforeL,eventIdxTotAfterL,VelForwGlob,timeFactGlob):
	"""
	This function finds the maximum value of the optic flow trace in a 2 seconds time window after the time point of the event detected.
	"""
	AllAPMean = []
	APListGlob = []
	for i in range(len(eventIdxTotBeforeL)):
		APListExp = []
		timeFact = timeFactGlob[i]
		for j in range(len(eventIdxTotBeforeL[i])):
			if eventIdxTotAfterL[i][j]<len(VelForwGlob[0]):
				idx0Before = eventIdxTotBeforeL[i][j]
				idxAfter = idx0Before + int(timeFact*windowOver)
				APVal = max(VelForwGlob[i][idx0Before:idxAfter])
				AllAPMean.append(APVal)
				APListExp.append(APVal)
		APListGlob.append(APListExp)

	return APListGlob, AllAPMean

def findPeakAPA1R(eventIdxTotBeforeL,eventIdxTotAfterL,VelForwGlob,timeFactGlob):
	"""
	This function finds the minimum value of the optic flow trace in 2 seconds time window after the time point of the event detected.
	"""
	AllAPMean = []
	APListGlob = []
	for i in range(len(eventIdxTotBeforeL)):
		APListExp = []
		timeFact = timeFactGlob[i]
		for j in range(len(eventIdxTotBeforeL[i])):
			if eventIdxTotAfterL[i][j]<len(VelForwGlob[0]):
				idx0Before = eventIdxTotBeforeL[i][j]
				idxAfter = idx0Before + int(timeFact*windowOver)
				APVal = min(VelForwGlob[i][idx0Before:idxAfter])
				AllAPMean.append(APVal)
				APListExp.append(APVal)
		APListGlob.append(APListExp)

	return APListGlob, AllAPMean

def find3EventsA1(AllDiffFluoCombined,FluoValListGlobL,eventIdxTotBeforeL,VelForwGlob,eventIdxTotAfterL):
	"""
	This function selects three events based on the maximum value of the optic flow measurements. 
	"""
	PosLeftList = []
	timeFact = 1500
	windowTF = 2
	AllIdx = []

	for i in range(len(FluoValListGlobL)):
		PosLeftList.append([])

	AllDiffFluoCombinedCopy = AllDiffFluoCombined[:]
	while len(AllIdx)<3:
		if LeftChoice == 1:
			MinAP = max(AllDiffFluoCombined)
		elif LeftChoice == 0:
			MinAP = min(AllDiffFluoCombined)
		AllDiffFluoCombined.remove(MinAP)
		for j in range(len(FluoValListGlobL)):
			try: 
				idx = FluoValListGlobL[j].index(MinAP)
				idxInFluoList = eventIdxTotBeforeL[j][idx]
				if len(AllIdx)>0:
					timeWind = windowTF*timeFact
					listInterTest = list(range(idxInFluoList-timeWind,idxInFluoList+timeWind))
					listInterAllIdx = list(set(AllIdx).intersection(listInterTest))
					if len(listInterAllIdx)==0:
						PosLeftList[j].append(idx)
						AllIdx.append(eventIdxTotBeforeL[j][idx])
				else :
					PosLeftList[j].append(idx)
					AllIdx.append(eventIdxTotBeforeL[j][idx])
			except :
				idxNo = 0
	AllDiffFluoCombinedCopy.sort()

	return PosLeftList

def find3EventsMDNMAN(AllDiffFluoCombined,FluoValListGlobL,FluoValListGlobR,eventIdxTotBeforeL,eventIdxTotBeforeR,VelForwGlob,eventIdxTotAfterL,eventIdxTotAfterR):
	"""
	This function selects three events based on the maximum value of the optic flow measurements. 
	"""
	PosLeftList = []
	PosRightList = []
	timeFact = 1500
	windowTF = 2
	AllIdx = []

	for i in range(len(FluoValListGlobL)):
		PosLeftList.append([])
	for i in range(len(FluoValListGlobR)):
		PosRightList.append([])

	AllDiffFluoCombinedCopy = AllDiffFluoCombined[:]
	while len(AllIdx)<3:
		if SelectChoice == 2:
			MinAP = max(AllDiffFluoCombined)
		elif SelectChoice == 3:
			MinAP = min(AllDiffFluoCombined)
		AllDiffFluoCombined.remove(MinAP)
		for j in range(len(FluoValListGlobL)):
			try: 
				idx = FluoValListGlobL[j].index(MinAP)
				idxInFluoList = eventIdxTotBeforeL[j][idx]
				if len(AllIdx)>0:
					timeWind = windowTF*timeFact
					listInterTest = list(range(idxInFluoList-timeWind,idxInFluoList+timeWind))
					listInterAllIdx = list(set(AllIdx).intersection(listInterTest))
					if len(listInterAllIdx)==0:
						PosLeftList[j].append(idx)
						AllIdx.append(eventIdxTotBeforeL[j][idx])
				else :
					PosLeftList[j].append(idx)
					AllIdx.append(eventIdxTotBeforeL[j][idx])
			except :
				idxNo = 0
		for l in range(len(FluoValListGlobR)):
			try: 
				idxR = FluoValListGlobR[l].index(MinAP)
				idxInFluoListR = eventIdxTotBeforeR[l][idxR]
				if len(AllIdx)>0:
					timeWind = windowTF*timeFact
					listInterTestR = list(range(idxInFluoListR-timeWind,idxInFluoListR+timeWind))
					listInterAllIdxR = list(set(AllIdx).intersection(listInterTestR))
					if len(listInterAllIdxR)==0:
						PosRightList[l].append(idxR)
						AllIdx.append(eventIdxTotBeforeR[l][idxR])
				else :
					PosRightList[l].append(idxR)
					AllIdx.append(eventIdxTotBeforeR[l][idxR])
			except :
				idxNo = 0
	AllDiffFluoCombinedCopy.sort()

	return PosLeftList, PosRightList

def findStartEventIdx(PosLeftList,PosRightList,eventIdxTotBeforeL,eventIdxTotBeforeR):
	"""
	This function selects three events based on the maximum value of the optic flow measurements. 
	"""
	TotIdxList = []
	for i in range(len(PosLeftList)):
		IdxListExp = []
		if len(PosLeftList[i])!=0:
			for j in range(len(PosLeftList[i])):
				IdxListExp.append(eventIdxTotBeforeL[i][PosLeftList[i][j]])
		TotIdxList.append(IdxListExp)

	for m in range(len(PosRightList)):
		if len(PosRightList[m])!=0:
			for n in range(len(PosRightList[m])):
				TotIdxList[m].append(eventIdxTotBeforeR[m][PosRightList[m][n]])

	return TotIdxList

def findPeakAPMAN(eventIdxTotBeforeL,eventIdxTotAfterL,VelForwGlob,timeFactGlob):
	"""
	This function computes the difference in Anterior-Posterior optic flow measurements before and after the event time point. 
	"""
	AllAPMean = []
	APListGlob = []
	AllAPAfter = []
	APAfterGlob=[]
	for i in range(len(eventIdxTotBeforeL)):
		APListExp = []
		timeFact = timeFactGlob[i]
		ExpAPAfter = []
		for j in range(len(eventIdxTotBeforeL[i])):
			if eventIdxTotAfterL[i][j]<len(VelForwGlob[0]):
				idx0Before = eventIdxTotBeforeL[i][j]
				idxAfter = idx0Before + int(timeFact*MinOverA)
				idx1SecBeforeStart = idx0Before - int(timeFact*MinOverB)
				APMaxBeforeStart = max(VelForwGlob[i][idx1SecBeforeStart:idx0Before])
				APMaxMinValAfterStart = min(VelForwGlob[i][idx0Before:idxAfter])
				DiffToAdd = APMaxBeforeStart-APMaxMinValAfterStart
				AllAPMean.append(DiffToAdd)
				APListExp.append(DiffToAdd)
		APListGlob.append(APListExp)

	return APListGlob, AllAPMean

def findPeakAPMDN(eventIdxTotBeforeL,eventIdxTotAfterL,VelForwGlob,timeFactGlob):
	"""
	This function finds the minimum value of anterior-posterior optic flow measurements in a time window of 2 seconds after the time point of the event.
	"""
	AllAPMean = []
	APListGlob = []
	for i in range(len(eventIdxTotBeforeL)):
		APListExp = []
		timeFact = timeFactGlob[i]
		for j in range(len(eventIdxTotBeforeL[i])):
			if eventIdxTotAfterL[i][j]<len(VelForwGlob[0]):
				idx0Before = eventIdxTotBeforeL[i][j]
				idxAfter = idx0Before + int(timeFact*MinOverA)
				APMinValAfterStart = min(VelForwGlob[i][idx0Before:idxAfter])
				AllAPMean.append(APMinValAfterStart)
				APListExp.append(APMinValAfterStart)
		APListGlob.append(APListExp)

	return APListGlob, AllAPMean

# **************** MAIN ***************

def myFunc(IncrIdx):

	fig = plt.figure(facecolor='black')

	rect = mpatches.Rectangle((50,50),60,60,linewidth=1,edgecolor='tomato',facecolor='tomato')
	rect1 = mpatches.Rectangle((50,50),60,60,linewidth=1,edgecolor='tomato',facecolor='tomato')
	rect2 = mpatches.Rectangle((50,50),60,60,linewidth=1,edgecolor='tomato',facecolor='tomato')
	rect3 = mpatches.Rectangle((50,50),60,60,linewidth=1,edgecolor='tomato',facecolor='tomato')
	rect4 = mpatches.Rectangle((50,50),60,60,linewidth=1,edgecolor='tomato',facecolor='tomato')
	rect5 = mpatches.Rectangle((50,50),60,60,linewidth=1,edgecolor='tomato',facecolor='tomato')
	rect6 = mpatches.Rectangle((50,50),60,60,linewidth=1,edgecolor='tomato',facecolor='tomato')
	rect7 = mpatches.Rectangle((50,50),60,60,linewidth=1,edgecolor='tomato',facecolor='tomato')
	rect8 = mpatches.Rectangle((50,50),60,60,linewidth=1,edgecolor='tomato',facecolor='tomato')

	normA = matplotlib.colors.Normalize(vmin=15.0, vmax=255.0)

	ax1 = plt.subplot(3,3,1)
	ax1.yaxis.label.set_color('white')
	ax1.xaxis.label.set_color('white')
	ax1.set_yticklabels([])
	ax1.get_xaxis().set_visible(True)
	ax1.get_yaxis().set_visible(True)
	ax1.xaxis.set_label_coords(0.5,1.12)
	ax1.yaxis.set_label_coords(-0.025,0.5)
	ax1.set_xticks([])
	ax1.set_yticks([])
	if SelectChoice == 1:
		if LeftChoice == 1:
			ax1.text(-100, -35, 'A1 Left', fontsize=12,color='white')
		elif LeftChoice == 0:
			ax1.text(-100, -35, 'A1 Right', fontsize=12,color='white')
	elif SelectChoice == 2:
		ax1.text(-100, -35, 'MAN', fontsize=12,color='white')
	elif SelectChoice == 3:
		ax1.text(-100, -35, 'MDN', fontsize=12,color='white')
	vidImageAx1 = Image.open(VidListTot[0][vidIdxF1E1+IncrIdx])
	if 0 <= timeLen[IncrIdx] <= 0.1:
		ax1.add_patch(rect3)
	ax1.imshow(vidImageAx1, cmap = 'gray', origin='upper',norm=normA)
	plt.ylabel('Event 1', fontsize=10)
	plt.xlabel('Fly 1',fontsize=10)

	ax2 = plt.subplot(3,3,2)
	ax2.yaxis.label.set_color('white')
	ax2.xaxis.label.set_color('white')
	ax2.get_xaxis().set_visible(True)
	ax2.get_yaxis().set_visible(False)
	ax2.xaxis.set_label_coords(0.5,1.12)
	ax2.set_xticks([])
	plt.xlabel('Fly 2',fontsize=10)
	vidImageAx2 = Image.open(VidListTot[3][vidIdxF2E1+IncrIdx])
	if 0 <= timeLen[IncrIdx] <= 0.1:
		ax2.add_patch(rect)
	ax2.imshow(vidImageAx2, cmap = 'gray', origin='upper',norm=normA)

	ax3 = plt.subplot(3,3,3)
	ax3.xaxis.label.set_color('white')
	ax3.set_yticklabels([])
	ax3.get_xaxis().set_visible(True)
	ax3.get_yaxis().set_visible(False)
	ax3.xaxis.set_label_coords(0.5,1.12)
	ax3.set_xticks([])
	plt.xlabel('Fly 3',fontsize=10)
	vidImageAx3 = Image.open(VidListTot[6][vidIdxF3E1+IncrIdx])
	if 0 <= timeLen[IncrIdx] <= 0.1:
		ax3.add_patch(rect6)
	ax3.imshow(vidImageAx3, cmap = 'gray', origin='upper',norm=normA)

	ax4 = plt.subplot(3,3,4)
	ax4.yaxis.label.set_color('white')
	ax4.set_yticklabels([])
	ax4.get_xaxis().set_visible(False)
	ax4.get_yaxis().set_visible(True)
	ax4.set_yticks([])
	ax4.yaxis.set_label_coords(-0.02,0.5)
	plt.ylabel('Event 2', fontsize=10)
	vidImageAx4 = Image.open(VidListTot[1][vidIdxF1E2+IncrIdx])
	if 0 <= timeLen[IncrIdx] <= 0.1:
		ax4.add_patch(rect4)
	ax4.imshow(vidImageAx4, cmap = 'gray', origin='upper',norm=normA)

	ax5 = plt.subplot(3,3,5)
	ax5.get_xaxis().set_visible(False)
	ax5.get_yaxis().set_visible(False)
	vidImageAx5 = Image.open(VidListTot[4][vidIdxF2E2+IncrIdx])
	ax5.imshow(vidImageAx5, cmap = 'gray', origin='upper',norm=normA)
	if 0 <= timeLen[IncrIdx] <= 0.1:
		ax5.add_patch(rect1)

	ax6 = plt.subplot(3,3,6)
	ax6.yaxis.label.set_color('white')
	ax6.set_yticklabels([])
	ax6.get_xaxis().set_visible(False)
	ax6.get_yaxis().set_visible(False)
	vidImageAx6 = Image.open(VidListTot[7][vidIdxF3E2+IncrIdx])
	if 0 <= timeLen[IncrIdx] <= 0.1:
		ax6.add_patch(rect7)
	ax6.imshow(vidImageAx6, cmap = 'gray', origin='upper',norm=normA)

	ax7 = plt.subplot(3,3,7)
	ax7.yaxis.label.set_color('white')
	ax7.set_yticklabels([])
	ax7.get_xaxis().set_visible(False)
	ax7.get_yaxis().set_visible(True)
	ax7.set_yticks([])
	ax7.yaxis.set_label_coords(-0.02,0.5)
	plt.ylabel('Event 3', fontsize=10)
	vidImageAx7 = Image.open(VidListTot[2][vidIdxF1E3+IncrIdx])
	if 0 <= timeLen[IncrIdx] <= 0.1:
		ax7.add_patch(rect5)
	ax7.imshow(vidImageAx7, cmap = 'gray', origin='upper',norm=normA)

	ax8 = plt.subplot(3,3,8)
	ax8.get_xaxis().set_visible(True)
	ax8.get_yaxis().set_visible(False)
	ax8.set_xticks([])
	ax8.xaxis.set_label_coords(0.6,-0.028)
	vidImageAx8 = Image.open(VidListTot[5][vidIdxF2E3+IncrIdx])
	ax8.imshow(vidImageAx8, cmap = 'gray', origin='upper',norm=normA)
	ax8.set_xlabel(str("%.3f" % timeLen[IncrIdx])+"s",fontsize=10,color='white',horizontalalignment='right',verticalalignment='top')
	if 0 <= timeLen[IncrIdx] <= 0.1:
		ax8.add_patch(rect2)

	ax9 = plt.subplot(3,3,9)
	ax9.yaxis.label.set_color('white')
	ax9.set_yticklabels([])
	ax9.get_xaxis().set_visible(False)
	ax9.get_yaxis().set_visible(False)
	vidImageAx9 = Image.open(VidListTot[8][vidIdxF3E3+IncrIdx])
	ax9.imshow(vidImageAx9, cmap = 'gray', origin='upper',norm=normA)
	if 0 <= timeLen[IncrIdx] <= 0.1:
		ax9.add_patch(rect8)

	plt.tight_layout()
	plt.savefig(unicode(outDirTest + "VidFrame" + "%04d" % IncrIdx + '.png'), facecolor=fig.get_facecolor(), edgecolor='none', transparent=True) 
	
	plt.close(fig)

if __name__ == '__main__':

	BehaviourIdxGlob = []
	YawTraceGlob = []
	APTraceGlob = []
	for m in range(len(ListDir)):

		DerivGlobL = []
		DerivGlobR = []
		L_DRGlob = []
		R_DRGlob = []
		VelForwGlob = []
		VelSideGlob = []
		VelTurnGlob = []
		timeFactGlob = []

		DerivTotL = []
		DerivTotR = []

		vidStopTimeAll = []
		timeSecAll = []

		TotalIdxBehaviourFrame = []
		FramePerSecondBehaviour = []
		AllVidStartIdx = []

		for i in ListDir[m]:
			print (i)
			outDir = i + '/output/'

			L_DR, R_DR, velForw, velSide, velTurn, timeSec, vidStopTime, vidStartSysTime, vidStopSysTime, cam_systime = OpenDicData(outDir)

			timeFact = len(timeSec)/vidStopTime
			timeFactGlob.append(timeFact)
			vidStopTimeAll.append(vidStopTime)
			timeSecAll.append(timeSec)

			startVidIdx, stopVidIdx = vidStartAndStop(cam_systime, vidStartSysTime, vidStopSysTime)
			TotalIdxBehaviourFrame.append(stopVidIdx-startVidIdx)
			FramePerSecondBehaviour.append((stopVidIdx-startVidIdx)/vidStopTime)
			AllVidStartIdx.append(startVidIdx)

			deriv_L_DR, interv = getDeriv(L_DR,timeSec)
			deriv_R_DR, interv = getDeriv(R_DR,timeSec)

			for j in deriv_L_DR:
				DerivTotL.append(j)
			for j in deriv_R_DR:
				DerivTotR.append(j)

			DerivGlobL.append(deriv_L_DR)
			DerivGlobR.append(deriv_R_DR)
			L_DRGlob.append(L_DR)
			R_DRGlob.append(R_DR)
			VelForwGlob.append(velForw)
			VelSideGlob.append(velSide)
			VelTurnGlob.append(velTurn)

		DerivLeftNoNan = [x for x in DerivTotL if pd.isnull(x)==False]
		DerivRightNoNan = [x for x in DerivTotR if pd.isnull(x)==False]

		percentL = np.percentile(DerivLeftNoNan,Percentile)
		percentR = np.percentile(DerivRightNoNan,Percentile)

		IndxThreshTotL = getIndxThresh(DerivGlobL,percentL)
		IndxThreshTotR = getIndxThresh(DerivGlobR,percentR)

		eventIdxTotBeforeL, eventIdxTotAfterL = getEventIdx(IndxThreshTotL,DerivGlobL,timeFactGlob,window,timeSecAll)
		eventIdxTotBeforeR, eventIdxTotAfterR = getEventIdx(IndxThreshTotR,DerivGlobR,timeFactGlob,window,timeSecAll)

		if SelectChoice == 1:
			LFromLAREventGlob, RFromLAREventGlob ,LEventOnlyGlob, REventOnlyGlob  = A1EventSeparated(eventIdxTotBeforeL,eventIdxTotBeforeR,timeFactGlob)
			LEventOnlyAfter = getEventAfter(LEventOnlyGlob,eventIdxTotBeforeL,eventIdxTotAfterL)
			REventOnlyAfter = getEventAfter(REventOnlyGlob,eventIdxTotBeforeR,eventIdxTotAfterR)
			if LeftChoice == 1 :
				FluoValListGlobL, AllDiffFluoValL = findPeakAPA1L(LEventOnlyGlob,LEventOnlyAfter,VelTurnGlob,timeFactGlob)
				AllDiffFluoCombined = AllDiffFluoValL
				PosLeftList = find3EventsA1(AllDiffFluoCombined,FluoValListGlobL,LEventOnlyGlob,VelTurnGlob,LEventOnlyAfter)
				TotIdxListStartEvent = findStartEventIdxA1(PosLeftList,LEventOnlyGlob)
			else : 
				FluoValListGlobR, AllDiffFluoValR = findPeakAPA1R(REventOnlyGlob,REventOnlyAfter,VelTurnGlob,timeFactGlob)
				AllDiffFluoCombined = AllDiffFluoValR
				PosLeftList = find3EventsA1(AllDiffFluoCombined,FluoValListGlobR,REventOnlyGlob,VelTurnGlob,REventOnlyAfter)
				TotIdxListStartEvent = findStartEventIdxA1(PosLeftList,REventOnlyGlob)

		elif SelectChoice == 2:
			APListGlobL, AllAPMeanL = findPeakAPMAN(eventIdxTotBeforeL,eventIdxTotAfterL,VelForwGlob,timeFactGlob)
			APListGlobR, AllAPMeanR = findPeakAPMAN(eventIdxTotBeforeR,eventIdxTotAfterR,VelForwGlob,timeFactGlob)
			AllAPMeanCombined = AllAPMeanL + AllAPMeanR
			PosLeftList, PosRightList = find3EventsMDNMAN(AllAPMeanCombined,APListGlobL,APListGlobR,eventIdxTotBeforeL,eventIdxTotBeforeR,VelForwGlob,eventIdxTotAfterL,eventIdxTotAfterR)
			TotIdxListStartEvent = findStartEventIdx(PosLeftList,PosRightList,eventIdxTotBeforeL,eventIdxTotBeforeR)
		
		elif SelectChoice == 3:
			APListGlobL, AllAPMeanMDN = findPeakAPMDN(eventIdxTotBeforeL,eventIdxTotAfterL,VelForwGlob,timeFactGlob)
			APListGlobR, AllAPMeanRMDN = findPeakAPMDN(eventIdxTotBeforeR,eventIdxTotAfterR,VelForwGlob,timeFactGlob)
			AllAPMeanCombined = AllAPMeanMDN + AllAPMeanRMDN
			PosLeftList, PosRightList = find3EventsMDNMAN(AllAPMeanCombined,APListGlobL,APListGlobR,eventIdxTotBeforeL,eventIdxTotBeforeR,VelForwGlob,eventIdxTotAfterL,eventIdxTotAfterR)
			TotIdxListStartEvent = findStartEventIdx(PosLeftList,PosRightList,eventIdxTotBeforeL,eventIdxTotBeforeR)

		if SelectChoice == 1: 
			YawTraceExp = getOpTrace(TotIdxListStartEvent,VelTurnGlob,timeFactGlob)
			YawTraceGlob.append(YawTraceExp)
		else: 
			APTraceExp = getOpTrace(TotIdxListStartEvent,VelForwGlob,timeFactGlob)
			APTraceGlob.append(APTraceExp)

		BehaviourIdxList = FindBehaviourIdx(TotIdxListStartEvent,FramePerSecondBehaviour,timeFactGlob,AllVidStartIdx,CamFudge[m])
		BehaviourIdxGlob.append(BehaviourIdxList)

	VidListTot = []
	for m in range(len(BehaviourIdxGlob)):
		for i in range(len(BehaviourIdxGlob[m])):
			LiDir = ListDir[m][BehaviourIdxGlob[m][i][0]]
			VidDir = LiDir + '/behavior_imgs'
			vidListTemp = getBehaviorImg(VidDir)
			VidListTot.append(vidListTemp)
"""
	# Uncomment this section to see the optic flow traces for the 9 videos selected. If you want to see the traces, you should comment two lines containing the 'Pool'
	if SelectChoice == 1: 
		TimeYAWTrace, YAWAllList = aligneAPTraces(YawTraceGlob)
		PlotOpTraces(TimeYAWTrace,YAWAllList,"Yaw rot./s")
	else: 
		TimeAPTrace, APAllList = aligneAPTraces(APTraceGlob)
		PlotOpTraces(TimeAPTrace,APAllList,"AP rot./s")
"""
#""" Comment this section if you want to see the optic flow traces coded from lines 964 to 970
	IncrIdx = int(windowVideo*30)

	vidIdxF1E1 = BehaviourIdxGlob[0][0][1]-int(IncrIdx/3)
	vidIdxF1E2 = BehaviourIdxGlob[0][1][1]-int(IncrIdx/3)
	vidIdxF1E3 = BehaviourIdxGlob[0][2][1]-int(IncrIdx/3)

	vidIdxF2E1 = BehaviourIdxGlob[1][0][1]-int(IncrIdx/3)
	vidIdxF2E2 = BehaviourIdxGlob[1][1][1]-int(IncrIdx/3)
	vidIdxF2E3 = BehaviourIdxGlob[1][2][1]-int(IncrIdx/3)

	vidIdxF3E1 = BehaviourIdxGlob[2][0][1]-int(IncrIdx/3)
	vidIdxF3E2 = BehaviourIdxGlob[2][1][1]-int(IncrIdx/3)
	vidIdxF3E3 = BehaviourIdxGlob[2][2][1]-int(IncrIdx/3)

	timeLen = np.linspace(-1,2,IncrIdx)

	pool = Pool()
	pool.map(myFunc,range(0,IncrIdx))
#"""

