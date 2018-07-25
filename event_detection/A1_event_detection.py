# -*- coding: utf-8 -*-
"""

Goal:
    This script computes the average and 95%  confidence interval of the DR/R%  traces and optic flow traces.
    
Method:
    The script opens data from the dictionnary for all experiments.
    It computes the first order derivative and detects the events based on the threshold chosen by the user (90 percentile of the derivative in this case) and separates events between left only events 
    related, right only events related or left and right events happening in a time window of 0.25 second. 
    It aligns all the events together by setting the time point of the event to 0 second and storing a time window of 20 seconds of data around the time point of the event detected. The script also 
    computes the average of all the optic flow and DR/R%  traces.
    Finally, it uses the Seaborn library to compute 95%  bootstrapped confidence interval and to plot them.

Note : 
	The script computes those plots either for left neuron related events or right neuron related events - if you want to compute
	left neuron related events, you need to set the boolL value to 1. If you want to compute the right neuron events you need to set 
	the boolL value to 0. 
	The optic flow values are given in rotation per seconds -> 1 rot/s = 360°/s. = 31.42 mm/s. , we are using a 10 mm diameter ball.

"""

window = 10
timeWindowA1 = 0.125
Percentile = 90
#boolL = 1
boolL = 0

import os, os.path
import sys
import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
from random import randint
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns

#Paths to load dictionnaries containing the data from the different experiments
ListDir = []
############ S 47 - 2 
ListDir.append('YOUR_PATH_TO/A1-2/fly1_S47_tdTom-GC6s_005')
ListDir.append('YOUR_PATH_TO/A1-2/fly1_S47_tdTom-GC6s_006')
ListDir.append('YOUR_PATH_TO/A1-2/fly1_S47_tdTom-GC6s_007')
ListDir.append('YOUR_PATH_TO/A1-2/fly1_S47_tdTom-GC6s_008')
ListDir.append('YOUR_PATH_TO/A1-2/fly1_S47_tdTom-GC6s_009')
ListDir.append('YOUR_PATH_TO/A1-2/fly1_S47_tdTom-GC6s_010')
ListDir.append('YOUR_PATH_TO/A1-2/fly1_S47_tdTom-GC6s_011')
ListDir.append('YOUR_PATH_TO/A1-2/fly1_S47_tdTom-GC6s_013')
ListDir.append('YOUR_PATH_TO/A1-2/fly1_S47_tdTom-GC6s_014')
ListDir.append('YOUR_PATH_TO/A1-2/fly1_S47_tdTom-GC6s_015')

############ S 47 - 3
ListDir.append('YOUR_PATH_TO/A1-3/fly1_S47_tdTom-GC6s_fly3_004')
ListDir.append('YOUR_PATH_TO/A1-3/fly1_S47_tdTom-GC6s_fly3_005')
ListDir.append('YOUR_PATH_TO/A1-3/fly1_S47_tdTom-GC6s_fly3_006')
ListDir.append('YOUR_PATH_TO/A1-3/fly1_S47_tdTom-GC6s_fly3_008')
ListDir.append('YOUR_PATH_TO/A1-3/fly1_S47_tdTom-GC6s_fly3_009')
ListDir.append('YOUR_PATH_TO/A1-3/fly1_S47_tdTom-GC6s_fly3_010')
ListDir.append('YOUR_PATH_TO/A1-3/fly1_S47_tdTom-GC6s_fly3_011')
ListDir.append('YOUR_PATH_TO/A1-3/fly1_S47_tdTom-GC6s_fly3_012')
ListDir.append('YOUR_PATH_TO/A1-3/fly1_S47_tdTom-GC6s_fly3_013')
ListDir.append('YOUR_PATH_TO/A1-3/fly1_S47_tdTom-GC6s_fly3_014')

########## S 47 - 4
ListDir.append('YOUR_PATH_TO/A1-4/imaging_S47_fly1_data_001')
ListDir.append('YOUR_PATH_TO/A1-4/imaging_S47_fly1_data_002')
ListDir.append('YOUR_PATH_TO/A1-4/imaging_S47_fly1_data_003')
ListDir.append('YOUR_PATH_TO/A1-4/imaging_S47_fly1_data_004')
ListDir.append('YOUR_PATH_TO/A1-4/imaging_S47_fly1_data_005')
ListDir.append('YOUR_PATH_TO/A1-4/imaging_S47_fly1_data_006')
ListDir.append('YOUR_PATH_TO/A1-4/imaging_S47_fly1_data_007')
ListDir.append('YOUR_PATH_TO/A1-4/imaging_S47_fly1_data_009')

######## S 47 - 5
ListDir.append('YOUR_PATH_TO/A1-5/imaging_S47_fly3_data_002')
ListDir.append('YOUR_PATH_TO/A1-5/imaging_S47_fly3_data_004')
ListDir.append('YOUR_PATH_TO/A1-5/imaging_S47_fly3_data_006')
ListDir.append('YOUR_PATH_TO/A1-5/imaging_S47_fly3_data_007')
ListDir.append('YOUR_PATH_TO/A1-5/imaging_S47_fly3_data_008')
ListDir.append('YOUR_PATH_TO/A1-5/imaging_S47_fly3_data_009')
ListDir.append('YOUR_PATH_TO/A1-5/imaging_S47_fly3_data_011')
ListDir.append('YOUR_PATH_TO/A1-5/imaging_S47_fly3_data_012')
ListDir.append('YOUR_PATH_TO/A1-5/imaging_S47_fly3_data_016')

expName = 'A1-allTogether'
outFigDir = 'YOUR_PATH_TO/Event-A1/'+expName
outFigDirSumT = 'YOUR_PATH_TO/Event-A1/'+expName+'/summary/'
outFigDirSum = outFigDirSumT+str(int(Percentile))+'/'

if boolL == 1:
	ExpNameP = 'A1-LO-'
elif boolL == 0:
	ExpNameP = 'A1-RO-'

if not os.path.exists(outFigDir):    
    os.makedirs(outFigDir)
if not os.path.exists(outFigDirSumT):    
    os.makedirs(outFigDirSumT)
if not os.path.exists(outFigDirSum):    
    os.makedirs(outFigDirSum)


def OpenDicData(outDir):
	"""
	Opens the data from the dictionnary for one experiment and extracts DR/R% and optic flow data.
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

	return L_DR, R_DR, velForw, velSide, velTurn, timeSec, vidStopTime

def getDeriv(L_DR,timeSec):
	"""
	Computes the first order derivative of DR/R% traces.
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
	Finds the indexes of the derivative when it crosses the threshold value.
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

def getEventIdx(idxThreshTot,derivGlob,timeFact,window):
	"""
	Find the indexes of the 0 crossing of the derivative occuring just before the idx found in getIndxThresh.
	"""
	eventIdxTot = []

	timeFrame = int(window*timeFact)

	for idxThresh,deriv in itertools.izip(idxThreshTot,derivGlob):
		eventIdx = []
		for j in idxThresh:
			for i in range(len(deriv[:j])):
				if deriv[j-i] < 0:
					if deriv[j-i+1]>0:
						if ((j-i+1) in eventIdx) == False:
							if (j-i+1)>timeFrame:
								eventIdx.append(j-i+1)
						break
		eventIdxTot.append(eventIdx)
	
	return eventIdxTot

def alignedDataToPlot(eventListIdxGlob,L_DRGlob,velForwGlob,velSideGlob, velTurnGlob,timeFactGlob,window):
	"""
	Stores the 20 seconds window of DR/R% and optic flow data around each event detected.
	"""
	allVelForw = []
	allVelSide = []
	allVelTurn = []
	allDrVal = []
	allDrValRandEvent = []
	baselineList = []
	countList = []
	countAll = []
	allDrValRandEvent = []
	allVelForwRand = []
	allVelSideRand = []
	allVelTurnRand = []
	allTimeSeq = []

	for eventListIdx,L_DR,velForw,velSide,velTurn,timeFact in itertools.izip(eventListIdxGlob,L_DRGlob,velForwGlob,velSideGlob,velTurnGlob,timeFactGlob):
		countTemp = []
		count = 0
		timeFrame = int(window*timeFact)
		timeSeq = np.linspace(-window,window,2*timeFrame)
		for i in eventListIdx:
			if i > timeFrame:
				tempList = [x - L_DR[i] for x in L_DR[i-timeFrame:i+timeFrame]]
				if len(tempList) < len(timeSeq):
					lenToAdd = len(timeSeq)-len(tempList)
	 	 			nanTail = np.empty(lenToAdd)
			 	 	nanTail.fill(np.nan)
					DRRList = list(np.append(tempList,nanTail))
					VelForwList = list(np.append(velForw[i-timeFrame:i+timeFrame],nanTail))
					VelSideList = list(np.append(velSide[i-timeFrame:i+timeFrame],nanTail))
					VelTurnList = list(np.append(velTurn[i-timeFrame:i+timeFrame],nanTail))
					allDrVal.append(DRRList)
					allVelForw.append(VelForwList)
					allVelSide.append(VelSideList)
					allVelTurn.append(VelTurnList)
					allTimeSeq.append(timeSeq)
				else :
					allDrVal.append(tempList)
					allVelForw.append(velForw[i-timeFrame:i+timeFrame])
					allVelSide.append(velSide[i-timeFrame:i+timeFrame])
					allVelTurn.append(velTurn[i-timeFrame:i+timeFrame])
					allTimeSeq.append(timeSeq)
				count += 1
				countTemp.append(count)		

		countList.append(countTemp)
		countAll.append(count)

	sumCount = sum(countAll)
	randEvent = []
	for j in range(len(countList)):
		timeFact = timeFactGlob[j]
		timeFrame = int(window*timeFact)
		randEventTemp = []
		L_DR = L_DRGlob[j]
		for i in range(len(countList[j])):
			randNum = randint(timeFrame+1,len(L_DR)-timeFrame-1)
			randEventTemp.append(randNum)
		randEvent.append(randEventTemp)

	for m,L_DR,velForw,velSide,velTurn,timeFact in itertools.izip(randEvent,L_DRGlob,velForwGlob,velSideGlob,velTurnGlob,timeFactGlob):
		timeFrame = int(window*timeFact)
		for i in m:
			tempRandList = [x - L_DR[i] for x in L_DR[i-timeFrame:i+timeFrame]]
			allDrValRandEvent.append(tempRandList)
			allVelForwRand.append(velForw[i-timeFrame:i+timeFrame])
			allVelSideRand.append(velSide[i-timeFrame:i+timeFrame])
			allVelTurnRand.append(velTurn[i-timeFrame:i+timeFrame])


	return allVelForw, allDrVal, allVelSide, allVelTurn, allDrValRandEvent, allVelForwRand, allVelSideRand, allVelTurnRand, allTimeSeq

def detectAverageVal(ValToAv):
	"""
	Computes the average value from all the DR/R% and optic flow traces.
	"""
	InterpolatedValToAv = []
	maxLengthTemplateList = []
	for i in ValToAv:
		maxLengthTemplateList.append(len(i))
	maxLengthTemplate = max(maxLengthTemplateList)

	for i in ValToAv:
		LinspaceTemp=np.linspace(0,len(i)-1,len(i))
		LinspaceGoodLength=np.linspace(0,len(i)-1,maxLengthTemplate)
		InterpolateList=np.interp(LinspaceGoodLength, LinspaceTemp, i) 
		InterpolatedValToAv.append(InterpolateList)

	col = maxLengthTemplate
	lines = len(ValToAv)
	tempArray = np.zeros((lines,col))

	for i in range(len(InterpolatedValToAv)):
		tempArray[i]=InterpolatedValToAv[i]

	averVal = np.nanmean(tempArray,axis=0)

	return averVal,tempArray

def A1EventSeparated(eventIdxTotL,eventIdxTotR,timeFactGlob):
	"""
	Separates events detected between left events only, right events only or left and right events happening in the same time window of 0.25 second.
	"""
	LFromLAREventGlob = []
	RFromLAREventGlob = []
	LEventOnlyGlob = []
	REventOnlyGlob = []
	lenL = 0
	lenR = 0
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
		lenL = lenL + len(LEventExp)
		lenR = lenR + len(REventExp)
		LEventOnlyGlob.append(LEventExp)
		REventOnlyGlob.append(REventExp)
		LFromLAREventGlob.append(LFromLAREventExp)
		RFromLAREventGlob.append(RFromLAREventExp)

	return LFromLAREventGlob, RFromLAREventGlob ,LEventOnlyGlob, REventOnlyGlob 

def detectAverageTime(AlltimeSeqAlignedL,averDrRValL):
	"""
	Finds the time sec that has the same number of data points as the average DR/R% trace.
	"""
	lenAver = len(averDrRValL)
	for i in AlltimeSeqAlignedL:
		if len(i)==lenAver:
			averTime=i

	return averTime 

def interpolAllData(averDrRValL,allVelForwL, allDrValL, allVelSideL, allVelTurnL, allDrValRandEventL, allVelForwRandL, allVelSideRandL, allVelTurnRandL):
	"""
	Interpoles all the traces to the larger trace in the 20 seconds batch. 
	"""
	allVelForwLN = []
	allDrValLN = []
	allVelSideLN=[]
	allVelTurnLN=[]
	allDrValRandEventLN=[]
	allVelForwRandLN=[]
	allVelSideRandLN=[]
	allVelTurnRandLN=[]

	lenToInter = len(averDrRValL)
	for i in range(len(allVelForwL)):
		LinspaceTemp=np.linspace(0,len(allVelForwL[i])-1,len(allVelForwL[i]))
		LinspaceGoodLength=np.linspace(0,len(allVelForwL[i])-1,lenToInter)
		allVelForwLN.append(list(np.interp(LinspaceGoodLength, LinspaceTemp, allVelForwL[i])))
		allDrValLN.append(list(np.interp(LinspaceGoodLength, LinspaceTemp, allDrValL[i])))
		allVelSideLN.append(list(np.interp(LinspaceGoodLength, LinspaceTemp, allVelSideL[i])))
		allVelTurnLN.append(list(np.interp(LinspaceGoodLength, LinspaceTemp, allVelTurnL[i])))
		allDrValRandEventLN.append(list(np.interp(LinspaceGoodLength, LinspaceTemp, allDrValRandEventL[i])))
		allVelForwRandLN.append(list(np.interp(LinspaceGoodLength, LinspaceTemp, allVelForwRandL[i])))
		allVelSideRandLN.append(list(np.interp(LinspaceGoodLength, LinspaceTemp, allVelSideRandL[i])))
		allVelTurnRandLN.append(list(np.interp(LinspaceGoodLength, LinspaceTemp, allVelTurnRandL[i])))

	return allVelForwLN, allDrValLN, allVelSideLN, allVelTurnLN, allDrValRandEventLN, allVelForwRandLN, allVelSideRandLN, allVelTurnRandLN

def diviseData(allVelForwL, allDrValL, allVelSideL, allVelTurnL, allDrValRandEventL, allVelForwRandL, allVelSideRandL, allVelTurnRandL,averTimeL,averDrRValL,RandAverL,averVelForwValL,RandVFL,averVelSideValL,RandVSL,averVelTurnValL,RandVTL):
	"""
	Reduces the number of data points to 500 data points per second (10000 data points for 20 seconds).
	"""
	allVelForwLN = []
	allDrValLN = []
	allVelSideLN=[]
	allVelTurnLN=[]
	allDrValRandEventLN=[]
	allVelForwRandLN=[]
	allVelSideRandLN=[]
	allVelTurnRandLN=[]
	averDrRValLN = []
	RandAverLN = []
	averTimeLN = []
	averVelForwValLN = []
	RandVFLN = []
	averVelSideValLN = []
	RandVSLN = []
	averVelTurnValLN = []
	RandVTLN = []

	factor = 10000
	lenAverMid = int(len(averTimeL)/2)
	countAver=0
	for j in range(factor):
		lenDiviAver = len(averTimeL)/factor
		averTimeLN.append(averTimeL[int(countAver*lenDiviAver)])
		averDrRValLN.append(averDrRValL[int(countAver*lenDiviAver)])
		RandAverLN.append(RandAverL[int(countAver*lenDiviAver)])
		averVelForwValLN.append(averVelForwValL[int(countAver*lenDiviAver)])
		RandVFLN.append(RandVFL[int(countAver*lenDiviAver)])
		averVelSideValLN.append(averVelSideValL[int(countAver*lenDiviAver)])
		RandVSLN.append(RandVSL[int(countAver*lenDiviAver)])
		averVelTurnValLN.append(averVelTurnValL[int(countAver*lenDiviAver)])
		RandVTLN.append(RandVTL[int(countAver*lenDiviAver)])
		countAver=countAver+1
	averTimeLN.append(averTimeL[-1])
	averDrRValLN.append(averDrRValL[-1])
	RandAverLN.append(RandAverL[-1])
	averVelForwValLN.append(averVelForwValL[-1])
	RandVFLN.append(RandVFL[-1])
	averVelSideValLN.append(averVelSideValL[-1])
	RandVSLN.append(RandVSL[-1])
	averVelTurnValLN.append(averVelTurnValL[-1])
	RandVTLN.append(RandVTL[-1])

	for i in range(len(allDrValL)):
		allVelForwLNT = []
		allDrValLNT = []
		allVelSideLNT=[]
		allVelTurnLNT=[]
		allDrValRandEventLNT=[]
		allVelForwRandLNT=[]
		allVelSideRandLNT=[]
		allVelTurnRandLNT=[]
		averDrRValLNT = []
		RandAverLNT = []
		count = 0
		lenDivi = len(averTimeL)/factor

		for j in range(factor):
			allVelForwLNT.append(allVelForwL[i][int(count*lenDivi)])
			allDrValLNT.append(allDrValL[i][int(count*lenDivi)])
			allVelSideLNT.append(allVelSideL[i][int(count*lenDivi)])
			allVelTurnLNT.append(allVelTurnL[i][int(count*lenDivi)])
			allDrValRandEventLNT.append(allDrValRandEventL[i][int(count*lenDivi)])
			allVelForwRandLNT.append(allVelForwRandL[i][int(count*lenDivi)])
			allVelSideRandLNT.append(allVelSideRandL[i][int(count*lenDivi)])
			allVelTurnRandLNT.append(allVelTurnRandL[i][int(count*lenDivi)])
			count=count+1

		allVelForwLNT.append(allVelForwL[i][-1])
		allDrValLNT.append(allDrValL[i][-1])
		allVelSideLNT.append(allVelSideL[i][-1])
		allVelTurnLNT.append(allVelTurnL[i][-1])
		allDrValRandEventLNT.append(allDrValRandEventL[i][-1])
		allVelForwRandLNT.append(allVelForwRandL[i][-1])
		allVelSideRandLNT.append(allVelSideRandL[i][-1])
		allVelTurnRandLNT.append(allVelTurnRandL[i][-1])

		allVelForwLN.append(allVelForwLNT)
		allDrValLN.append(allDrValLNT)
		allVelSideLN.append(allVelSideLNT)
		allVelTurnLN.append(allVelTurnLNT)
		allDrValRandEventLN.append(allDrValRandEventLNT)
		allVelForwRandLN.append(allVelForwRandLNT)
		allVelSideRandLN.append(allVelSideRandLNT)
		allVelTurnRandLN.append(allVelTurnRandLNT)

	return allVelForwLN, allDrValLN, allVelSideLN, allVelTurnLN, allDrValRandEventLN, allVelForwRandLN, allVelSideRandLN, allVelTurnRandLN, averTimeLN,averDrRValLN,RandAverLN,averVelForwValLN,RandVFLN,averVelSideValLN,RandVSLN,averVelTurnValLN,RandVTLN

def diviseDataR(allDrValR, allDrValRandEventR,averTimeR,averDrRValR,RandAverR):
	"""
	Reduces the number of data points to 500 per second.
	"""
	allDrValLN = []
	allDrValRandEventLN=[]
	averDrRValLN = []
	RandAverLN = []
	averTimeLN = []

	factor = 10000
	lenAverMid = int(len(averTimeL)/2)
	countAver=0
	for j in range(factor):
		lenDiviAver = len(averTimeR)/factor
		averTimeLN.append(averTimeR[int(countAver*lenDiviAver)])
		averDrRValLN.append(averDrRValR[int(countAver*lenDiviAver)])
		RandAverLN.append(RandAverR[int(countAver*lenDiviAver)])
		countAver=countAver+1
	averTimeLN.append(averTimeR[-1])
	averDrRValLN.append(averDrRValR[-1])
	RandAverLN.append(RandAverR[-1])

	for i in range(len(allDrValR)):
		allDrValLNT = []
		allDrValRandEventLNT=[]
		count = 0
		lenDivi = len(averTimeR)/factor

		for j in range(factor):
			allDrValLNT.append(allDrValR[i][int(count*lenDivi)])
			allDrValRandEventLNT.append(allDrValRandEventR[i][int(count*lenDivi)])
			count=count+1

		allDrValLNT.append(allDrValR[i][-1])
		allDrValRandEventLNT.append(allDrValRandEventR[i][-1])

		allDrValLN.append(allDrValLNT)
		allDrValRandEventLN.append(allDrValRandEventLNT)

	return allDrValLN, allDrValRandEventLN, averTimeLN,averDrRValLN,RandAverLN

def figICSNSOneSubplot(allDrValL,allDrValRandEventL,timeSeqL,averDrRValL,RandAverL,nameF,ylimMin,ylimMax,boolT):
	"""
	Plots the average values and 95% confidence interval around the average thanks to the Seaborn library.
	"""
	fig = plt.figure(facecolor='white')
	ax1 = fig.add_subplot(111)
	if boolT == 0:
		ax1.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
		plt.ylabel('Baseline - $\, \%$ DR/R ', fontsize=10)
	elif boolT == 1 :
		ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
		plt.ylabel('rot./s', fontsize=10)
	elif boolT == 2 :
		ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
		plt.ylabel('rot./s', fontsize=10)
	ax1.xaxis.label.set_color('black')
	ax1.spines['left'].set_color('black')
	ax1.spines['bottom'].set_color('black')
	ax1.spines['top'].set_visible(False)
	ax1.spines['right'].set_visible(False)
	ax1.tick_params(axis='x', colors='black',direction='out',top='off',bottom='on')
	ax1.tick_params(axis='y', colors='black',direction='out',left='on',right='off')

	ax1.axhline(0, linestyle='dashed',color='gray',linewidth=0.5)
	ax1.yaxis.label.set_color('black')
	plt.xlim(timeSeqL[0],timeSeqL[-1])
	plt.ylim(ylimMin,ylimMax)
	plt.axvline(0, linestyle='dashed',color='gray',linewidth=0.5)

	ax1 = sns.tsplot(allDrValL,timeSeqL,ci=95,color='lightskyblue',estimator=np.nanmean)
	ax1 = sns.tsplot(allDrValRandEventL,timeSeqL,ci=95,color='darksalmon',estimator=np.nanmean)

	plt.plot(timeSeqL,averDrRValL,color="midnightblue",alpha=1,label="average Value")
	plt.plot(timeSeqL,RandAverL,color="red",alpha=1,label="average Value")

	plt.savefig(outFigDirSum +'IC-'+ExpNameP+nameF+'.eps',format='eps',facecolor=fig.get_facecolor(), edgecolor='none', transparent=False) 
	plt.savefig(outFigDirSum +'IC-'+ExpNameP+nameF+'.png',format='png',facecolor=fig.get_facecolor(), edgecolor='none', transparent=False)  
	plt.close(fig)
	return

# ********** MAIN *****************

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

VidStopTimeTot = 0
 
for i in ListDir:
	print (i)
	outDir = i + '/output/'

	L_DR, R_DR, velForw, velSide, velTurn, timeSec, vidStopTime = OpenDicData(outDir)

	timeFact = len(timeSec)/vidStopTime
	timeFactGlob.append(timeFact)
	vidStopTimeAll.append(vidStopTime)
	timeSecAll.append(timeSec)
	VidStopTimeTot = VidStopTimeTot + vidStopTime

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

eventIdxTotL = getEventIdx(IndxThreshTotL,DerivGlobL,timeFactGlob[0],window)
eventIdxTotR = getEventIdx(IndxThreshTotR,DerivGlobR,timeFactGlob[0],window)

LFromLAREventGlob, RFromLAREventGlob ,LEventOnlyGlob, REventOnlyGlob = A1EventSeparated(eventIdxTotL,eventIdxTotR,timeFactGlob)

if boolL == 1:
	allVelForwL, allDrValL, allVelSideL, allVelTurnL, allDrValRandEventL, allVelForwRandL, allVelSideRandL, allVelTurnRandL, AlltimeSeqAlignedL = alignedDataToPlot(LEventOnlyGlob,L_DRGlob,VelForwGlob,VelSideGlob, VelTurnGlob,timeFactGlob,window)
	allVelForwR, allDrValR, allVelSideR, allVelTurnR, allDrValRandEventR, allVelForwRandR, allVelSideRandR, allVelTurnRandR, AlltimeSeqAlignedR = alignedDataToPlot(LEventOnlyGlob,R_DRGlob,VelForwGlob,VelSideGlob, VelTurnGlob,timeFactGlob,window)
elif boolL == 0:
	allVelForwL, allDrValL, allVelSideL, allVelTurnL, allDrValRandEventL, allVelForwRandL, allVelSideRandL, allVelTurnRandL, AlltimeSeqAlignedL = alignedDataToPlot(REventOnlyGlob,R_DRGlob,VelForwGlob,VelSideGlob, VelTurnGlob,timeFactGlob,window)
	allVelForwR, allDrValR, allVelSideR, allVelTurnR, allDrValRandEventR, allVelForwRandR, allVelSideRandR, allVelTurnRandR, AlltimeSeqAlignedR = alignedDataToPlot(REventOnlyGlob,L_DRGlob,VelForwGlob,VelSideGlob, VelTurnGlob,timeFactGlob,window)


################# LEFT AND RIGHT ALL EVENTS TOGETHER ############################
# # # DETECTING AVERAGE OF TRACES - LEFT if boolL == 1 or RIGHT if boolL == 0
averDrRValL, tempArrayFL = detectAverageVal(allDrValL)
averVelForwValL, tempArrayVFL = detectAverageVal(allVelForwL)
averVelSideValL, tempArrayVLL = detectAverageVal(allVelSideL)
averVelTurnValL, tempArrayVTL = detectAverageVal(allVelTurnL)
averTimeL = detectAverageTime(AlltimeSeqAlignedL,averDrRValL)


# # # DETECTING AVERAGE OF TRACES - LEFT if boolL == 1 (or RIGHT if boolL==0) - RANDOM EVENT
RandAverL, tempArrayRandLeft = detectAverageVal(allDrValRandEventL)
RandVFL, TARVFL = detectAverageVal(allVelForwRandL)
RandVSL, TARVSL = detectAverageVal(allVelSideRandL)
RandVTL, TARVTL = detectAverageVal(allVelTurnRandL)

# # # DETECTING AVERAGE OF TRACES - RIGHT if boolL == 1 of LEFT if boolL==0
averDrRValR, tempArrayFR = detectAverageVal(allDrValR)
averTimeR = detectAverageTime(AlltimeSeqAlignedR,averDrRValR)

# # # DETECTING AVERAGE OF TRACES - RIGHT if boolL == 1 (of LEFT if boolL==0 ) - RANDOM EVENT
RandAverR, tempArrayRandRight = detectAverageVal(allDrValRandEventR)

allVelForwL, allDrValL, allVelSideL, allVelTurnL, allDrValRandEventL, allVelForwRandL, allVelSideRandL, allVelTurnRandL = interpolAllData(averDrRValL,allVelForwL, allDrValL, allVelSideL, allVelTurnL, allDrValRandEventL, allVelForwRandL, allVelSideRandL, allVelTurnRandL)
allVelForwR, allDrValR, allVelSideR, allVelTurnR, allDrValRandEventR, allVelForwRandR, allVelSideRandR, allVelTurnRandR = interpolAllData(averDrRValR,allVelForwR, allDrValR, allVelSideR, allVelTurnR, allDrValRandEventR, allVelForwRandR, allVelSideRandR, allVelTurnRandR)

allVelForwL, allDrValL, allVelSideL, allVelTurnL, allDrValRandEventL, allVelForwRandL, allVelSideRandL, allVelTurnRandL,averTimeL,averDrRValL,RandAverL,averVelForwValL,RandVFL,averVelSideValL,RandVSL,averVelTurnValL,RandVTL = diviseData(allVelForwL, allDrValL, allVelSideL, allVelTurnL, allDrValRandEventL, allVelForwRandL, allVelSideRandL, allVelTurnRandL,averTimeL,averDrRValL,RandAverL,averVelForwValL,RandVFL,averVelSideValL,RandVSL,averVelTurnValL,RandVTL)
allDrValR,allDrValRandEventR,averTimeR,averDrRValR,RandAverR = diviseDataR(allDrValR, allDrValRandEventR,averTimeR,averDrRValR,RandAverR)

if boolL == 1:
	figICSNSOneSubplot(allDrValL,allDrValRandEventL,averTimeL,averDrRValL,RandAverL,'-S1-L',-15,40,0)
	figICSNSOneSubplot(allDrValR,allDrValRandEventR,averTimeR,averDrRValR,RandAverR,'-S2-R',-15,40,0)
elif boolL == 0:
	figICSNSOneSubplot(allDrValL,allDrValRandEventL,averTimeL,averDrRValL,RandAverL,'-S1-R',-20,40,0)
	figICSNSOneSubplot(allDrValR,allDrValRandEventR,averTimeR,averDrRValR,RandAverR,'-S2-L',-20,40,0)
figICSNSOneSubplot(allVelTurnL,allVelTurnRandL,averTimeL,averVelTurnValL,RandVTL,'-S5-Yaw',-0.04,0.08,2)
figICSNSOneSubplot(allVelForwL,allVelForwRandL,averTimeL,averVelForwValL,RandVFL,'-S3-AP',0.0,0.1,1)
figICSNSOneSubplot(allVelSideL,allVelSideRandL,averTimeL,averVelSideValL,RandVSL,'-S4-ML',-0.025,0.02,2)

