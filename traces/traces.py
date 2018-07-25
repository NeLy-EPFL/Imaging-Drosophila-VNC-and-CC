# -*- coding: utf-8 -*-
"""
Goal:
    This script generates the traces of DR/R%  and optic flow measurements with the events detected on top of them.
    
Method:
    The script opens the data from the dictionary computed for the alignment (script P4).
    It generates the first order derivatives of the DR/R%  traces and detects the events based on the threshold.
    It plots the traces and the events detected on top of them as white vertical lines. It can also plot the first order 
    derivative on the traces if the user uncomment the derivative plot lines in the plotFigOpflowDr() function.
"""

import os, os.path
import cPickle as pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import sys
import pandas as pd

#Paths

# MDN -  uncomment this section and comment the other paths sections
# dataDir = 'YOUR_PATH_TO_MDN_EXPERIMENT'
# Percentile = 97.5
# boolExp = 0
# boolComp = 1

#A1 -  uncomment this section and comment the other paths sections
dataDir = 'YOUR_PATH_TO_A1_EXPERIMENT'
Percentile = 90
boolExp = 0
boolComp = 0
timeWindowA1 = 0.25

#MAN -  uncomment this section and comment the other paths sections
# dataDir = 'YOUR_PATH_TO_MAN_EXPERIMENT'
# Percentile = 97.5
# boolExp = 0
# boolComp = 1


###### Horizontal and coronal section of VNC
#Horizontal section VNC - uncomment this section and comment the other paths sections
# dataDir = 'YOUR_PATH_TO_HORIZONTAL_IMAGING_EXPERIMENT'
# boolExp = 1

# Coronal section VNC -  uncomment this section and comment the other paths sections
# dataDir = 'YOUR_PATH_TO_CORONAL_IMAGING_EXPERIMENT'
# boolExp = 2

#Cervical connective -  uncomment this section and comment the other paths sections
# dataDir = 'YOUR_PATH_TO_CORONAL_IMAGING_EXPERIMENT'
# boolExp = 2


windowLARnotEqual = 1
window = 10

print dataDir
outDir= dataDir + '/output/'


def OpenDicDataFigHor():
	"""
	Opens the data from the dictionnary for one experiment and extracts DF/F% and optic flow data.
	"""
	if os.path.exists(outDir+"DicDataAnalysisPAB.p"):
		DicData = pickle.load( open( outDir + "DicDataAnalysisPAB.p", "rb" ) )

		frameCntr = DicData['frameCntr']
		velForw = DicData['velForw']
		velSide = DicData['velSide']
		velTurn = DicData['velTurn']
		timeSec = DicData['timeSec']
		cam_systime = DicData['cam_systime']
		AddValues = DicData['AddValues']
		puffSampled = DicData['puffSampled']
		Walking = DicData['walking']
		Grooming = DicData['grooming']

		vidStartSysTime = AddValues[0]
		vidStopSysTime = AddValues[1]
		stepOnIdx = AddValues[2]

		vidStopTime = vidStopSysTime - vidStartSysTime

	else :
		print ("File not found - Data not analysed yet - please go to 4th part of data analysis")
		sys.exit(0) 

	return velForw, velSide, velTurn, timeSec, vidStopTime, Walking, Grooming

def OpenDicDataFigCor():
	"""
	Opens the data from the dictionnary for one experiment and extracts DF/F% and optic flow data.
	"""
	if os.path.exists(outDir+"DicDataAnalysisPAB.p"):
		DicData = pickle.load( open( outDir + "DicDataAnalysisPAB.p", "rb" ) )

		frameCntr = DicData['frameCntr']
		velForw = DicData['velForw']
		velSide = DicData['velSide']
		velTurn = DicData['velTurn']
		timeSec = DicData['timeSec']
		Walking = DicData['walking']
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

	return velForw, velSide, velTurn, timeSec, vidStopTime, Walking


def OpenDicData():
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

def getYAxisMinMax(OpFlow):
	"""
	Finds max and min of optic flow y axis
	"""
	ylimMin = np.nanmin(OpFlow)
	ylimMax = np.nanmax(OpFlow)

	return ylimMin, ylimMax

def getYAxisMinMaxGC(LGC,RGC):
	"""
	Finds max and min of DR/R% y axis
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

	return ylimMinGC, ylimMaxGC

def adjustYLimDR(OylimMaxGC,ylimMinGC):

	if OylimMaxGC < 100:
		ylimMaxGC = 100
		boolA = 1
	elif OylimMaxGC < 150:
		ylimMaxGC = 150
		boolA = 2
	elif OylimMaxGC<200 : 
		ylimMaxGC = 200
		boolA = 3
	else : 
		ylimMaxGC = 250
		boolA = 3

	if ylimMinGC < 0:
		if boolA != 1:
			ylimMinGCN = -50
		else : 
			ylimMinGCN = 0
	elif ylimMinGC < (-50):
		ylimMinGCN = -100
	else : 
		ylimMinGCN = 0

	ylimMinGC = ylimMinGCN

	return ylimMinGC, ylimMaxGC


def plotFigOpflowDrFigCor(i,interval,timeSec,velForw,ylimMinAP,ylimMaxAP,velSide,ylimMinML,ylimMaxML,velTurn,ylimMinYaw,ylimMaxYaw,Walking,ylimMinWalk, ylimMaxWalk):

	fig = plt.figure(facecolor='white')
	gs = gridspec.GridSpec(4, 10)

	ax20 = plt.subplot(gs[0,1:])
	ax20.spines['bottom'].set_visible(True)
	ax20.spines['top'].set_visible(False)
	ax20.spines['left'].set_visible(True)
	ax20.spines['right'].set_visible(False)
	ax20.xaxis.label.set_color('black')
	ax20.yaxis.label.set_color('black')
	ax20.tick_params(axis='x', colors='black',direction='out',top='off')
	ax20.tick_params(axis='y', colors='black',direction='out',right='off')    
	ax20.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
	plt.ylim(ylimMinWalk, ylimMaxWalk)
	yaxisGCL = np.array([round(ylimMinWalk,0),round((ylimMinWalk+(ylimMaxWalk-ylimMinWalk)/2),0),round(ylimMaxWalk,0)])
	ax20.yaxis.set_ticks(yaxisGCL)
	labels = [item.get_text() for item in ax20.get_xticklabels()]
	empty_string_labels = ['']*len(labels)
	ax20.set_xticklabels(empty_string_labels)

	plt.axhline(0,linestyle='dashed',color='gray',linewidth=0.5)
	plt.axvline(0,linestyle='dashed',color='gray',linewidth=0.5)
	plt.plot(timeSec, Walking, label = "L", color='y',linewidth=1)
	plt.xlim(i, i+interval)
	plt.ylabel(unicode('Walking\n$\Delta$F/F\n(%)'), fontsize=10,color='black') 

	ax21=plt.subplot(gs[0,0])
	ax21.tick_params(axis='x', colors='black',direction='out',top='off',bottom='off')
	ax21.tick_params(axis='y', colors='black',direction='out',right='off',left='off') 
	ax21.set_xticklabels([])
	ax21.set_yticklabels([])
	ax21.spines['bottom'].set_visible(False)
	ax21.spines['top'].set_visible(False)
	ax21.spines['left'].set_visible(False)
	ax21.spines['right'].set_visible(False)

	ax14 = plt.subplot(gs[1,1:])
	ax14.spines['bottom'].set_visible(True)
	ax14.spines['top'].set_visible(False)
	ax14.spines['left'].set_visible(True)
	ax14.spines['right'].set_visible(False)
	ax14.yaxis.label.set_color('black')
	ax14.tick_params(axis='x', colors='black',direction='out',top='off')
	ax14.tick_params(axis='y', colors='black',direction='out',right='off')    
	ax14.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

	labels = [item.get_text() for item in ax14.get_xticklabels()]
	empty_string_labels = ['']*len(labels)
	ax14.set_xticklabels(empty_string_labels)

	plt.axhline(0, linestyle='dashed',color='gray',linewidth=0.5)
	plt.axvline(0, linestyle='dashed',color='gray',linewidth=0.5)
	plt.plot(timeSec,velForw, label = "AP", color='r',linewidth=1)
	plt.xlim(i, i+interval)
	plt.ylim(ylimMinAP, ylimMaxAP)
	yaxisAP = np.array([round(ylimMinAP,2),round((ylimMinAP+(ylimMaxAP-ylimMinAP)/2),2),round(ylimMaxAP,2)])
	ax14.yaxis.set_ticks(yaxisAP)
	plt.ylabel('Rot./s', fontsize=10)


	ax15=plt.subplot(gs[1,0]) 
	ax15.tick_params(axis='x', colors='black',direction='out',top='off',bottom='off')
	ax15.tick_params(axis='y', colors='black',direction='out',right='off',left='off') 
	ax15.set_xticklabels([])
	ax15.set_yticklabels([])
	ax15.spines['top'].set_visible(False)
	ax15.spines['bottom'].set_visible(False)
	ax15.spines['left'].set_visible(False)
	ax15.spines['right'].set_visible(False)
	plt.ylabel('AP', fontsize=10,color='r')

	ax16 = plt.subplot(gs[2,1:])
	ax16.spines['bottom'].set_visible(True)
	ax16.spines['top'].set_visible(False)
	ax16.spines['left'].set_visible(True)
	ax16.spines['right'].set_visible(False)
	ax16.yaxis.label.set_color('black')
	ax16.tick_params(axis='x', colors='black',direction='out',top='off')
	ax16.tick_params(axis='y', colors='black',direction='out',right='off')    
	ax16.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
	plt.ylim(ylimMinML, ylimMaxML)
	yaxisML = np.array([round(ylimMinML,3),round((ylimMinML+(ylimMaxML-ylimMinML)/2),3),round(ylimMaxML,3)])
	ax16.yaxis.set_ticks(yaxisML)

	labels = [item.get_text() for item in ax16.get_xticklabels()]
	empty_string_labels = ['']*len(labels)
	ax16.set_xticklabels(empty_string_labels)

	plt.axhline(0,linestyle='dashed',color='gray',linewidth=0.5)
	plt.axvline(0,linestyle='dashed',color='gray',linewidth=0.5)
	plt.plot(timeSec,velSide, label = "ML", color='c',linewidth=1)
	plt.xlim(i, i+interval)
	plt.ylabel('Rot./s', fontsize=10)

	ax17=plt.subplot(gs[2,0]) 
	ax17.tick_params(axis='x', colors='black',direction='out',top='off',bottom='off')
	ax17.tick_params(axis='y', colors='black',direction='out',right='off',left='off') 
	ax17.set_xticklabels([])
	ax17.set_yticklabels([])
	ax17.spines['top'].set_visible(False)
	ax17.spines['bottom'].set_visible(False)
	ax17.spines['left'].set_visible(False)
	ax17.spines['right'].set_visible(False)
	plt.ylabel('ML', fontsize=10, color='c')

	ax18 = plt.subplot(gs[3,1:])
	ax18.spines['bottom'].set_color('black')
	ax18.spines['top'].set_visible(False)
	ax18.spines['left'].set_visible(True)
	ax18.spines['right'].set_visible(False)
	ax18.xaxis.label.set_color('black')
	ax18.yaxis.label.set_color('black')
	ax18.tick_params(axis='x', colors='black',direction='out',top='off')
	ax18.tick_params(axis='y', colors='black',direction='out',right='off')    
	ax18.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	plt.ylim(ylimMinYaw, ylimMaxYaw)
	yaxisYaw = np.array([round(ylimMinYaw,2),round((ylimMinYaw+(ylimMaxYaw-ylimMinYaw)/2),2),round(ylimMaxYaw,2)])
	ax18.yaxis.set_ticks(yaxisYaw)
	plt.xlabel('Time (s)', fontsize=10)

	plt.axhline(0,linestyle='dashed',color='gray',linewidth=0.5)
	plt.axvline(0,linestyle='dashed',color='gray',linewidth=0.5)
	plt.plot(timeSec,velTurn, label = "Yaw", color='g',linewidth=1)
	plt.xlim(i, i+interval)
	plt.ylabel('Rot./s', fontsize=10)

	ax19=plt.subplot(gs[3,0])
	ax19.tick_params(axis='x', colors='black',direction='out',top='off',bottom='off')
	ax19.tick_params(axis='y', colors='black',direction='out',right='off',left='off') 
	ax19.set_xticklabels([])
	ax19.set_yticklabels([])
	ax19.spines['top'].set_visible(False)
	ax19.spines['bottom'].set_visible(False)
	ax19.spines['left'].set_visible(False)
	ax19.spines['right'].set_visible(False)
	plt.ylabel('Yaw', fontsize=10,color='g')
  
	plt.savefig(outDir + 'Overview'+ str(i) + '-' + str(i+interval) + '-epsTest-Walk.eps',format='eps', facecolor=fig.get_facecolor(), edgecolor='none', transparent=True) #bbox_inches='tight', 
	plt.close(fig)

	return

def plotFigOpflowDrFigHor(i,interval,timeSec,velForw,ylimMinAP,ylimMaxAP,velSide,ylimMinML,ylimMaxML,velTurn,ylimMinYaw,ylimMaxYaw,Walking,Grooming,yFluoMax,yFluoMin):

	fig = plt.figure(facecolor='white')
	gs = gridspec.GridSpec(5, 10)

	ax14 = plt.subplot(gs[2,1:])
	ax14.spines['bottom'].set_visible(True)
	ax14.spines['top'].set_visible(False)
	ax14.spines['left'].set_visible(True)
	ax14.spines['right'].set_visible(False)
	ax14.yaxis.label.set_color('black')
	ax14.tick_params(axis='x', colors='black',direction='out',top='off')
	ax14.tick_params(axis='y', colors='black',direction='out',right='off')    
	ax14.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

	labels = [item.get_text() for item in ax14.get_xticklabels()]
	empty_string_labels = ['']*len(labels)
	ax14.set_xticklabels(empty_string_labels)

	plt.axhline(0, linestyle='dashed',color='gray',linewidth=0.5)
	plt.axvline(0, linestyle='dashed',color='gray',linewidth=0.5)
	plt.plot(timeSec,velForw, label = "AP", color='r',linewidth=1)
	plt.xlim(i, i+interval)
	plt.ylim(ylimMinAP, ylimMaxAP)
	yaxisAP = np.array([round(ylimMinAP,2),round((ylimMinAP+(ylimMaxAP-ylimMinAP)/2),2),round(ylimMaxAP,2)])
	ax14.yaxis.set_ticks(yaxisAP)
	plt.ylabel('Rot./s', fontsize=10)


	ax15=plt.subplot(gs[2,0]) 
	ax15.tick_params(axis='x', colors='black',direction='out',top='off',bottom='off')
	ax15.tick_params(axis='y', colors='black',direction='out',right='off',left='off') 
	ax15.set_xticklabels([])
	ax15.set_yticklabels([])
	ax15.spines['top'].set_visible(False)
	ax15.spines['bottom'].set_visible(False)
	ax15.spines['left'].set_visible(False)
	ax15.spines['right'].set_visible(False)
	plt.ylabel('AP', fontsize=10,color='r')

	ax16 = plt.subplot(gs[3,1:])
	ax16.spines['bottom'].set_visible(True)
	ax16.spines['top'].set_visible(False)
	ax16.spines['left'].set_visible(True)
	ax16.spines['right'].set_visible(False)
	ax16.yaxis.label.set_color('black')
	ax16.tick_params(axis='x', colors='black',direction='out',top='off')
	ax16.tick_params(axis='y', colors='black',direction='out',right='off')    
	ax16.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
	plt.ylim(ylimMinML, ylimMaxML)
	yaxisML = np.array([round(ylimMinML,3),round((ylimMinML+(ylimMaxML-ylimMinML)/2),3),round(ylimMaxML,3)])
	ax16.yaxis.set_ticks(yaxisML)

	labels = [item.get_text() for item in ax16.get_xticklabels()]
	empty_string_labels = ['']*len(labels)
	ax16.set_xticklabels(empty_string_labels)

	plt.axhline(0,linestyle='dashed',color='gray',linewidth=0.5)
	plt.axvline(0,linestyle='dashed',color='gray',linewidth=0.5)
	plt.plot(timeSec,velSide, label = "ML", color='c',linewidth=1)
	plt.xlim(i, i+interval)
	plt.ylabel('Rot./s', fontsize=10)

	ax17=plt.subplot(gs[3,0]) 
	ax17.tick_params(axis='x', colors='black',direction='out',top='off',bottom='off')
	ax17.tick_params(axis='y', colors='black',direction='out',right='off',left='off') 
	ax17.set_xticklabels([])
	ax17.set_yticklabels([])
	ax17.spines['top'].set_visible(False)
	ax17.spines['bottom'].set_visible(False)
	ax17.spines['left'].set_visible(False)
	ax17.spines['right'].set_visible(False)
	plt.ylabel('ML', fontsize=10, color='c')

	ax18 = plt.subplot(gs[4,1:])
	ax18.spines['bottom'].set_color('black')
	ax18.spines['top'].set_visible(False)
	ax18.spines['left'].set_visible(True)
	ax18.spines['right'].set_visible(False)
	ax18.xaxis.label.set_color('black')
	ax18.yaxis.label.set_color('black')
	ax18.tick_params(axis='x', colors='black',direction='out',top='off')
	ax18.tick_params(axis='y', colors='black',direction='out',right='off')    
	ax18.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	plt.ylim(ylimMinYaw, ylimMaxYaw)
	yaxisYaw = np.array([round(ylimMinYaw,2),round((ylimMinYaw+(ylimMaxYaw-ylimMinYaw)/2),2),round(ylimMaxYaw,2)])
	ax18.yaxis.set_ticks(yaxisYaw)
	plt.xlabel('Time (s)', fontsize=10)

	plt.axhline(0,linestyle='dashed',color='gray',linewidth=0.5)
	plt.axvline(0,linestyle='dashed',color='gray',linewidth=0.5)
	plt.plot(timeSec,velTurn, label = "Yaw", color='g',linewidth=1)
	plt.xlim(i, i+interval)
	plt.ylabel('Rot./s', fontsize=10)

	ax19=plt.subplot(gs[4,0])
	ax19.tick_params(axis='x', colors='black',direction='out',top='off',bottom='off')
	ax19.tick_params(axis='y', colors='black',direction='out',right='off',left='off') 
	ax19.set_xticklabels([])
	ax19.set_yticklabels([])
	ax19.spines['top'].set_visible(False)
	ax19.spines['bottom'].set_visible(False)
	ax19.spines['left'].set_visible(False)
	ax19.spines['right'].set_visible(False)
	plt.ylabel('Yaw', fontsize=10,color='g')

	ax20 = plt.subplot(gs[0,1:])
	ax20.spines['bottom'].set_visible(True)
	ax20.spines['top'].set_visible(False)
	ax20.spines['left'].set_visible(True)
	ax20.spines['right'].set_visible(False)
	ax20.xaxis.label.set_color('black')
	ax20.yaxis.label.set_color('black')
	ax20.tick_params(axis='x', colors='black',direction='out',top='off')
	ax20.tick_params(axis='y', colors='black',direction='out',right='off')    
	ax20.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
	plt.ylim(yFluoMin, yFluoMax)
	yaxisGCL = np.array([round(yFluoMin,0),round((yFluoMin+(yFluoMax-yFluoMin)/2),0),round(yFluoMax,0)])
	ax20.yaxis.set_ticks(yaxisGCL)
	labels = [item.get_text() for item in ax20.get_xticklabels()]
	empty_string_labels = ['']*len(labels)
	ax20.set_xticklabels(empty_string_labels)

	plt.axhline(0,linestyle='dashed',color='gray',linewidth=0.5)
	plt.axvline(0,linestyle='dashed',color='gray',linewidth=0.5)
	plt.plot(timeSec, Walking, label = "L", color='y',linewidth=1)
	plt.xlim(i, i+interval)
	plt.ylabel(unicode('Walking\n$\Delta$F/F\n(%)'), fontsize=10,color='black') 

	ax21=plt.subplot(gs[0,0])
	ax21.tick_params(axis='x', colors='black',direction='out',top='off',bottom='off')
	ax21.tick_params(axis='y', colors='black',direction='out',right='off',left='off') 
	ax21.set_xticklabels([])
	ax21.set_yticklabels([])
	ax21.spines['bottom'].set_visible(False)
	ax21.spines['top'].set_visible(False)
	ax21.spines['left'].set_visible(False)
	ax21.spines['right'].set_visible(False)

	ax22 = plt.subplot(gs[1,1:])
	ax22.spines['bottom'].set_visible(True)
	ax22.spines['top'].set_visible(False)
	ax22.spines['left'].set_visible(True)
	ax22.spines['right'].set_visible(False)
	ax22.tick_params(axis='x', colors='black',direction='out',top='off')
	ax22.tick_params(axis='y', colors='black',direction='out',right='off')    
	ax22.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
	plt.ylim(yFluoMin, yFluoMax)
	yaxisGRR = np.array([round(yFluoMin,0),round((yFluoMin+(yFluoMax-yFluoMin)/2),0),round(yFluoMax,0)])
	ax22.yaxis.set_ticks(yaxisGRR)

	labels = [item.get_text() for item in ax22.get_xticklabels()]
	empty_string_labels = ['']*len(labels)
	ax22.set_xticklabels(empty_string_labels)

	plt.axhline(0,linestyle='dashed',color='gray',linewidth=0.5)
	plt.axvline(0,linestyle='dashed',color='gray',linewidth=0.5)
	plt.plot(timeSec, Grooming, label = "R", color='y',linewidth=1)
	plt.xlim(i, i+interval)
	plt.ylabel(unicode('Grooming\n$\Delta$F/F\n(%)'), fontsize=10)

	ax23=plt.subplot(gs[1,0])
	ax23.tick_params(axis='x', colors='black',direction='out',top='off',bottom='off')
	ax23.tick_params(axis='y', colors='black',direction='out',right='off',left='off') 
	ax23.set_xticklabels([])
	ax23.set_yticklabels([])
	ax23.spines['top'].set_visible(False)
	ax23.spines['bottom'].set_visible(False)
	ax23.spines['left'].set_visible(False)
	ax23.spines['right'].set_visible(False)

	plt.savefig(outDir + 'Overview'+ str(i) + '-' + str(i+interval) + '-eps-Gr-Wa.eps',format='eps', facecolor=fig.get_facecolor(), edgecolor='none', transparent=True) #bbox_inches='tight', 
	plt.close(fig)

	return


def plotFigOpflowDr(eventIdxTotL,eventIdxTotR,i,interval,timeSec,velForw,ylimMinAP,ylimMaxAP,velSide,ylimMinML,ylimMaxML,velTurn,ylimMinYaw,ylimMaxYaw,L_DR,ylimMinGC,ylimMaxGC,R_DR):

	fig = plt.figure(facecolor='white')
	gs = gridspec.GridSpec(5, 10)

	ax14 = plt.subplot(gs[2,1:])
	ax14.spines['bottom'].set_visible(True)
	ax14.spines['top'].set_visible(False)
	ax14.spines['left'].set_visible(True)
	ax14.spines['right'].set_visible(False)
	ax14.yaxis.label.set_color('black')
	ax14.tick_params(axis='x', colors='black',direction='out',top='off')
	ax14.tick_params(axis='y', colors='black',direction='out',right='off')    
	ax14.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

	labels = [item.get_text() for item in ax14.get_xticklabels()]
	empty_string_labels = ['']*len(labels)
	ax14.set_xticklabels(empty_string_labels)

	plt.axhline(0, linestyle='dashed',color='gray',linewidth=0.5)
	plt.axvline(0, linestyle='dashed',color='gray',linewidth=0.5)
	for l in eventIdxTotL:
		ax14.axvline(timeSec[l], linestyle='dashed',color='black',linewidth=1)
	for l in eventIdxTotR:
		ax14.axvline(timeSec[l], linestyle='dashed',color='black',linewidth=1)
	plt.plot(timeSec,velForw, label = "AP", color='r',linewidth=1)
	plt.xlim(i, i+interval)
	plt.ylim(ylimMinAP, ylimMaxAP)
	yaxisAP = np.array([round(ylimMinAP,2),round((ylimMinAP+(ylimMaxAP-ylimMinAP)/2),2),round(ylimMaxAP,2)])
	ax14.yaxis.set_ticks(yaxisAP)
	plt.ylabel('Rot./s', fontsize=10)


	ax15=plt.subplot(gs[2,0]) 
	ax15.tick_params(axis='x', colors='black',direction='out',top='off',bottom='off')
	ax15.tick_params(axis='y', colors='black',direction='out',right='off',left='off') 
	ax15.set_xticklabels([])
	ax15.set_yticklabels([])
	ax15.spines['top'].set_visible(False)
	ax15.spines['bottom'].set_visible(False)
	ax15.spines['left'].set_visible(False)
	ax15.spines['right'].set_visible(False)
	plt.ylabel('AP', fontsize=10,color='r')

	ax16 = plt.subplot(gs[3,1:])
	ax16.spines['bottom'].set_visible(True)
	ax16.spines['top'].set_visible(False)
	ax16.spines['left'].set_visible(True)
	ax16.spines['right'].set_visible(False)
	ax16.yaxis.label.set_color('black')
	ax16.tick_params(axis='x', colors='black',direction='out',top='off')
	ax16.tick_params(axis='y', colors='black',direction='out',right='off')    
	ax16.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
	plt.ylim(ylimMinML, ylimMaxML)
	yaxisML = np.array([round(ylimMinML,3),round((ylimMinML+(ylimMaxML-ylimMinML)/2),3),round(ylimMaxML,3)])
	ax16.yaxis.set_ticks(yaxisML)

	labels = [item.get_text() for item in ax16.get_xticklabels()]
	empty_string_labels = ['']*len(labels)
	ax16.set_xticklabels(empty_string_labels)

	plt.axhline(0,linestyle='dashed',color='gray',linewidth=0.5)
	plt.axvline(0,linestyle='dashed',color='gray',linewidth=0.5)
	for l in eventIdxTotL:
		ax16.axvline(timeSec[l], linestyle='dashed',color='black',linewidth=1)
	for l in eventIdxTotR:
		ax16.axvline(timeSec[l], linestyle='dashed',color='black',linewidth=1)
	plt.plot(timeSec,velSide, label = "ML", color='c',linewidth=1)
	plt.xlim(i, i+interval)
	plt.ylabel('Rot./s', fontsize=10)

	ax17=plt.subplot(gs[3,0]) 
	ax17.tick_params(axis='x', colors='black',direction='out',top='off',bottom='off')
	ax17.tick_params(axis='y', colors='black',direction='out',right='off',left='off') 
	ax17.set_xticklabels([])
	ax17.set_yticklabels([])
	ax17.spines['top'].set_visible(False)
	ax17.spines['bottom'].set_visible(False)
	ax17.spines['left'].set_visible(False)
	ax17.spines['right'].set_visible(False)
	plt.ylabel('ML', fontsize=10, color='c')

	ax18 = plt.subplot(gs[4,1:])
	ax18.spines['bottom'].set_color('black')
	ax18.spines['top'].set_visible(False)
	ax18.spines['left'].set_visible(True)
	ax18.spines['right'].set_visible(False)
	ax18.xaxis.label.set_color('black')
	ax18.yaxis.label.set_color('black')
	ax18.tick_params(axis='x', colors='black',direction='out',top='off')
	ax18.tick_params(axis='y', colors='black',direction='out',right='off')    
	ax18.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	plt.ylim(ylimMinYaw, ylimMaxYaw)
	yaxisYaw = np.array([round(ylimMinYaw,2),round((ylimMinYaw+(ylimMaxYaw-ylimMinYaw)/2),2),round(ylimMaxYaw,2)])
	ax18.yaxis.set_ticks(yaxisYaw)
	plt.xlabel('Time (s)', fontsize=10)

	plt.axhline(0,linestyle='dashed',color='gray',linewidth=0.5)
	plt.axvline(0,linestyle='dashed',color='gray',linewidth=0.5)
	for l in eventIdxTotL:
		ax18.axvline(timeSec[l], linestyle='dashed',color='black',linewidth=1)
	for l in eventIdxTotR:
		ax18.axvline(timeSec[l], linestyle='dashed',color='black',linewidth=1)
	plt.plot(timeSec,velTurn, label = "Yaw", color='g',linewidth=1)
	plt.xlim(i, i+interval)
	plt.ylabel('Rot./s', fontsize=10)

	ax19=plt.subplot(gs[4,0])
	ax19.tick_params(axis='x', colors='black',direction='out',top='off',bottom='off')
	ax19.tick_params(axis='y', colors='black',direction='out',right='off',left='off') 
	ax19.set_xticklabels([])
	ax19.set_yticklabels([])
	ax19.spines['top'].set_visible(False)
	ax19.spines['bottom'].set_visible(False)
	ax19.spines['left'].set_visible(False)
	ax19.spines['right'].set_visible(False)
	plt.ylabel('Yaw', fontsize=10,color='g')

	ax20 = plt.subplot(gs[0,1:])
	ax20.spines['bottom'].set_visible(True)
	ax20.spines['top'].set_visible(False)
	ax20.spines['left'].set_visible(True)
	ax20.spines['right'].set_visible(False)
	ax20.xaxis.label.set_color('black')
	ax20.yaxis.label.set_color('black')
	ax20.tick_params(axis='x', colors='black',direction='out',top='off')
	ax20.tick_params(axis='y', colors='black',direction='out',right='off')    
	ax20.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
	plt.ylim(ylimMinGC, ylimMaxGC)
	yaxisGCL = np.array([round(ylimMinGC,0),round((ylimMinGC+(ylimMaxGC-ylimMinGC)/2),0),round(ylimMaxGC,0)])
	ax20.yaxis.set_ticks(yaxisGCL)

	labels = [item.get_text() for item in ax20.get_xticklabels()]
	empty_string_labels = ['']*len(labels)
	ax20.set_xticklabels(empty_string_labels)

	plt.axhline(0,linestyle='dashed',color='gray',linewidth=0.5)
	plt.axvline(0,linestyle='dashed',color='gray',linewidth=0.5)
	for l in eventIdxTotL:
		ax20.axvline(timeSec[l], linestyle='dashed',color='black',linewidth=1)
	plt.plot(timeSec, L_DR, label = "L", color='y',linewidth=1)
	plt.xlim(i, i+interval)
	plt.ylabel(unicode('Left\n$\Delta$R/R\n(%)'), fontsize=10,color='black') 

	ax21=plt.subplot(gs[0,0])
	ax21.tick_params(axis='x', colors='black',direction='out',top='off',bottom='off')
	ax21.tick_params(axis='y', colors='black',direction='out',right='off',left='off') 
	ax21.set_xticklabels([])
	ax21.set_yticklabels([])
	ax21.spines['bottom'].set_visible(False)
	ax21.spines['top'].set_visible(False)
	ax21.spines['left'].set_visible(False)
	ax21.spines['right'].set_visible(False)


	ax22 = plt.subplot(gs[1,1:])
	ax22.spines['bottom'].set_visible(True)
	ax22.spines['top'].set_visible(False)
	ax22.spines['left'].set_visible(True)
	ax22.spines['right'].set_visible(False)
	ax22.tick_params(axis='x', colors='black',direction='out',top='off')
	ax22.tick_params(axis='y', colors='black',direction='out',right='off')    
	ax22.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
	plt.ylim(ylimMinGC, ylimMaxGC)
	yaxisGCL = np.array([round(ylimMinGC,0),round((ylimMinGC+(ylimMaxGC-ylimMinGC)/2),0),round(ylimMaxGC,0)])
	ax22.yaxis.set_ticks(yaxisGCL)

	labels = [item.get_text() for item in ax22.get_xticklabels()]
	empty_string_labels = ['']*len(labels)
	ax22.set_xticklabels(empty_string_labels)

	plt.axhline(0,linestyle='dashed',color='gray',linewidth=0.5)
	plt.axvline(0,linestyle='dashed',color='gray',linewidth=0.5)
	for l in eventIdxTotR:
		ax22.axvline(timeSec[l], linestyle='dashed',color='black',linewidth=1)
	plt.plot(timeSec, R_DR, label = "R", color='y',linewidth=1)
	plt.xlim(i, i+interval)
	plt.ylabel(unicode('Right\n$\Delta$R/R\n(%)'), fontsize=10)

	ax23=plt.subplot(gs[1,0])
	ax23.tick_params(axis='x', colors='black',direction='out',top='off',bottom='off')
	ax23.tick_params(axis='y', colors='black',direction='out',right='off',left='off') 
	ax23.set_xticklabels([])
	ax23.set_yticklabels([])
	ax23.spines['top'].set_visible(False)
	ax23.spines['bottom'].set_visible(False)
	ax23.spines['left'].set_visible(False)
	ax23.spines['right'].set_visible(False)
	     
	plt.savefig(outDir + 'Overview'+ str(i) + '-' + str(i+interval) + '-eps-Event.eps',format='eps', facecolor=fig.get_facecolor(), edgecolor='none', transparent=True) #bbox_inches='tight', 
	plt.close(fig)

	return

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

def complementaryDetectionLAR(eventIdxTotL,eventIdxTotR,windowLARNotEqual,timeFactGlob):
	"""
	Detects if events detected for left neurons have a complementary event time point detected in right neuron within a time window of 2 seconds around
	the event. If not, left event time point is added to right event time point and vice versa.
	"""
	timeFact = int(windowLARNotEqual*timeFactGlob)
	eventLCopy = eventIdxTotL[:]
	eventRCopy = eventIdxTotR[:]
	eventInLeftNotInR = list(set(eventLCopy)-set(eventRCopy))
	eventInRightNotInL = list(set(eventRCopy)-set(eventLCopy))
	for j in eventInLeftNotInR:
		testList = list(range(j-timeFact,j+timeFact))
		interLAR = list(set(eventRCopy).intersection(testList))
		if len(interLAR)==0:
			eventIdxTotR.append(j)
	for m in eventInRightNotInL:
		testListR = list(range(m-timeFact,m+timeFact))
		interLARR = list(set(eventLCopy).intersection(testListR))
		if len(interLARR)==0:
			eventIdxTotL.append(m)
	eventIdxTotL.sort()
	eventIdxTotR.sort()

	return eventIdxTotL, eventIdxTotR


def getIndxThresh(deriv,thresh):
	"""
	Finds the indexes of the derivative when it crosses the threshold value.
	"""
	IndxThresh = []
	for i in range(len(deriv)):
		if deriv[i] > thresh:
			if deriv[i+1]>thresh:
				if deriv[i-1]<thresh:
					IndxThresh.append(i)

	return IndxThresh

def getEventIdx(idxThresh,deriv,timeFact,window):
	"""
	Finds the indexes of the 0 crossing of the derivative occuring just before the idx found in getIndxThresh.
	"""
	eventIdxTot = []
	timeFrame = int(window*timeFact)
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
	
	return eventIdxTot[0]

def A1EventSeparated(eventIdxTotL,eventIdxTotR,timeFact):
	"""
	Separates events detected between left events only, right events only or left and right events happening in the same time window of 0.25 second.
	"""
	LFromLAREventExp = []
	RFromLAREventExp = []
	timeFactExp = int(timeWindowA1*timeFact)
	eventLCopy = eventIdxTotL[:]
	eventRCopy = eventIdxTotR[:]
	for j in eventIdxTotL:
		IdxListToMatch = list(np.arange(j-timeFactExp,j+timeFactExp,1))
		LARDetected = list(set(eventIdxTotR).intersection(set(IdxListToMatch)))
		if len(LARDetected)!=0:
			LFromLAREventExp.append(j)
			RFromLAREventExp+=LARDetected
	LEventExp = list(set(eventIdxTotL)-set(LFromLAREventExp))
	REventExp = list(set(eventIdxTotR)-set(RFromLAREventExp))

	return LFromLAREventExp, RFromLAREventExp ,LEventExp, REventExp 

# ********** MAIN **************

# ### 
if boolExp == 0:
	L_DR, R_DR, velForw, velSide, velTurn, timeSec, vidStopTime = OpenDicData()

	timeFact = len(timeSec)/vidStopTime

	deriv_L_DR, interv = getDeriv(L_DR,timeSec)
	deriv_R_DR, interv = getDeriv(R_DR,timeSec)

	DerivLeftNoNan = [x for x in deriv_L_DR if pd.isnull(x)==False]
	DerivRightNoNan = [x for x in deriv_R_DR if pd.isnull(x)==False]

	percentL = np.percentile(DerivLeftNoNan,Percentile)
	percentR = np.percentile(DerivRightNoNan,Percentile)

	IndxThreshTotL = getIndxThresh(deriv_L_DR,percentL)
	IndxThreshTotR = getIndxThresh(deriv_R_DR,percentR)

	eventIdxTotL = getEventIdx(IndxThreshTotL,deriv_L_DR,timeFact,window)
	eventIdxTotR = getEventIdx(IndxThreshTotR,deriv_R_DR,timeFact,window)

	if boolComp == 1:
		eventIdxTotL,eventIdxTotR = complementaryDetectionLAR(eventIdxTotL,eventIdxTotR,windowLARnotEqual,timeFact)
	elif boolComp == 0:
		LFromLAREventGlob, RFromLAREventGlob ,eventIdxTotL, eventIdxTotR = A1EventSeparated(eventIdxTotL,eventIdxTotR,timeFact)

	ylimMinGC, ylimMaxGC = getYAxisMinMaxGC(L_DR,R_DR)
	ylimMinAP, ylimMaxAP = getYAxisMinMax(velForw)
	ylimMinML, ylimMaxML = getYAxisMinMax(velSide)
	ylimMinYaw, ylimMaxYaw = getYAxisMinMax(velTurn)

	ylimMinGC, ylimMaxGC = adjustYLimDR(ylimMaxGC,ylimMinGC)

	xlimMin=0
	xlimMax=xlimMin+vidStopTime 
	interval=60

	for i in range(xlimMin,int(xlimMax),interval):
		plotFigOpflowDr(eventIdxTotL,eventIdxTotR,i,interval,timeSec,velForw,ylimMinAP,ylimMaxAP,velSide,ylimMinML,ylimMaxML,velTurn,ylimMinYaw,ylimMaxYaw,L_DR,ylimMinGC,ylimMaxGC,R_DR)

# ### Horizontal VNC 
elif boolExp == 1:
	velForw, velSide, velTurn, timeSec, vidStopTime,Walking, Grooming = OpenDicDataFigHor()
	ylimMinAP, ylimMaxAP = getYAxisMinMax(velForw)
	ylimMinML, ylimMaxML = getYAxisMinMax(velSide)
	ylimMinYaw, ylimMaxYaw = getYAxisMinMax(velTurn)
	ylimMinWalk, ylimMaxWalk = getYAxisMinMax(Walking)
	ylimMinGr, ylimMaxGr = getYAxisMinMax(Grooming)

	if ylimMaxWalk>ylimMaxGr:
		yFluoMax = ylimMaxWalk
	else : 
		yFluoMax = ylimMaxGr

	if ylimMinWalk<ylimMinGr:
		yFluoMin = ylimMinWalk
	else :
		yFluoMin = ylimMinGr

	xlimMin=0
	xlimMax=xlimMin+vidStopTime 
	interval = 60

	for i in range(xlimMin,int(xlimMax),interval):
		plotFigOpflowDrFigHor(i,interval,timeSec,velForw,ylimMinAP,ylimMaxAP,velSide,ylimMinML,ylimMaxML,velTurn,ylimMinYaw,ylimMaxYaw,Walking,Grooming,yFluoMax,yFluoMin)

# ### Coronal VNC or cervical connectives 
elif boolExp == 2:
	velForw, velSide, velTurn, timeSec, vidStopTime,Walking = OpenDicDataFigCor()
	ylimMinAP, ylimMaxAP = getYAxisMinMax(velForw)
	ylimMinML, ylimMaxML = getYAxisMinMax(velSide)
	ylimMinYaw, ylimMaxYaw = getYAxisMinMax(velTurn)
	ylimMinWalk, ylimMaxWalk = getYAxisMinMax(Walking)

	xlimMin=0
	xlimMax=xlimMin+vidStopTime 
	interval=60

	for i in range(xlimMin,int(xlimMax),interval):
		plotFigOpflowDrFigCor(i,interval,timeSec,velForw,ylimMinAP,ylimMaxAP,velSide,ylimMinML,ylimMaxML,velTurn,ylimMinYaw,ylimMaxYaw,Walking,ylimMinWalk, ylimMaxWalk)









