# -*- coding: utf-8 -*-
"""

Goal:
    This script aligns all the data together and stores them in a dictionary before the script that creates the frames for the final movie.
    
Method:
    The fluorescence DR/R*100 data are opened from the txt files and interpolated to 1500 points per seconds. 
    The timing data of the behavior images, fluorescence images and optic flow measurements are opened and aligned to 1500 points per second.
    The optic flow measurements from the sensors are translated into Anterior-Posterior, Medio-Lateral, Yaw rotations per second.
    All the data are stored in a dictionary.
     
"""

from skimage import io
import sys
import cv2
import numpy as np
import os, os.path
import h5py
import pandas as pd
import re
import time
import cPickle as pickle

dataDir = 'YOUR_PATH_TO_EXPERIMENT_FOLDER'

print dataDir
vidDir = dataDir + '/behavior_imgs'
outDir= dataDir + '/output/'
outDirGC6_UsrCorrected = outDir + 'GC6_UsrCorrected/'
outDirGC6_auto = outDir + 'GC6_auto/'
inVidFiles = os.listdir(vidDir)
inFiles = os.listdir(dataDir)

samplingFact = 1500 
windowTotalTimeSec = 10000 
gain0X = 1.49
gain0Y = 1.35
gain1X = 1.43
gain1Y = 1.32
DictDataList = {}

def GetFluoData():
    """
    This function opens the fluorescence values.
    """
    if os.path.exists(outDirGC6_UsrCorrected+'L_GC_tdtom_norm_UsrW.txt'):
        print(">>>Reading GCaMP6s data from manual draw correction")
        fileGC_L = open(outDirGC6_UsrCorrected+'L_GC_tdtom_norm_UsrW.txt')
        fileGC_R = open(outDirGC6_UsrCorrected+'R_GC_tdtom_norm_UsrW.txt')  

    elif os.path.exists(outDirGC6_UsrCorrected+'L_GC_tdtom_norm_Usr.txt'):
        print(">>>Reading GCaMP6s data from manual selection")
        fileGC_L = open(outDirGC6_UsrCorrected+'L_GC_tdtom_norm_Usr.txt')
        fileGC_R = open(outDirGC6_UsrCorrected+'R_GC_tdtom_norm_Usr.txt')  
    else:
        print(">>>Reading GCaMP6s data from auto selection")
        fileGC_L = open(outDirGC6_auto+'L_GC_tdtom_norm.txt')
        fileGC_R = open(outDirGC6_auto+'R_GC_tdtom_norm.txt')

    L_GCamP6_tdtom_norm=[]
    R_GCamP6_tdtom_norm=[]

    string_readGC_L=(fileGC_L.read()).split("\n")
    string_readGC_R=(fileGC_R.read()).split("\n")

    fileGC_L.close()
    fileGC_R.close()

    for i in range(0,len(string_readGC_L)-1): 
        L_GCamP6_tdtom_norm.append(float(string_readGC_L[i]))
        R_GCamP6_tdtom_norm.append(float(string_readGC_R[i]))

    return L_GCamP6_tdtom_norm, R_GCamP6_tdtom_norm

def readH5file():
    """
    This function opens the .h file from ThorSync software that contains the frame counter, step, puff and piezo data.
    Step and frame counter are two list of the same length. Pulse is a signal sent from computer 1 (acquiring cam stamps) 
    to computer 2 saying that the camera is acquiring frames. When the computer receives the signal from pulse, his "step" 
    value will go from 0 to 10 and will stay set to 10 as long as pulse is sent to computer number 2.
    """
    inH5Files = [f for f in inFiles if f[-3:]=='.h5']
    inFileH5 = h5py.File(os.path.join(dataDir,inH5Files[0]), 'r')
    frameCntr = inFileH5['CI']['Frame Counter'][:].squeeze()
    AIDict = {}
    for key in inFileH5['AI'].keys():
       AIDict[key] = inFileH5['AI'][key][:].squeeze()
    piezo = AIDict['Piezo Monitor'] 
    step = AIDict['timeStamp Step']
    puff = AIDict['puff']

    return frameCntr, piezo, step, puff

def StepInfoIdx(step):
    """
    This function creates a list named StepOnOff_Bool_series. This list contains only True or False values at the positions where the step original list had number >9 or <9
    stepOnTrueIdx is a list containing only the index values at which we have a "True" in the list described above.
    """
    x=[]
    i=[]
    stepOnOff_Bool_series = step > 9 
    stepOnTrueIdx = [i for i, x in enumerate(stepOnOff_Bool_series) if x] #find True value indices
    stepOnTrueIdx = np.array(stepOnTrueIdx) 

    return stepOnTrueIdx, stepOnOff_Bool_series

def AdaptDataSampling(stepOnTrueIdx,vidStopTime,frameCntr,samplingFact,puff):
    """
    This function reduces the number of data points used for frameCounter, puff and the stepOnTrue list. The sampling factor is defined globally. 
    The function returns the sampled frame counter and puff lists and the sampled stepOnTrueIdx. 
    As the stepOnTrueIdx is a list of indexes of the step original list, we needed to divide all the values by the length
    adaptor before cutting in the list itself.
    """
    oldStepLength = len(stepOnTrueIdx)
    lengthAdaptor = int(oldStepLength/(vidStopTime*samplingFact))
    for i in range(len(stepOnTrueIdx)):
        stepOnTrueIdx[i]=int(stepOnTrueIdx[i]/lengthAdaptor)

    StepOnIdx = stepOnTrueIdx[0] 
    StepOffIdx = stepOnTrueIdx[-1] 

    puffN = []
    frameCntrN = []
    stepOnN = []
    for j in range(len(stepOnTrueIdx)):
        if j%lengthAdaptor==1:
            stepOnN.append(stepOnTrueIdx[j])
    for l in range(len(frameCntr)):
        if l%lengthAdaptor==1:
            frameCntrN.append(frameCntr[l])
            puffN.append(puff[l])
    frameCntr = frameCntrN
    stepOnTrueIdx = stepOnN

    return stepOnTrueIdx, frameCntr, StepOnIdx, StepOffIdx, lengthAdaptor, oldStepLength, puffN

def OpenToList(name,split,number,FileName,DirName):
    """
    This function opens values stored in text or csv files and returns a list containing the requested values.
    """
    f=[]
    TimeList_temp=[f for f in FileName if f[number:]==name]
    TimeListOpen = open(dataDir+DirName+TimeList_temp[0]) 
    SplitTemp=re.split(split,TimeListOpen.read())
    SplitTemp.pop()

    return SplitTemp

def OpenOpflowVector():
    """
    This function opens the optic flow measurements values initially stored in a text file.
    """
    inOpFlowFiles = [f for f in inFiles if f[-10:]=='opflow.txt']
    opFlowCols = ['sens0X','sens0Y','sens1X','sens1Y','date','time']
    opFlow = pd.read_table(os.path.join(dataDir,inOpFlowFiles[0]), sep='\s+', header=None, names=opFlowCols)

    return opFlow


def OpflowModulo(tempOpflowTime):
    """
    This function reads the optic flow measurements from the txt file and returns a list containing only the time values of the optic flow data.
    """
    OpflowList=[]
    for i in range(0,len(tempOpflowTime)):
        if i%2==1:
            OpflowList.append(tempOpflowTime[i])
        else:
            continue

    return OpflowList

def TimeToTimeSyst(tempTime,M):
    """
    This function translates the time contained in the text file from optic flow measurements and behavior frames to seconds.
    """
    ListSysTime=[]
    for i in range(0,len(tempTime)):
        tempTime1=tempTime[i].split(".") 
        tempTimeTuple=time.strptime(tempTime1[0], "%Y-%m-%d %H:%M:%S")
        if M == 1:
            if len(tempTime1) == 1:
                tempTime1.append("000000")
        tempTime_stamp=time.mktime(tempTimeTuple)+float(tempTime1[1])/(10**(len(tempTime1[1])))
        ListSysTime.append(tempTime_stamp)

    return ListSysTime

def CamTimeToTimeSyst(camTempTime,PulseSysTime):
    """
    This function shifts the time stored from the behavior images to the first time point of the pulse recorded data.
    """
    Cam_systime_shift=[]
    for i in range(0,len(camTempTime)):
        Cam_systime_shift.append(float(camTempTime[i]))
    cam_sysTime=[]
    for i in range(0,len(Cam_systime_shift)):
        cam_sysTime.append(Cam_systime_shift[i]-(Cam_systime_shift[0]-PulseSysTime[0]))

    return cam_sysTime

def GetStartAndStopVidTime(opflow,pulse,cam):
    """
    This function finds the start and stop time of the video. Start is defined as first time where a behavior frame, 
    fluorescence image and optic flow measurements were recorded simultaneously. 
    """
    vidStartSysTime=max(opflow[0], pulse[0], cam[0])
    vidStopSysTime=min(opflow[-1], pulse[-1], cam[-1])

    vidStartTimeShift=vidStartSysTime-vidStartSysTime
    vidStopTimeShift=vidStopSysTime-vidStartSysTime

    return vidStartSysTime, vidStopSysTime, vidStartTimeShift, vidStopTimeShift

def SpaceAndInterpolate(listToInterpolate,SizeListToMatch):
    """
    This function interpolates a list to match the length of another one.
    """
    LinspaceTemp=np.linspace(0,len(listToInterpolate)-1,len(listToInterpolate))
    LinspaceGoodLength=np.linspace(0,len(listToInterpolate)-1,len(SizeListToMatch))
    InterpolateList=np.interp(LinspaceGoodLength, LinspaceTemp, listToInterpolate) 

    return InterpolateList

def GetIdx(vidSysTime,pulse_systime_HD,stepOnTrueIdx,boe):
    """
    The function finds the index of the pulse value equal to the videoStartSysTime
    and returns the index in the pulse list which is equivalent to the index in stepOnTrue because both lists are aligned.
    """
    if vidSysTime==pulse_systime_HD[boe]:
        frameCntrIdx=stepOnTrueIdx[boe]
        StepIdx=stepOnTrueIdx[boe]
        pulseIdx=boe
    else:
        for i in range(0,len(pulse_systime_HD)):
            if vidSysTime-float(pulse_systime_HD[i])<0:
                if min(abs(vidSysTime-float(pulse_systime_HD[i-1])),abs(vidSysTime-float(pulse_systime_HD[i])))==(vidSysTime-float(pulse_systime_HD[i-1])):
                    frameCntrIdx=stepOnTrueIdx[i-1]
                    StepIdx=stepOnTrueIdx[i-1]
                    pulseIdx=i-1
                    break
                else:
                    frameCntrIdx=stepOnTrueIdx[i]
                    StepIdx=stepOnTrueIdx[i]
                    pulseIdx=i
                    break
            else:
                continue

    return frameCntrIdx, StepIdx, pulseIdx

def OpflowStartAndStop(vidStartSysTime,vidStopSysTime,opflow_systime):
    """
    This function finds opflow start and stop idx based on the start and stop time of the video.
    """
    if vidStartSysTime==opflow_systime[0]:
        opflowStartIdx=0
    else:     
        for i in range(0,len(opflow_systime)):
            if vidStartSysTime-opflow_systime[i]<0:
                if min(abs(vidStartSysTime-opflow_systime[i]),abs(vidStartSysTime-opflow_systime[i+1]))==(vidStartSysTime-opflow_systime[i]):
                    opflowStartIdx=i
                    break
                else:
                    opflowStartIdx=i-1
                    break
            else:
                continue

    if vidStopSysTime==opflow_systime[-1]:
        opflowStopIdx=len(opflow_systime)
    else:
        for i in range(len(opflow_systime)-1,1,-1):
            if vidStopSysTime-opflow_systime[i]>0:
                if min(abs(vidStopSysTime-opflow_systime[i-1]),abs(vidStopSysTime-opflow_systime[i]))==(vidStopSysTime-opflow_systime[i-1]):
                    opflowStopIdx=i-1
                    break
                else:
                    opflowStopIdx=i
                    break
            else:
                continue

    return opflowStartIdx, opflowStopIdx

def OpflowVectorSelection(timeSec,opflowStartIdx,opflowStopIdx,opFlow):
    """
    This function extracts the four vectors from optic flow measurements and interpolate them to match the 1500 points per seconds.
    """
    opFlow_HD = np.full((len(timeSec),4), np.nan) 
    flowDiff = (opflowStopIdx - opflowStartIdx) 
    opflowLength = len(opFlow[opflowStartIdx:opflowStopIdx])  
    ratioFlow = flowDiff/float(len(timeSec)) 

    opFlow_HD[:,0] = np.interp(np.linspace(0,opflowLength-1,len(timeSec)),np.linspace(0,opflowLength-1,opflowLength),pd.Series(opFlow['sens0X'][opflowStartIdx:opflowStopIdx]).values)
    opFlow_HD[:,1] = np.interp(np.linspace(0,opflowLength-1,len(timeSec)),np.linspace(0,opflowLength-1,opflowLength),pd.Series(opFlow['sens0Y'][opflowStartIdx:opflowStopIdx]).values)
    opFlow_HD[:,2] = np.interp(np.linspace(0,opflowLength-1,len(timeSec)),np.linspace(0,opflowLength-1,opflowLength),pd.Series(opFlow['sens1X'][opflowStartIdx:opflowStopIdx]).values)
    opFlow_HD[:,3] = np.interp(np.linspace(0,opflowLength-1,len(timeSec)),np.linspace(0,opflowLength-1,opflowLength),pd.Series(opFlow['sens1Y'][opflowStartIdx:opflowStopIdx]).values)

    return opFlow_HD[:,0], opFlow_HD[:,1], opFlow_HD[:,2], opFlow_HD[:,3], ratioFlow 

def GetSensorChannel(sensor,ratioFlow,gain,windowTotalTimeSec,timeSec,oldLength):
    """
    This function returns the sensor values times the calibration gain.
    """
    newWindow = windowTotalTimeSec*len(timeSec)/oldLength
    windowTemp = np.int(np.floor(newWindow*ratioFlow))
    window= np.ones(int(windowTemp))/float(windowTemp)
    smooth_series=np.convolve(sensor,window,'same')
    SensorTG = smooth_series*gain

    return SensorTG

def GetVelocities(sensor0X,sensor0Y,sensor1X,sensor1Y):
    """
    This function computes the AP, ML and Yaw rot/s from the sensor measurements.
    """
    velForw = -((sensor0Y + sensor1Y) * np.cos(np.deg2rad(45)))
    velSide = (sensor0Y - sensor1Y) * np.sin(np.deg2rad(45))
    velTurn = (sensor0X + sensor1X) / float(2)

    return velForw, velSide, velTurn

def GetFLuoIdx(frameCntr,L_GCamP6_tdtom_norm,R_GCamP6_tdtom_norm,stepOnOff_Bool_series,StepOnIdx,StepOffIdx):
    """
    This function interpolates fluorescence DR/R*100 to 15OO points per seconds. If NaN values are present within the fluorescence data, the function 
    removes them to compute the interpolation then replaces them at the good position. 
    """
    for i in range(0,len(frameCntr)):
        if frameCntr[i]/3==1:
            GC_stepOnIdx=i
            break
    for j in range(int(len(frameCntr)/2),len(frameCntr)):
        if frameCntr[j]/3==len(L_GCamP6_tdtom_norm):
            GC_stepOffIdx=j
            break

    GC_timesec_temp=np.linspace(0,len(L_GCamP6_tdtom_norm)-1,len(L_GCamP6_tdtom_norm))
    GC_timesec_temp_HD_whole=np.linspace(0,len(L_GCamP6_tdtom_norm)-1,len(stepOnOff_Bool_series[GC_stepOnIdx:GC_stepOffIdx]))

    leftNanList = np.argwhere(np.isnan(L_GCamP6_tdtom_norm))

    if (len(leftNanList) ==0 ) == True:

        L_GCamP6_tdtom_norm_HD_whole=np.interp(GC_timesec_temp_HD_whole,GC_timesec_temp,L_GCamP6_tdtom_norm)

    else : 
        print ("Left has nan values")
        idxLNan = []
        for i in range(len(leftNanList)):
            idxLNan.append(leftNanList[i][0])

        for j in idxLNan:
            if j != 0:
                L_GCamP6_tdtom_norm[j] = L_GCamP6_tdtom_norm[j-1]

        L_GCamP6_tdtom_norm_HD_whole=np.interp(GC_timesec_temp_HD_whole,GC_timesec_temp,L_GCamP6_tdtom_norm)
        factranspL = len(L_GCamP6_tdtom_norm_HD_whole)/len(L_GCamP6_tdtom_norm)
        idxLNanAfterInt = []
        for j in idxLNan :
            nouvInd = (j)*factranspL+j
            idxLNanAfterInt.append(nouvInd)
        for j in idxLNanAfterInt:
            if (j+factranspL != len(L_GCamP6_tdtom_norm_HD_whole)-1) == True:
                L_GCamP6_tdtom_norm_HD_whole[j:j+factranspL+1] = np.nan
            else : 
                L_GCamP6_tdtom_norm_HD_whole[j:j+factranspL] = np.nan


    rightNanList = np.argwhere(np.isnan(R_GCamP6_tdtom_norm))

    if (len(rightNanList) ==0 ) == True:

        R_GCamP6_tdtom_norm_HD_whole=np.interp(GC_timesec_temp_HD_whole,GC_timesec_temp,R_GCamP6_tdtom_norm)

    else : 
        print ("Right has nan values")
        idxRNan = []
        for i in range(len(rightNanList)):
            idxRNan.append(rightNanList[i][0])

        for j in idxRNan:
            if j != 0:
                R_GCamP6_tdtom_norm[j] = R_GCamP6_tdtom_norm[j-1]

        R_GCamP6_tdtom_norm_HD_whole=np.interp(GC_timesec_temp_HD_whole,GC_timesec_temp,R_GCamP6_tdtom_norm)
        factranspR = len(R_GCamP6_tdtom_norm_HD_whole)/len(R_GCamP6_tdtom_norm)
        idxRNanAfterInt = []
        for j in idxRNan :
            nouvInd = (j)*factranspR+j
            idxRNanAfterInt.append(nouvInd)
        for j in idxRNanAfterInt:
            if (j+factranspR != len(R_GCamP6_tdtom_norm_HD_whole)-1) == True:
                R_GCamP6_tdtom_norm_HD_whole[j:j+factranspR+1] = np.nan
            else : 
                R_GCamP6_tdtom_norm_HD_whole[j:j+factranspR] = np.nan

    L_GCamP6_tdtom_norm_HD0=L_GCamP6_tdtom_norm_HD_whole[StepOnIdx-GC_stepOnIdx:-1]
    R_GCamP6_tdtom_norm_HD0=R_GCamP6_tdtom_norm_HD_whole[StepOnIdx-GC_stepOnIdx:-1]

    if StepOffIdx>GC_stepOffIdx:
        nanTail=np.empty(StepOffIdx-GC_stepOffIdx+1)
        nanTail.fill(np.nan)
        L_GCamP6_tdtom_norm_HD=np.append(L_GCamP6_tdtom_norm_HD0,nanTail)
        R_GCamP6_tdtom_norm_HD=np.append(R_GCamP6_tdtom_norm_HD0,nanTail)

    else:
        L_GCamP6_tdtom_norm_HD = L_GCamP6_tdtom_norm_HD0
        R_GCamP6_tdtom_norm_HD = R_GCamP6_tdtom_norm_HD0
    
    return L_GCamP6_tdtom_norm_HD, R_GCamP6_tdtom_norm_HD 


def saveDataForAnalysis(frameCntr,L_DRSR_HD,R_DRSR_HD,velForw,velSide,velTurn,timeSec,cam_systime,vidStartSysTime,vidStopSysTime,stepOnIdx,puffSampled):
    """
    This function stores all the aligned data in a dictionary.
    """
    ValuesToSave = []
    ValuesToSave.append(vidStartSysTime)
    ValuesToSave.append(vidStopSysTime)
    ValuesToSave.append(stepOnIdx)

    DictDataList.update({'frameCntr':frameCntr})
    DictDataList.update({'L_DR':L_DRSR_HD})
    DictDataList.update({'R_DR':R_DRSR_HD})
    DictDataList.update({'velForw':velForw})
    DictDataList.update({'velSide':velSide})
    DictDataList.update({'velTurn':velTurn})
    DictDataList.update({'timeSec':timeSec})
    DictDataList.update({'cam_systime':cam_systime})
    DictDataList.update({'AddValues':ValuesToSave})
    DictDataList.update({'puffSampled':puffSampled})

    pickle.dump( DictDataList, open( outDir + "DicDataAnalysisPAB.p", "wb" ) ) 

    return

# ********** MAIN ***************

t1 = time.time()

L_DRR_norm, R_DRR_norm = GetFluoData()
frameCntr, piezo, step, puff = readH5file()
stepOnTrueIdx, stepOnOff_Bool_series = StepInfoIdx(step)

pulseTimeFile = OpenToList('timeStamps.txt','\n',-14,inFiles,'/')
opflowTimeFile = OpenToList('opflow.txt','\n|\t',-10,inFiles,'/')
camTimeFile = OpenToList('cam_tstamps.csv','\n',-15,inVidFiles,'/behavior_imgs/')

opflow = OpenOpflowVector()
opflowTimeFileUpdate = OpflowModulo(opflowTimeFile)
opflow_systime = TimeToTimeSyst(opflowTimeFileUpdate,1)
pulse_systime = TimeToTimeSyst(pulseTimeFile,0)
cam_systime = CamTimeToTimeSyst(camTimeFile,pulse_systime)

vidStartSysTime, vidStopSysTime, vidStartTimeShift, vidStopTimeShift = GetStartAndStopVidTime(opflow_systime,pulse_systime,cam_systime)

stepOnTrueIdx, frameCntr, stepOnIdx, stepOffIdx, LengthAdapt, oldStepLength, puffSampled = AdaptDataSampling(stepOnTrueIdx,vidStopTimeShift,frameCntr,samplingFact,puff)
pulse_systime_HD = SpaceAndInterpolate(pulse_systime,stepOnTrueIdx)
frameCntrStartIdx, StepOnIdxN, pulseStartIdx = GetIdx(vidStartSysTime,pulse_systime_HD,stepOnTrueIdx,0)
frameCntrStopIdx, StepOffIdx, pulseStopIdx = GetIdx(vidStopSysTime,pulse_systime_HD,stepOnTrueIdx,-1)
timeSec = np.linspace(vidStartTimeShift,vidStopTimeShift,StepOffIdx-StepOnIdxN)

opflowStartIdx, opflowStopIdx = OpflowStartAndStop(vidStartSysTime,vidStopSysTime,opflow_systime)
sensor0X, sensor0Y, sensor1X, sensor1Y, ratioFlow  = OpflowVectorSelection(timeSec,opflowStartIdx,opflowStopIdx,opflow)
sensor0X = GetSensorChannel(sensor0X,ratioFlow,gain0X,windowTotalTimeSec,timeSec,oldStepLength)
sensor0Y = GetSensorChannel(sensor0Y,ratioFlow,gain0Y,windowTotalTimeSec,timeSec,oldStepLength)
sensor1X = GetSensorChannel(sensor1X,ratioFlow,gain1X,windowTotalTimeSec,timeSec,oldStepLength)
sensor1Y = GetSensorChannel(sensor1Y,ratioFlow,gain1Y,windowTotalTimeSec,timeSec,oldStepLength)
velForw, velSide, velTurn = GetVelocities(sensor0X,sensor0Y,sensor1X,sensor1Y)

L_DRSR_HD, R_DRSR_HD = GetFLuoIdx(frameCntr,L_DRR_norm,R_DRR_norm,stepOnOff_Bool_series,StepOnIdxN,StepOffIdx)

saveDataForAnalysis(frameCntr,L_DRSR_HD,R_DRSR_HD,velForw,velSide,velTurn,timeSec,cam_systime,vidStartSysTime,vidStopSysTime,StepOnIdxN,puffSampled)

t2 = time.time()
print ("took",t2-t1,"seconds")

