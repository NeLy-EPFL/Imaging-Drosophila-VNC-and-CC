# -*- coding: utf-8 -*-
"""

Goal:
    This script allows the user to draw ellipses on the GC6s or tdTomato frames that will be used as the regions-of-interest.
    
Method:
    The script opens the text file containing the number of the frames that need to be fixed.
    The script opens the text file containing the fluorescence values of GC6s and tdTomato.
    The selected frames are presented one by one to the user and the user that has to enter the location of the center of the ellipses 
    followed by the dimensions of the new ellipses. The user has the opportunity to do it over and over until he is satisfied with the new ROIs.
    Once the new ROIs are validated by the user, the new fluorescence means are extracted for GC6s and tdTomato following the same procedure as in P1 and P2 and 
    the values are stored in text files.
     
"""

import cv2 
import numpy as np
from skimage import io
import os, os.path
import sys
import shutil
from PIL import Image, ImageDraw
import re

dataDir = 'YOUR_PATH_TO_EXPERIMENT_FOLDER'

outDir= dataDir + '/output/'
outDirROI_UsrCorrected = outDir + 'ROI_UsrCorrected/'
outDirGC6_UsrCorrected = outDir + 'GC6_UsrCorrected/'
outDircropROI_UsrCorrected = outDir + 'cropROI_UsrCorrected/'
outDirGC6_auto = outDir + 'GC6_auto/'

imgStackFileNameG = 'GC6s.tif'
imgStackFileNameR = 'tdTom.tif'
imgStackFileNameRGB = 'RGB.tif'
imgStackFileNameRGB1 = 'RGB1.tif'
imgStackFileNameRGB2 = 'RGB2.tif'
imgStackFileNameRGBtemp = 'RGBtemp.tif'

shutil.copyfile(dataDir + '/registered/' + imgStackFileNameRGB,dataDir + '/registered/' + imgStackFileNameRGB1) 
shutil.copyfile(dataDir + '/registered/' + imgStackFileNameRGB,dataDir + '/registered/' + imgStackFileNameRGB2) 
shutil.copyfile(dataDir + '/registered/' + imgStackFileNameRGB,dataDir + '/registered/' + imgStackFileNameRGBtemp) 

gcamp_chan = io.imread(dataDir + '/registered/' + imgStackFileNameG)
RGB_chan = io.imread(dataDir + '/registered/' + imgStackFileNameRGB)
tdTom_chan = io.imread(dataDir + '/registered/' + imgStackFileNameR)
RGB1_chan = io.imread(dataDir + '/registered/' + imgStackFileNameRGB1)
RGB2_chan = io.imread(dataDir + '/registered/' + imgStackFileNameRGB2)

imgRGBC = RGB_chan[:,:,:,:]
ValList1 = []
ValList2 = []
start = 0
stop = len(tdTom_chan)
LoopAgain = 0
cropLength=35
bin_s=2.5
sample_rate_hz=2.418032787 

def draw_circle(event,x,y,flags,param):

    global mouseX,mouseY
    if event == cv2.EVENT_RBUTTONDOWN :
        ValList2.append([x,y])
    if event == cv2.EVENT_LBUTTONDOWN : 
        ValList1.append([x,y])       
        mouseX,mouseY = x,y    

def ManualDrawFrame(imgRGBC,imgRGB,img,LoopAgain):
    """
    This function creates the user interface with OpenCV library to allow the user to draw the center and define the dimensions of the ellipse that will be stored as the new ROIs.
    """
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_circle)
    contours = []
    QuelROI = 0
    corrAnswer  = ["Y","y","N","n"]
    Yes = ["Y","y"]
    No = ["N","n"]

    while(1):
        cv2.imshow('image',imgRGBC)
        k = cv2.waitKey(20) & 0xFF==27
        if (not ValList1)==False:
            cv2.circle(imgRGBC,(ValList1[0][0],ValList1[0][1]),2,(255,255,255),-1) 
        if (not ValList2)==False:
            cv2.circle(imgRGBC,(ValList2[0][0],ValList2[0][1]),2,(255,255,255),-1) 
        if (len(ValList1)>=2 and len(ValList2)==0):
            print ("please select only one center for the Left ROI - start over")
            imgRGBC = imgRGB[:,:,:].copy()
            del ValList1[:]
            del ValList2[:]
        elif (len(ValList2)>=2 and len(ValList1)==0) :
            print ("please select only one center for the right ROI - start over")
            imgRGBC = imgRGB[:,:,:].copy()
            del ValList1[:]
            del ValList2[:]
        elif (len(ValList2)>1 and len(ValList1)==1) or (len(ValList1)>1 and len(ValList2)==1) :
            finished1 = False
            while not finished1:    
                finished1 = True 
                print ("Are the center well positioned ? ")
                KeyinAnsC=raw_input('if yes enter Y, if no - you can choose new centers position - enter N ->')
                if len(KeyinAnsC) > 1 :
                    print("Incorrect input, please try again!\n")
                    finished1 = False
                elif (KeyinAnsC not in corrAnswer) == True:
                    print("Incorrect input, please try again!\n")
                    finished1 = False
            if KeyinAnsC in Yes:
                while(LoopAgain==0):
                    finished2 = False
                    while not finished2:
                        finished2 = True
                        print ("Enter Width, Height and Inclination Angle of Left ellipse ? ")
                        KeyinAnsL=raw_input('example : 10 + 20 + 0 ->')
                        KeyinAnsLSplit  = KeyinAnsL.split("+")
                        if (len(KeyinAnsLSplit) != 3) :
                            print("Incorrect input, please try again!\n")
                            finished2 = False
                        elif (KeyinAnsLSplit[0].isdigit)==False:
                            print("Incorrect input, please key in a number again!\n")
                            finished2 = False
                        elif (KeyinAnsLSplit[1].isdigit)==False:
                            print("Incorrect input, please key in a number again!\n")
                            finished2 = False
                        elif (KeyinAnsLSplit[2].isdigit)==False:
                            print("Incorrect input, please key in a number again!\n")
                            finished2 = False                      
                    finished3 = False
                    while not finished3:
                        finished3 = True
                        print ("Enter Width, Height and Inclination Angle of Right ellipse ? ")
                        KeyinAnsR=raw_input('example : 10 + 20 + 0 ->')
                        KeyinAnsRSplit  = KeyinAnsR.split("+")
                        if (len(KeyinAnsRSplit) != 3) :
                            print("Incorrect input, please try again!\n")
                            finished3 = False
                        elif (KeyinAnsRSplit[0].isdigit)==False:
                            print("Incorrect input, please key in a number again!\n")
                            finished3 = False
                        elif (KeyinAnsRSplit[1].isdigit)==False:
                            print("Incorrect input, please key in a number again!\n")
                            finished3 = False
                        elif (KeyinAnsRSplit[2].isdigit)==False:
                            print("Incorrect input, please key in a number again!\n")
                            finished3 = False
                    tupleCenterL = (int(ValList1[0][0]),int(ValList1[0][1]))
                    tupleCenterR = (int(ValList2[0][0]),int(ValList2[0][1]))
                    dimL = (int(KeyinAnsLSplit[0]),int(KeyinAnsLSplit[1]))
                    dimR = (int(KeyinAnsRSplit[0]),int(KeyinAnsRSplit[1]))
                    angL = int(KeyinAnsLSplit[2])
                    angR = int(KeyinAnsRSplit[2])
                    imgRGBC = imgRGB[:,:,:].copy()
                    imgTemp = img[:,:].copy()
                    imgTestN = np.zeros(np.shape(imgTemp))
                    img2TestN = cv2.convertScaleAbs(imgTestN, alpha=(255.0/65535.0))
                    img5N = img2TestN.astype(np.uint8)
                    cv2.ellipse(img5N, tupleCenterL, dimL, angL, 0, 360, 255, -1)
                    cv2.ellipse(img5N, tupleCenterR, dimR, angR, 0, 360, 255, -1)
                    _, contours, _= cv2.findContours(img5N,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
                    cv2.drawContours(imgRGBC,contours,-1,(255,255,255),1) #Draw on RGBtemp image 
                    cv2.imshow('image',imgRGBC)
                    finished4 = False
                    while not finished4:
                        finished4 = True
                        print ("Are those ROI well selected ? ")
                        KeyinAnsROI=raw_input('key in Y for Yes and N for no (you will go through the dimensions selection again) ->')
                        if len(KeyinAnsROI)>1:
                            print("Incorrect input, please try again!\n")
                            finished4 = False
                        elif (KeyinAnsROI not in corrAnswer) == True:
                            print("Incorrect input, please try again!\n")
                            finished4 = False
                        elif KeyinAnsROI in Yes:
                            LoopAgain = 1
                            del ValList1[:]
                            del ValList2[:]
                            break 
                        elif KeyinAnsROI in No:
                            print ("Go through dimension selection again")
                            LoopAgain = 0
                            imgRGBC = imgRGB[:,:,:].copy()
                            break 
                    if LoopAgain == 1:
                        break
                break
            else :
                print ("Enter new centers location")
                imgRGBC = imgRGB[:,:,:].copy()
                del ValList1[:]
                del ValList2[:]

    return contours

def OpenFrameToReview():
    """
    This function opens - if it exists - the text file that was stored from Part 2 (P2) with the number of wrongly detected frames and returns the list with the number of the
    frames that need to be manually corrected.
    """
    if os.path.exists(outDirGC6_UsrCorrected +'wrongROIShape.txt')==True: 
        readframes_for_fix = open(outDirGC6_UsrCorrected +'wrongROIShape.txt')
    else : 
        print("No wrong ROI detected - please check that WrongROIShape.txt file exists")
        sys.exit(0) 

    frames_for_fix=[]
    string_readframes_for_fix=(readframes_for_fix.read()).split()
    for i in range(len(string_readframes_for_fix)):
        frames_for_fix.append(int(string_readframes_for_fix[i]))
    frames_for_fix.sort()

    return frames_for_fix

def OpenPausedFrameToReview(FrameToFix):
    """
    This function recovers the number of frames that are still waiting to be corrected if the script Part 3 was run and stopped/paused. 
    """
    frames_for_fixTemp=[]

    if os.path.exists(outDirGC6_UsrCorrected +'FrameToFix_TempW.txt')==True: 
        readframes_for_fix = open(outDirGC6_UsrCorrected +'FrameToFix_TempW.txt')
        string_readframes_for_fix=(readframes_for_fix.read()).split("\n")
        for i in range(0,len(string_readframes_for_fix)-1): 
            frames_for_fixTemp.append(int(string_readframes_for_fix[i]))

    if (not frames_for_fixTemp )== True:
        FrameRet = FrameToFix
    else : 
        FrameRet = frames_for_fixTemp

    return FrameRet

def OpenOldAbsValues():
    """
    This function loads the fluorescence means of every ROI that were calculated by the autoselection or manual correction script. It returns 4 lists containing the 
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

    return L_GCamP6_abs_correct,R_GCamP6_abs_correct,L_tdtom_abs_correct,R_tdtom_abs_correct

def OpenPausedValues(L_GC,R_GC,L_TT,R_TT):
    """
    This function recovers the fluorescence means of each ROI from GC6s and tdTomato that were stored if this script (P3) was run and paused.
    """
    LGCTemp = []
    LTTTemp = []
    RGCTemp = []
    RTTTemp = []
    WrongDetTemp = []
    WrongDetPart = []
    WrongDetAllList = []

    if os.path.exists(outDirGC6_UsrCorrected +'L_GC_TempW.txt')==True: 
        L_GC_Temp = open(outDirGC6_UsrCorrected +'L_GC_TempW.txt')
        
        L_GC_Temp_split=(L_GC_Temp.read()).split("\n")

        for i in range(0,len(L_GC_Temp_split)-1): 
            LGCTemp.append(float(L_GC_Temp_split[i]))

    if os.path.exists(outDirGC6_UsrCorrected +'L_TT_TempW.txt')==True: 
        L_TT_Temp = open(outDirGC6_UsrCorrected +'L_TT_TempW.txt')
        
        L_TT_Temp_split=(L_TT_Temp.read()).split("\n")

        for i in range(0,len(L_TT_Temp_split)-1): 
            LTTTemp.append(float(L_TT_Temp_split[i]))

    if os.path.exists(outDirGC6_UsrCorrected +'R_GC_TempW.txt')==True: 
        R_GC_Temp = open(outDirGC6_UsrCorrected +'R_GC_TempW.txt')
        
        R_GC_Temp_split=(R_GC_Temp.read()).split("\n")

        for i in range(0,len(R_GC_Temp_split)-1): 
            RGCTemp.append(float(R_GC_Temp_split[i]))

    if os.path.exists(outDirGC6_UsrCorrected +'R_TT_TempW.txt')==True: 
        R_TT_Temp = open(outDirGC6_UsrCorrected +'R_TT_TempW.txt')
        
        R_TT_Temp_split=(R_TT_Temp.read()).split("\n")

        for i in range(0,len(R_TT_Temp_split)-1): 
            RTTTemp.append(float(R_TT_Temp_split[i]))

    if not LGCTemp == True:
        return L_GC,R_GC,L_TT,R_TT

    else : 
        return LGCTemp, RGCTemp, LTTTemp, RTTTemp

def ApplyContours(contours,imgRGB1,imgRGB2,imgRGB):
    """
    This function calculates the centroid of each contour and draws the left and right contours together and on separate images.
    """
    cv2.drawContours(imgRGB,contours,-1,(255,255,255),1)
    MLT = cv2.moments(contours[0]) 
    cxLT = float(MLT['m10']/MLT['m00'])
    cyLT = float(MLT['m01']/MLT['m00'])                        
    MRT = cv2.moments(contours[1])
    cxRT = float(MRT['m10']/MRT['m00'])
    cyRT = float(MRT['m01']/MRT['m00'])  

    contoursN = []
    if cxLT < cxRT:
        cv2.drawContours(imgRGB1,contours[0],-1,(255,255,255),1)
        cxL = cxLT
        cyL = cyLT
        cv2.drawContours(imgRGB2,contours[1],-1,(255,255,255),1)
        cxR = cxRT
        cyR = cyRT
        contoursN.append(contours[0])
        contoursN.append(contours[1])
    else : 
        cv2.drawContours(imgRGB1,contours[1],-1,(255,255,255),1)
        cxL = cxRT
        cyL = cyRT
        cv2.drawContours(imgRGB2,contours[0],-1,(255,255,255),1)
        cxR = cxLT
        cyR = cyLT
        contoursN.append(contours[1])
        contoursN.append(contours[0])

    return cxL, cxR, cyL, cyR, contoursN, imgRGB

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

def GetAbsFluoValues(imgGC,imgTdTom,contours,listG6,listTT,k):
    """
    This function calculates the fluorescence mean of the ROI selected and replaces the old value by the new one in the list containing
    all the fluorescence means. 
    """
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

def CreateAndStoreTempFluoValues(L_GC, L_TT, R_GC, R_TT):
    """
    This function stores the temporary fluorescence values to be able to recover them it the script P3 is paused.
    """
    np.savetxt(outDirGC6_UsrCorrected + "L_GC_TempW" + '.txt', L_GC, delimiter=' ', newline=os.linesep, fmt="%s") 
    np.savetxt(outDirGC6_UsrCorrected + "L_TT_TempW" + '.txt', L_TT, delimiter=' ', newline=os.linesep, fmt="%s")
    np.savetxt(outDirGC6_UsrCorrected + "R_GC_TempW" + '.txt', R_GC, delimiter=' ', newline=os.linesep, fmt="%s")
    np.savetxt(outDirGC6_UsrCorrected + "R_TT_TempW" + '.txt', R_TT, delimiter=' ', newline=os.linesep, fmt="%s")

    return

def CreateAndSaveFramesToFix(k,FrameToFixCopy):
    """
    This function stores the temporary number of frames to fix to be able to recover them if the script P3 is paused.
    """
    if k in FrameToFixCopy:
        FrameToFixCopy.remove(k)
    np.savetxt(outDirGC6_UsrCorrected + "FrameToFix_TempW" + '.txt', FrameToFixCopy, delimiter=' ', newline=os.linesep, fmt="%s")

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
    binned_TTom=[]
    binned_GC_TTom=[]
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
        TT_Norm.append(((GFluoAbs[b]/TFluoAbs[b])-baseline_GC_TT)/baseline_GC_TT*100 )

    np.savetxt(outDirGC6_UsrCorrected + name + "_GC_dF_UsrW" + '.txt', G6_Norm, delimiter=' ', newline=os.linesep, fmt="%s") 
    np.savetxt(outDirGC6_UsrCorrected + name +"_GC_tdtom_norm_UsrW" + '.txt', TT_Norm, delimiter=' ', newline=os.linesep, fmt="%s")
    np.savetxt(outDirGC6_UsrCorrected + name + "_GC_orig_UsrW" + '.txt', GFluoAbs, delimiter=' ', newline=os.linesep, fmt="%s")
    np.savetxt(outDirGC6_UsrCorrected + name + "_tdtom_orig_UsrW" + '.txt', TFluoAbs, delimiter=' ', newline=os.linesep, fmt="%s")

    return G6_Norm, TT_Norm

def ChannelSelection():
    """
    This function provides a tool for the user to select the channels (red and green or red or green) on which he will select
    the ROI.
    """
    ChannelAnalysis = 0
    finished = False
    while not finished:
        print("Do you want to use RGB channel (RG), or GCaMP6s channel alone (G) or TdTom channel alone (R) ? ") 
        frameKeyin=raw_input("Keyin the channels you want to use to detect the ROI (eg.,'R' or 'RG')\n->") 
        try:
            finished = True
            if frameKeyin=="RG":
                ChannelAnalysis = 0
            elif frameKeyin == "R":
                ChannelAnalysis = 1
            elif frameKeyin == "G":
                ChannelAnalysis = 2
            else:
                print("Incorrect format of input, Please try again\n")
                finished = False
            
        except IndexError:
            print("Please keyin the channel selection in correct format!")
            
    return ChannelAnalysis


#****** MAIN ************

FrameToFix = OpenFrameToReview()
FrameToFix = OpenPausedFrameToReview(FrameToFix)
print ('\033[1m' + "Please complete this script entirely before proceeding to P4.")
print '\033[0m'
print ("Frames to Fix",FrameToFix)
L_GC,R_GC,L_TT,R_TT = OpenOldAbsValues()
L_GC,R_GC,L_TT,R_TT = OpenPausedValues(L_GC,R_GC,L_TT,R_TT)
k = 0
count = 0
ChanAn = ChannelSelection()
StoredFrameToFix = len(FrameToFix)
newSavingList = FrameToFix[:]
print FrameToFix
while k in range(start,stop): 
    if k in (FrameToFix):
        print ("There are", StoredFrameToFix-count , "frames to fix.")
        print ("Frame number:",k)

        if ChanAn == 1:
            imgRGBC = tdTom_chan[k,:,:].copy()
        elif ChanAn == 2 :
            imgRGBC = gcamp_chan[k,:,:].copy()
        else :
            imgRGBC = RGB_chan[k,:,:,:].copy()

        contoursD = ManualDrawFrame(imgRGBC,RGB_chan[k,:,:,:].copy(),tdTom_chan[k,:,:],LoopAgain)
        cxL, cxR, cyL, cyR, contours, imgRGB = ApplyContours(contoursD,RGB1_chan[k,:,:,:],RGB2_chan[k,:,:,:],RGB_chan[k,:,:,:])
        imgCropL, imgCropR = CroppingImgNew(RGB1_chan[k,:,:,:],RGB2_chan[k,:,:,:],cxL,cxR,cyL,cyR)
        CreateAndStoreImages(imgCropR,imgCropL,imgRGB)
        L_GC, L_TT = GetAbsFluoValues(gcamp_chan[k,:,:],tdTom_chan[k,:,:],contours[0],L_GC,L_TT,k)
        R_GC, R_TT = GetAbsFluoValues(gcamp_chan[k,:,:],tdTom_chan[k,:,:],contours[1],R_GC,R_TT,k)
        CreateAndStoreTempFluoValues(L_GC, L_TT, R_GC, R_TT)
        CreateAndSaveFramesToFix(k,newSavingList)
        count = count + 1
    k+=1

G6_Norm_L, TT_Norm_L = GraphValues(L_GC,L_TT,"L")
G6_Norm_R, TT_Norm_R = GraphValues(R_GC,R_TT,"R")

if os.path.exists(outDirGC6_UsrCorrected +'FrameToFix_TempW.txt')==True:
    os.remove(outDirGC6_UsrCorrected +'FrameToFix_TempW.txt')

if os.path.exists(outDirGC6_UsrCorrected +'L_GC_TempW.txt')==True:
    os.remove(outDirGC6_UsrCorrected +'L_GC_TempW.txt')
    os.remove(outDirGC6_UsrCorrected +'L_TT_TempW.txt')
    os.remove(outDirGC6_UsrCorrected +'R_GC_TempW.txt')
    os.remove(outDirGC6_UsrCorrected +'R_TT_TempW.txt')



