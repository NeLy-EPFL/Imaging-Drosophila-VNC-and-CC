# -*- coding: utf-8 -*-
"""

Goal:
    This script generates automatic detection of regions-of-interest (ROI) based on first manual selection of ROI from the user and stores the DR/R *100 of the 
    fluorescence signal. R is the baseline of the ratio of GC6s to tdTomato.
    
Method:
    The script finds all the objects within a reference frame selected by the user and presents them to the user. 
    The user selects the left and right ROI. The script then performs cross correlation between each frame and the reference frame to find the best position of the ROI in the frame being processed.
    All the objects within the same frame are found using OpenCV library and the ROI that have centroids closer to the one calculated from the cross correlation function are selected as left and right
    ROIs respectively. 
    The fluorescence values of each ROI are then calculated by applying a mask array composed of 0 and 1 values (1 in the ROI regions) and the average of the GC6s and TdTomato within the ROI regions are calculated.
    The DR/R*100 value is then computed for each ROI and stored in a txt file. The ROIs selected are drawn on the RGB images and those images are stored as png files. 

"""

from skimage import io
import cv2
import shutil
import numpy as np
import matplotlib.pyplot as plt
import os, os.path
import time
from PIL import Image, ImageDraw
import math

t1 = time.time()

dataDir = 'YOUR_PATH_TO_EXPERIMENT_FOLDER'
print(dataDir," is in process...")


imgStackFileNameG = 'GC6s.tif'
imgStackFileNameR = 'tdTom.tif'
imgStackFileNameRGB = 'RGB.tif'
imgStackFileNameRGB1 = 'RGB1.tif'
imgStackFileNameRGB2 = 'RGB2.tif'
imgStackFileNameRGBtemp = 'RGBtemp.tif'

#%% COPY RGB file for different purpose(ROI total, cropROI L, cropROI R)
shutil.copyfile(dataDir + '/registered/' + imgStackFileNameRGB, dataDir + '/registered/' + imgStackFileNameRGB1) 
shutil.copyfile(dataDir + '/registered/' + imgStackFileNameRGB, dataDir + '/registered/' + imgStackFileNameRGB2) 
shutil.copyfile(dataDir + '/registered/' + imgStackFileNameRGB, dataDir + '/registered/' + imgStackFileNameRGBtemp) 

#%% IMPORT imaging data
gcamp_chan = io.imread(dataDir + '/registered/' + imgStackFileNameG)
tdTom_chan = io.imread(dataDir + '/registered/' + imgStackFileNameR)
RGB_chan = io.imread(dataDir + '/registered/' + imgStackFileNameRGB)
RGB1_chan = io.imread(dataDir + '/registered/' + imgStackFileNameRGB1)
RGB2_chan = io.imread(dataDir + '/registered/' + imgStackFileNameRGB2)
RGBtemp_chan = io.imread(dataDir + '/registered/' + imgStackFileNameRGBtemp)


del imgStackFileNameG, imgStackFileNameR


outDir= dataDir + '/output/'
outDirROI_auto = outDir + 'ROI_auto/'
outDirGC6_auto = outDir + 'GC6_auto/'
outDircropROI_auto = outDir + 'cropROI_auto/'

if not os.path.exists(outDir):
    os.makedirs(outDir)
if not os.path.exists(outDirROI_auto):    
    os.makedirs(outDirROI_auto)
if not os.path.exists(outDirGC6_auto):    
    os.makedirs(outDirGC6_auto)
if not os.path.exists(outDircropROI_auto):    
    os.makedirs(outDircropROI_auto)


L_TdTomFluoAbs = []
L_GCFluoAbs = []
R_TdTomFluoAbs = []
R_GCFluoAbs = []
start = 0
stop = len(tdTom_chan)
sizeExcl = 1
blurVal = 9
blurVal_BiFltr=120
blurVal_med=6
erodeVal = 5
cropLength=35
bin_s=2.5
sample_rate_hz=2.418032787 


def ChannelSelection():
    """
    This function allows the user to select the from Green, Red or Green and Red channels for the ROI detection.
    """
    ChannelAnalysis = 0
    finished = False
    while not finished:
        print("Do you want to use both TdTom & GCaMP6s (RG), or GCaMP6s channel alone (G) or TdTom channel alone (R) ? ") 
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


def ThresholdingImage(img,k,blurVal,erodeval):
    """
    This function converts the image to an 8 bit image, then extends the colour range (and actually augment 
    image contrast to better detect the ROI). We substract the baseline and then scale the max value 
    of the colour to 255 (max value usable on 8 bit). Afterwards, we apply a blur filter (other filters can be selected)
    to the image before applying the Otsu Treshold. The blur filter is used to smooth the image before applying
    the threshold. Once the image is thresholded we use an erosion function to avoid very small ROI to be detected
    & to avoid larger ROI size detection due to the blur filter. If you want to increase the ROI size detection
    you should decreade the erosion value (erodeval). 
    """
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

def DrawAllContours(img,erodeIm,imgRGBtemp):
    """
    This function receives the eroded image and detects all the ROIs in the frame. The contours of all ROIs are detected and are added
    to the img with the respective index of the contour. The function returns the img containing the contours, the contours list, the number 
    of objects detected on the frame (ret) and markers which is an array representing the image with all the objects (attributed to their respective label value) detected in it.
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
    for c in ContoursD :         
        if countingN<=len(ContoursD)-1: 
            M1 = cv2.moments(c)
            if M1['m00'] == 0.0:
                cx = countingN
                cy = countingN
            else :
                cx = float(M1['m10']/M1['m00'])
                cy = float(M1['m01']/M1['m00'])
                cv2.putText(imgtemp, str(countingN), (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            countingN=countingN+1

    return imgtemp, ContoursD, ret, markers


def findMostProbableContour(img,erodeIm,PRCRC,PRCLC):
    """
    This function finds the left and right ROIs that are closer to the ones selected by the user on the reference frame. 
    This function receives the eroded image and the projected centroids of the contours based on the cross correlation function.
    The function detects the ROIs on the image and stores them. The centroid of the found ROIs are then compared to the centroids
    received as argument of the function and the ROIs that have the centroids closer to the reference ones are selected
    as left and right ROIs. The function returns the centroids of the selected contours and the contours list for 
    left and right ROIs.
    """
    ret, markers = cv2.connectedComponents(erodeIm)
    allroisC = np.zeros(np.shape(img), dtype=bool) 

    for j in range(1, ret+1): 
        test = markers == j 
        if np.sum(test) >= sizeExcl:
            allroisC = allroisC + test
    thresh2C = np.array(allroisC,np.uint8) 

    _, ContoursD, _= cv2.findContours(thresh2C,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  

    countingN=0
    ContNewList = []
    GlobList = []
    for c in ContoursD :         
        if countingN<=len(ContoursD)-1: 
            M1 = cv2.moments(c)
            if M1['m00'] == 0.0:
                cx = countingN
                cy = countingN
            else :
                tempList = []
                cx = float(M1['m10']/M1['m00'])
                cy = float(M1['m01']/M1['m00'])
                tempList.append(countingN)
                tempList.append(M1)
                tempList.append(cx)
                tempList.append(cy)
                GlobList.append(tempList)
            
            countingN=countingN+1

    tempValR = math.hypot((GlobList[0][2])-PRCRC[0],(GlobList[0][3])-PRCRC[1])
    tempValL = math.hypot((GlobList[0][2])-PRCLC[0],(GlobList[0][3])-PRCLC[1])
    currentIdxR = 0
    currentIdxL = 0
    countCurrent = 0
    GlobListIndR = 0
    GlobListIndL = 0

    for i in GlobList:
        tempValRC = math.hypot((i[2])-PRCRC[0],(i[3])-PRCRC[1])
        tempValLC = math.hypot((i[2])-PRCLC[0],(i[3])-PRCLC[1])
        if abs(tempValRC) < abs(tempValR):
            tempValR = tempValRC
            currentIdxR = i[0]
            GlobListIndR = countCurrent
        if abs(tempValLC) < abs(tempValL):
            tempValL = tempValLC
            currentIdxL = i[0]
            GlobListIndL = countCurrent
        countCurrent += 1

    cxR = GlobList[GlobListIndR][2]
    cyR = GlobList[GlobListIndR][3]
    RindCont = GlobList[GlobListIndR][0]
    contR = ContoursD[RindCont]

    cxL = GlobList[GlobListIndL][2]
    cyL = GlobList[GlobListIndL][3]
    LindCont = GlobList[GlobListIndL][0]
    contL = ContoursD[LindCont]


    return cxR, cyR, cxL, cyL, contR, contL

def findingContoursFirst(contoursD,ret,markers,Indexlist,imgRGB1,imgRGB2,imgRGB):
    """
    This function calculates the centroid of each contour and draws the previously selected left and right contours on the 3 different images
    that will be stored later. 
    """
    ContList = []

    for l in range(len(Indexlist)):
        Cont = contoursD[Indexlist[l]]
        ContList.append(Cont)

    cv2.drawContours(imgRGB,ContList,-1,(255,255,255),1)

    cv2.drawContours(imgRGB1,ContList[0],-1,(255,255,255),1)
    ML = cv2.moments(ContList[0]) #summation of all contours to gain the sum
    cxL = float(ML['m10']/ML['m00'])
    cyL = float(ML['m01']/ML['m00'])                        
     
    cv2.drawContours(imgRGB2,ContList[1],-1,(255,255,255),1)
    MR = cv2.moments(ContList[1])
    cxR = float(MR['m10']/MR['m00'])
    cyR = float(MR['m01']/MR['m00'])  

    return ML, MR, cxL, cxR, cyL, cyR, ContList, imgRGB

def GetAbsFluoValues(imgGC,imgTdTom,contours,listG6,listTT):
    """
    This function calculates the mean of the fluorescence values of the ROI on each frame and adds the values for green
    and red channel in their respective lists.
    """
    mask = np.zeros((imgGC.shape))
    cv2.fillConvexPoly(mask, contours, 1)
    mask = mask.astype(np.bool)
    masked_imgGC = imgGC*mask
    masked_imgTdTom = imgTdTom*mask
    fluorGC = masked_imgGC[masked_imgGC.nonzero()].mean()
    fluorTdTom = masked_imgTdTom[masked_imgGC.nonzero()].mean()
    listG6.append(fluorGC)
    listTT.append(fluorTdTom)

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

    for b in range(0,len(GFluoAbs)):
        G6_Norm.append(((GFluoAbs[b]-baseline_GC)/baseline_GC)*100)

    for b in range(0,len(TFluoAbs)):
        TT_Norm.append(((GFluoAbs[b]/TFluoAbs[b])-baseline_GC_TT)/baseline_GC_TT*100)


    np.savetxt(outDirGC6_auto + name + "_GC_dF" + '.txt', G6_Norm, delimiter=' ', newline=os.linesep, fmt="%s") 

    np.savetxt(outDirGC6_auto + name +"_GC_tdtom_norm" + '.txt', TT_Norm, delimiter=' ', newline=os.linesep, fmt="%s")

    np.savetxt(outDirGC6_auto + name + "_GC_orig" + '.txt', GFluoAbs, delimiter=' ', newline=os.linesep, fmt="%s")

    np.savetxt(outDirGC6_auto + name + "_tdtom_orig" + '.txt', TFluoAbs, delimiter=' ', newline=os.linesep, fmt="%s")


    return G6_Norm, TT_Norm

def CreateAndStoreImages(imgCropR,imgCropL,imgRGB,k):
    """
    This function creates and stores the images with the selected ROIs. The function crops the images that have the left and right
    ROIs to center the ROIs then combines the cropped images to store both ROIs on one image. It also creates an image with the full
    fluorescence image and the cropped regions around the selected ROIs.
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

    imNewA.save(outDirROI_auto + "%04d" % k + '.png')

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

    imNewCrop.save(outDircropROI_auto + "%04d" % k + '.png')

    return

def AskROI(img,lenCont):
    """
    This function presents the img containing all the detected ROIs to the user and the user can select which ROIs will be used as reference.
    """
    testList = []
    for j in range(lenCont):
        testList.append(str(j))

    plt.imshow(img)
    plt.pause(0.2)
                           
    finished1 = False
    while not finished1:    
        finished1 = True       
        print('which is L ROI?')
        KeyinAnsL=raw_input('key in number of the ROI ->')
        print('which is R ROI ?')
        KeyinAnsR=raw_input('key in number of the ROI ->')
        
        LSplit = KeyinAnsL.split("+")
        RSplit = KeyinAnsR.split("+")
        if (LSplit[0] not in testList) == True:
            print("Incorrect input, please try again!\n")
            finished1 = False
        if (RSplit[0] not in testList) == True:
            print("Incorrect input, please try again!\n")
            finished1 = False

    arrayLR=[int(KeyinAnsL), int(KeyinAnsR)] 
    plt.close()

    return arrayLR

def CroppingImgNew(imgRGB1,imgRGB2,imgRGB,cxL,cxR,cyL,cyR,contoursL,contoursR):
    """
    This function adds black edges on imgRGB1 and imgRGB2 before cropping around the ROI. 
    """
    cv2.drawContours(imgRGB1,contoursL,-1,(255,255,255),1)
    cv2.drawContours(imgRGB2,contoursR,-1,(255,255,255),1)

    cv2.drawContours(imgRGB,contoursL,-1,(255,255,255),1)
    cv2.drawContours(imgRGB,contoursR,-1,(255,255,255),1)

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

    return imgCropL, imgCropR, imgRGB


def register_images(refImage, shiftedImage, return_error=False):
    """
    shiftedBackImage, colShift, rowShift, phaseDiff, [error] = register_images(refImage, shiftedImage, return_error=False)
    
    Translated from MATLAB code written by Manuel Guizar
    downloaded from http://www.mathworks.com/matlabcentral/fileexchange/18401-efficient-subpixel-image-registration-by-cross-correlation.
    """
    buf1ft = np.fft.fft2(refImage)
    buf2ft = np.fft.fft2(shiftedImage)
    m,n = buf1ft.shape
    CC = np.fft.ifft2(buf1ft*buf2ft.conjugate())

    max1 = np.max(CC,axis=0)
    loc1 = np.argmax(CC,axis=0)
    cloc = np.argmax(max1)

    rloc = loc1[cloc]
    CCmax = CC[rloc,cloc]
    
    if return_error:
        rfzero = np.sum(np.abs(buf1ft.ravel())**2)/(m*n)
        rgzero = np.sum(np.abs(buf2ft.ravel())**2)/(m*n)

        error = 1.0 - CCmax*CCmax.conjugate()/(rgzero*rfzero)
        error = np.sqrt(np.abs(error));

    phaseDiff=np.angle(CCmax);

    md2 = np.fix(m/2.0) 
    nd2 = np.fix(n/2.0)
    if rloc > md2:
        rowShift = rloc - m; 
    else:
        rowShift = rloc;

    if cloc > nd2:
        colShift = cloc - n;
    else:
        colShift = cloc;

    # Compute registered version of buf2ft
    nr,nc = buf2ft.shape
    Nr = np.fft.ifftshift(np.arange(-np.fix(nr/2.0),np.ceil(nr/2.0)));
    Nc = np.fft.ifftshift(np.arange(-np.fix(nc/2.0),np.ceil(nc/2.0)));
    [Nc,Nr] = np.meshgrid(Nc,Nr)
    greg = buf2ft*np.exp(np.complex(0,1)*2*np.pi*(-rowShift*Nr/nr-colShift*Nc/nc));
    greg = greg*np.exp(np.complex(0,1)*phaseDiff);

    if np.can_cast(np.float32,shiftedImage.dtype): 
        shiftedBackImage = np.abs(np.fft.ifft2(greg))
    else:
        shiftedBackImage = np.round(np.abs(np.fft.ifft2(greg))).astype(shiftedImage.dtype)
    
    if return_error:
        output = [shiftedBackImage, colShift, rowShift, phaseDiff, error]
    else:
        output = [shiftedBackImage, colShift, rowShift, phaseDiff]
    return output

def CloseCentroidDetect(col,row,PR) :

    prx = PR[0]-col
    pry = PR[1]-row

    return prx, pry

#**************** MAIN ******************

ChanAn = ChannelSelection()

if ChanAn == 1:
    imgS = tdTom_chan[0,:,:]
elif ChanAn == 2 :
    imgS = gcamp_chan[0,:,:]
else :
    imgS = (tdTom_chan[0,:,:]+gcamp_chan[0,:,:])/2


imgRGBtempS = RGBtemp_chan[0,:,:,:]
erodeImS = ThresholdingImage(imgS,0,blurVal,erodeVal)
imgTemp, ContoursD, ret, markers= DrawAllContours(imgS,erodeImS,imgRGBtempS)
NewROIListS = AskROI(imgTemp,len(ContoursD))
ML, MR, cxL, cxR, cyL, cyR, ContList, imgRGB = findingContoursFirst(ContoursD,ret,markers,NewROIListS,RGB1_chan[0,:,:,:],RGB2_chan[0,:,:,:],RGB_chan[0,:,:,:])
PRCL = [cxL,cyL]
PRCR = [cxR,cyR]
imgCropL, imgCropR, imgRGB = CroppingImgNew(RGB1_chan[0,:,:,:],RGB2_chan[0,:,:,:],RGB_chan[0,:,:,:],cxL,cxR,cyL,cyR,ContList[0],ContList[1])
GetAbsFluoValues(gcamp_chan[0,:,:],tdTom_chan[0,:,:],ContList[0],L_GCFluoAbs,L_TdTomFluoAbs)
GetAbsFluoValues(gcamp_chan[0,:,:],tdTom_chan[0,:,:],ContList[1],R_GCFluoAbs,R_TdTomFluoAbs)
CreateAndStoreImages(imgCropR,imgCropL,imgRGB,0)

for k in range(start+1,stop): 

    if ChanAn == 1:
        img = tdTom_chan[k,:,:]
    else :
        img = (tdTom_chan[k,:,:]+gcamp_chan[k,:,:])/2

    shiftedImg, col, row, phase = register_images(imgS, img, return_error=False)
    bpxL, bpyL = CloseCentroidDetect(col,row,PRCL) 
    bpxR, bpyR = CloseCentroidDetect(col,row,PRCR) 
    PRCRC = [bpxR,bpyR]
    PRCLC = [bpxL,bpyL]

    erodeIm = ThresholdingImage(img,k,blurVal,erodeVal)
    cxR, cyR, cxL, cyL, contR, contL = findMostProbableContour(img,erodeIm,PRCRC,PRCLC)
    imgCropL, imgCropR, imgRGB = CroppingImgNew(RGB1_chan[k,:,:,:],RGB2_chan[k,:,:,:],RGB_chan[k,:,:,:],cxL,cxR,cyL,cyR,contL,contR)
    GetAbsFluoValues(gcamp_chan[k,:,:],tdTom_chan[k,:,:],contL,L_GCFluoAbs,L_TdTomFluoAbs)
    GetAbsFluoValues(gcamp_chan[k,:,:],tdTom_chan[k,:,:],contR,R_GCFluoAbs,R_TdTomFluoAbs)
    CreateAndStoreImages(imgCropR,imgCropL,imgRGB,k)

G6_Norm_L, TT_Norm_L = GraphValues(L_GCFluoAbs,L_TdTomFluoAbs,"L")
G6_Norm_R, TT_Norm_R = GraphValues(R_GCFluoAbs,R_TdTomFluoAbs,"R")

#%% DELETE the copies of RGB stacks
os.remove(dataDir + '/registered/' + imgStackFileNameRGB1)
os.remove(dataDir + '/registered/' + imgStackFileNameRGB2)
os.remove(dataDir + '/registered/' + imgStackFileNameRGBtemp)

t2 = time.time()
print('took ', t2-t1, 'seconds')
