import cv2
cv2.namedWindow("Bah")   #workaround for arch linux
cv2.destroyAllWindows() #workaround for arch linux
import cv
import pylab
from SIGBTools import *
import SIGBTools2 as sbt2
import numpy as np
import math
import sys
from scipy.cluster.vq import *
from datetime import datetime
import unittest


"""
SETUP Parameters
"""
kMeans, gradients, hough = False, False, False


inputFile = "Sequences/eye1.avi"
outputFile = "eyeTrackerResult.mp4"

#--------------------------
#         Global variable
#--------------------------
imgOrig = [];
#These are used for template matching
leftTemplate = []
rightTemplate = []
tempSet = False
frameNr =0
props = RegionProps()


def detectPupilKMeans(gray,K=4,distanceWeight=1,reSize=(30,30)):
    smallI = cv2.resize(gray, reSize)
    M,N = smallI.shape
    X,Y = np.meshgrid(range(M),range(N))

    z = smallI.flatten()
    x = X.flatten()
    y = Y.flatten()
    O = len(x)

    #make a feature vectors containing (x,y,intensity)
    features = np.zeros((O,3))
    features[:,0] = z;
    features[:,1] = y/distanceWeight; #Divide so that the distance of position weighs less

    features[:,2] = x/distanceWeight;
    features = np.array(features,'f')
    # cluster data
    centroids,variance = kmeans(features,K)
    #use the found clusters to map
    label,distance = vq(features,centroids)
    # re-create image from
    labelIm = np.array(np.reshape(label,(M,N)))

    # Find the lowest valued class
    thr = 255
    for i in range(K):
        if(centroids[i][0] < thr):
            thr = centroids[i][0]

    return thr


def GetPupil(gray,thr,minArea=4200,maxArea=6000):
    """
    Locate the best matches for pupil in a gray scale image
    """
    props = RegionProps()

    val,binI =cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)
    #cv2.imshow("Aulars",binI)

    binI = cv2.morphologyEx(binI,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_RECT, (10,10)))

    binI = cv2.morphologyEx(binI,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_CROSS,(10,10)))
    #cv2.imshow("Aux", binI)

    #Calculate blobs
    contours, hierarchy = cv2.findContours(binI, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    matches = []
    for con in contours:
        a = cv2.contourArea(con)
        # extend = props.CalcContourProperties(con,properties=["extend"]) # We don't use this because it's not needed

        if(a==0 or a<minArea or a>maxArea):
            continue
        p = cv2.arcLength(con, True)
        m = p/(2.0*math.sqrt(math.pi * a))
        if (m<1.7):
            if(len(con)>=5):
                ellips = cv2.fitEllipse(con)
                matches.append(ellips)
    return matches


def GetGlints(gray,thr,minSize, maxSize):
        ''' Given a gray level image, gray and threshold
        value return a list of glint locations'''

        gray = gray.copy()

        val, binI = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)
        # Opening
        binI = cv2.morphologyEx(binI,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_CROSS,(20,20)))
        contours, hierarchy = cv2.findContours(binI, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


        matches = []
        for con in contours:
            a = cv2.contourArea(con)

            if(a<maxSize and a>minSize):
                # cv2.drawContours(gray2,[con],0,(255,0,0),1)
                if(len(con) >=5):
                    matches.append(cv2.fitEllipse(con))


        # returning a list of candidate ellipsis
        return matches

def zeit(fun,desc=""):
    print desc
    timestart = datetime.now()
    res = fun()
    print datetime.now() - timestart
    return res

def sigurt(v1,v2):
    v2mark = v2 - v1
    return math.fabs(v2mark)

def angle(v1, v2):
    return np.arccos(np.dot(v1,v2)/(length(v1)*length(v2)))

def length(v):
    return np.sqrt(np.dot(v,v))

def getGradientImageInfo(I,C,radius):

    cx,cy = C

    # Preserves edges
    I2 = cv2.bilateralFilter(I.copy(), 10, 4, 4)

    # Using the derivitive kernel from -1 to 1.
    # Dx
    gradientX = cv2.Sobel(I2, cv2.CV_32F, 1, 0)
    # Dy
    gradientY = cv2.Sobel(I2, cv2.CV_32F, 0, 1)

    m,n = I2.shape

    orientImg = np.zeros((m,n))
    # magnitudes fra 0 to 360
    magnitudeImg = np.zeros((m,n), dtype="uint8")

    def orient(gradX,gradY):
        l=((gradX**2)+(gradY**2))**0.5
        return math.atan2(gradX/l,gradY/l)

    # Limit the area we work in. For SPEEEEED
    nRange = range(max(int(cx-radius*1.5),0),min(int(cx+radius*1.5),n))
    mRange = range(max(int(cy-radius*1.5),0),min(int(cy+radius*1.5),m))

    for i in mRange:
        for j in nRange:
            xpow2 = math.pow(int(gradientX[i][j]),2)
            ypow2 = math.pow(int(gradientY[i][j]),2)
            length = int(np.sqrt(xpow2+ypow2))
            magnitudeImg[i][j] = length

            orientImg[i][j] = orient(gradientX[i][j],gradientY[i][j])
    
    #cv2.imshow("aux2",magnitudeImg)
    
    return {"magnitude" : magnitudeImg,
                "dx":gradientX,
                "dy":gradientY,
                "direction":orientImg,
             }

def circleTest(I, pupil):
    I2 = I.copy()
    nPts = 20
    C = pupil[0]
    circleRadius = pupil[1][0] / 2

    gradientInfo = getGradientImageInfo(I,C,circleRadius)
    findEllipseContour(I2,gradientInfo,C,circleRadius,nPts) 

def findEllipseContour(img, gradientInfo, C, circleRadius,nPts=30):
    M,N = img.shape
    P= getCircleSamples(center=C, radius=circleRadius, nPoints=nPts)
    c2 = (int(C[0]),int(C[1]))

    cv2.circle(img,c2,int(circleRadius),(255,0,0))

    gradientImg = gradientInfo["magnitude"]

    for (x,y,dx,dy) in P:
        factor = 1

        deltaX = (x-c2[0])
        deltaY = (y-c2[1])

        vdx = deltaX*factor
        vdy = deltaY*factor

        newX = max(0,min(vdx+x,N-1))
        newY = max(0,min(vdy+y,M-1))
        pp = (int(newX), int(newY))
        unitPP = np.divide(pp,length(pp))#np.sqrt(np.array(pp).dot(pp))) 

        cv2.line(img, c2, pp,(124,144,0))
        cv2.circle(img,pp,1,(255,0,0))

        irisNorm = GetIrisUsingNormals(gradientInfo,c2,circleRadius, pp, unitPP,(deltaX,deltaY),img)

        grads = findMaxGradientValueOnNormal(
            gradientImg,
            c2,(newX,newY),irisNorm)

        if(grads != None):
            cv2.circle(img,grads[0],1,(0,255,0))
            cv2.circle(img,grads[1],1,(0,255,0))

        #cv2.imshow("Aux",img)
        #cv2.imshow("Aux2",gradientImg)

def findMaxGradientValueOnNormal(gradientMagnitude,p1,p2,irisNorm):
    pts = sbt2.getLineCoordinates(p1,p2)

    #normalVals = gradientMagnitude[pts[:,1],pts[:,0]]
    grads = {}
    for p in pts:
        p =(p[0],p[1])
        if p in irisNorm:
            grads[gradientMagnitude[p[1]][p[0]]] = p

    sGrads = sorted(grads,reverse=True)
    if(len(sGrads) < 2): 
        return
    
    r = []
    for i in range(len(sGrads)):
        r.append(grads[sGrads[i]]) 
    return r

## Threshold
## Blob of proper size
## Blob of Shape

def Distance(a, b):
    """
    Calculates distance between two 2d points.

    """
    x1,y1 = a
    x2,y2 = b
    return math.sqrt(math.pow((x2-x1),2)+math.pow((y2-y1),2))


def GetIrisUsingThreshold(gray,pupil):
	''' Given a gray level image, gray and threshold
	value return a list of iris locations'''
	pass


def GetIrisUsingNormals(gradientInfo,pupil,pupilRadius,point, uv, normals,img=None):
    ''' Given a gray level image, gray and the length of the normals, normalLength
	 return a list of iris locations'''

    orientation = gradientInfo["direction"]

    normalAngle = math.atan2(uv[0],uv[1])#angle(uv[0],uv[1])
    pts = getLineCoordinates(pupil,point)

    coords = []
    threshold = 0.7
    for p in pts:
        x = p[0]
        y = p[1]
        diff = sigurt(orientation[y][x],normalAngle)
        if(diff < threshold):
            #cv2.circle(img,(x,y),3,(0,0,0))
            coords.append((x,y))
    return coords

def GetEyeCorners(img, leftTemplate, rightTemplate,pupilPosition=None):
    sliderVals = getSliderVals()
    matchLeft = cv2.matchTemplate(img,leftTemplate,cv2.TM_CCOEFF_NORMED)
    matchRight = cv2.matchTemplate(img,rightTemplate,cv2.TM_CCOEFF_NORMED)
    matchListRight = np.nonzero(matchRight > (sliderVals['templateThr']*0.01))
    matchListLeft =  np.nonzero(matchLeft > (sliderVals['templateThr']*0.01))
    matchList = (matchListLeft,matchListRight)
    return matchList


def FilterPupilGlint(glints, pupils):
    glintList = []
    glintList1 = []
    pupilList = []
    sliderVals = getSliderVals()
    result = []

    for candA in glints:
        for candB in glints:
        #only accepting points with a certain distance to each other.
            if (Distance(candA[0],candB[0])> sliderVals['glintMinDist'] and Distance(candA[0],candB[0]) < sliderVals['glintMaxDist']):
                glintList.append(candA)

    #run through the remaining glints, keeping those that are close to the pupil candidates.
    for glintCand in glintList:
            for pupCand in pupils:
                if(Distance(glintCand[0],pupCand[0])>sliderVals['glint&pubMINDist'] and Distance(glintCand[0],pupCand[0])<sliderVals['glint&pubMAXDist']):
                    glintList1.append(glintCand)

    #run through the pupil candidates keeping those that are close to the fina glints list
    for candP in pupils:
        for glintCand in glintList1:
            if(Distance(candP[0],glintCand[0])>sliderVals['glint&pubMINDist'] and Distance(candP[0],glintCand[0])<sliderVals['glint&pubMAXDist']):
                pupilList.append(candP)
    #sort out the pupils too far away from the found glints.
    return (set(glintList1),set(pupilList))


def detectIrisHough(gray):
    blur = cv2.GaussianBlur(gray, (11,11),11)
    ##Pupil parameters
    slidervals = getSliderVals()
    dp = 6; minDist = 10
    highThr = 30 #High threshold for canny
    accThr = 300; #accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected
    minRadius = slidervals['Hough Iris size']-7;
    maxRadius = slidervals['Hough Iris size']+7;
    #See help for http://opencv.itseez.com/modules/imgproc/doc/feature_detection.html? highlight=houghcircle#cv2.HoughCirclesIn thus
    circles = cv2.HoughCircles(blur,cv2.cv.CV_HOUGH_GRADIENT, dp,minDist, None, highThr,accThr ,minRadius, maxRadius)
    #Print the circles
    gColor = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if (circles !=None):
        #print circles
        all_circles = circles[0]
        M,N = all_circles.shape
        k=1
        for c in all_circles:
            cv2.circle(gColor, (int(c[0]),int(c[1])),c[2], (int(k*255/M),k*128,0))
            K=k+1
        #Circle with max votes
        c=all_circles[0,:]
        cv2.circle(gColor, (int(c[0]),int(c[1])),c[2], (0,0,255),5)
    cv2.imshow("houghIris",gColor)


def detectPupilHough(gray):
    ''' Performs a circular hough transform of the image, gray and shows the  detected circles
    The circe with most votes is shown in red and the rest in green colors '''
    #See help for http://opencv.itseez.com/modules/imgproc/doc/feature_detection.html?highlight=houghcircle#cv2.HoughCircles
    blur = cv2.GaussianBlur(gray, (81,81), 11)
    slidervals = getSliderVals()
    dp = 6; minDist = 30
    highThr = 20 #High threshold for canny
    accThr = 150; #accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected
    minRadius = slidervals['Hough pupil size']-7;
    maxRadius = slidervals['Hough pupil size']+7;
    circles = cv2.HoughCircles(blur,cv2.cv.CV_HOUGH_GRADIENT, dp,minDist, None, highThr,accThr,minRadius, maxRadius)
    #Make a color image from gray for display purposes
    gColor = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if (circles !=None):
        #print circles
        all_circles = circles[0]
        M,N = all_circles.shape
        k=1
        for c in all_circles:
            cv2.circle(gColor, (int(c[0]),int(c[1])),c[2], (int(k*255/M),k*128,0))
            K=k+1
        c=all_circles[0,:]
        cv2.circle(gColor, (int(c[0]),int(c[1])),c[2], (0,0,255),5)
    cv2.imshow("houghPupil",gColor)

# vwriter = cv2.VideoWriter("test.avi",('F','F','V','1'));
def update(I):
        update2(I)

def update2(I):
    '''Calculate the image features and display the result based on the slider values
    :param I:
    '''
    #global drawImg
    global frameNr,drawImg,kMeans,gradients,hough
    sliderVals = getSliderVals()
    img = I#.copy()
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    
    #getGradientImageInfo(gray)

    if(kMeans):
        cv2.setTrackbarPos('pupilThr','Threshold',detectPupilKMeans(gray,8,15))
    else:
        cv2.setTrackbarPos('pupilThr','Threshold',sliderVals["pupilThr"])

    if(hough):
        detectPupilHough(gray)
        detectIrisHough(gray)

# Do the magic  pupils = ellipsis, glints = ellipsis
    pupils = GetPupil(gray,sliderVals['pupilThr'],sliderVals['pupMinSize'],sliderVals['pupMaxSize'])
    glints = GetGlints(gray,sliderVals['glintThr'],sliderVals['glintMinSize'],sliderVals['glintMaxSize'])
    glints, pupils = FilterPupilGlint(glints,pupils)

    for pupil in pupils:
        if(gradients):
            circleTest(gray,pupil)
        cv2.ellipse(img, pupil, (255,0,0),2)

    for glint in glints:
        cv2.ellipse(img, glint,(0,255,0),2)


    #Do template matching
    global leftTemplate
    global rightTemplate
    global pupilPos
    if(tempSet):
        if(pupilPos!=None): #if the pupil position variable is set.
            leftCords,rightCords = GetEyeCorners(imgOrig, leftTemplate, rightTemplate, pupilPos)
        else:
            leftCords,rightCords = GetEyeCorners(imgOrig, leftTemplate, rightTemplate)
        for i in range(len(leftCords[1])):
            x = leftCords[1][i]
            y = leftCords[0][i]
            cv2.rectangle(img,(x,y),((x+len(leftTemplate[0])),(y+len(leftTemplate[1]))),(0,0,255))
        for i in range(len(rightCords[1])):
            pupx,pupy = pupilPos
            x = rightCords[1][i]
            y = rightCords[0][i]
            cv2.rectangle(img,(x,y),((x+len(rightTemplate[0])),(y+len(rightTemplate[1]))),(0,0,255))
    #Display results
    global frameNr,drawImg
    x,y = 10,10
    #setText(img,(x,y),"Frame:%d" %frameNr)

    # for non-windows machines we print the values of the threshold in the original image
    if sys.platform != 'win32':
        step=18
    #    cv2.putText(img, "pupilThr :"+str(sliderVals['pupilThr']), (x, y+step), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.CV_AA)
    #    cv2.putText(img, "glintThr :"+str(sliderVals['glintThr']), (x, y+2*step), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.CV_AA)


#Uncomment these lines as your methods start to work to display the result in the
#original image

#     cv2.imshow("Result", img)

#For Iris detection - Week 2
#circularHough(gray)

    #copy the image so that the result image (img) can be saved in the movie
    drawImg = img.copy()
    #
    #
    cv2.imshow('Result',drawImg)

def printUsage():
    print "Q or ESC: Stop"
    print "SPACE: Pause"
    print "r: reload video"
    print 'm: Mark region when the video has paused'
    print 's: toggle video  writing'
    print 'c: close video sequence'

def run(fileName,resultFile='eyeTrackingResults.avi'):
    global imgOrig, frameNr,drawImg,leftTemplate,rightTemplate,tempSet;
    setupWindowSliders()
    props = RegionProps()
    cap,imgOrig,sequenceOK = getImageSequence(fileName)
    videoWriter = 0;

    frameNr =0
    if(sequenceOK):
        update(imgOrig)
    printUsage()
    saveFrames = False
    while(sequenceOK):
        sliderVals = getSliderVals();
        frameNr=frameNr+1
        ch = cv2.waitKey(1)
        #Select regions
        if(ch==ord('m')):
            if(not sliderVals['Running']):
                roiSelect=ROISelector(imgOrig)
                pts,regionSelected= roiSelect.SelectArea('Select left eye corner',(400,200))
                if(regionSelected):
                    leftTemplate = imgOrig[pts[0][1]:pts[1][1],pts[0][0]:pts[1][0]]
                    tempSet = True

                roiSelect=ROISelector(imgOrig)
                pts,regionSelected= roiSelect.SelectArea('Select right eye corner',(400,200))
                if(regionSelected):
                    rightTemplate = imgOrig[pts[0][1]:pts[1][1],pts[0][0]:pts[1][0]]
                update(imgOrig)
        if ch == 27:
            break
        if (ch==ord('s')):
            if((saveFrames)):
                videoWriter.release()
                saveFrames=False
                print "End recording"
            else:
                imSize = np.shape(imgOrig)
                videoWriter = cv2.VideoWriter(resultFile, cv.CV_FOURCC('D','I','V','3'), 15.0,(imSize[1],imSize[0]),True) #Make a video writer
                saveFrames = True
                print "Recording..."



        if(ch==ord('q')):
            break
        if(ch==32): #Spacebar
            sliderVals = getSliderVals()
            cv2.setTrackbarPos('Stop/Start','Threshold',not sliderVals['Running'])
        if(ch==ord('r')):
            frameNr =0
            sequenceOK=False
            cap,imgOrig,sequenceOK = getImageSequence(fileName)
            update(imgOrig)
            sequenceOK=True

        sliderVals=getSliderVals()
        if(sliderVals['Running']):
            sequenceOK, imgOrig = cap.read()
            if(sequenceOK): #if there is an image
                update(imgOrig)
            if(saveFrames):
                videoWriter.write(drawImg)


        # videoWriter.release



#--------------------------
#         UI related
#--------------------------

def setText(dst, (x, y), s):
	cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.CV_AA)
	cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.CV_AA)


def setupWindowSliders():
    cv2.namedWindow("Result")
    cv2.namedWindow('Threshold')
    #cv2.namedWindow("Temp")
    #cv2.namedWindow("Aux")
    #Threshold value for the pupil intensity
    cv2.createTrackbar('pupilThr','Threshold', 129, 255, onSlidersChange)
    #Threashold value for template matching
    cv2.createTrackbar('templateThr','Threshold', 85, 100, onSlidersChange)
    #Threshold value for the glint intensities
    cv2.createTrackbar('glintThr','Threshold', 240, 255,onSlidersChange)
    #define the minimum and maximum areas of the pupil
    cv2.createTrackbar('pupMinSize','Threshold', 30, 200, onSlidersChange)
    cv2.createTrackbar('pupMaxSize','Threshold', 120,600, onSlidersChange)
    cv2.createTrackbar('glintMinSize','Threshold', 15, 200, onSlidersChange)
    cv2.createTrackbar('glintMaxSize','Threshold', 200, 200, onSlidersChange)
    cv2.createTrackbar('glintMinDist','Threshold', 0, 100, onSlidersChange)
    cv2.createTrackbar('glintMaxDist','Threshold', 100, 100, onSlidersChange)
    cv2.createTrackbar('glint&pubMINDist','Threshold', 0, 500, onSlidersChange)
    cv2.createTrackbar('glint&pubMAXDist','Threshold', 79, 500, onSlidersChange)
    #for the hough transform
    cv2.createTrackbar('Hough Iris size','Threshold', 80, 160, onSlidersChange)
    cv2.createTrackbar('Hough pupil size','Threshold', 25, 50, onSlidersChange)
    #Value to indicate whether to run or pause the video
    cv2.createTrackbar('Stop/Start','Threshold', 0,1, onSlidersChange)

def getSliderVals():
    sliderVals={}
    sliderVals['pupilThr'] = cv2.getTrackbarPos('pupilThr', 'Threshold')
    sliderVals['templateThr'] = cv2.getTrackbarPos('templateThr', 'Threshold')
    sliderVals['glintThr'] = cv2.getTrackbarPos('glintThr', 'Threshold')
    sliderVals['pupMinSize'] = 50*cv2.getTrackbarPos('pupMinSize', 'Threshold')
    sliderVals['pupMaxSize'] = 50*cv2.getTrackbarPos('pupMaxSize', 'Threshold')
    sliderVals['glintMinSize'] = cv2.getTrackbarPos('glintMinSize', 'Threshold')
    sliderVals['glintMaxSize'] = cv2.getTrackbarPos('glintMaxSize', 'Threshold')
    sliderVals['glintMinDist'] = cv2.getTrackbarPos('glintMinDist', 'Threshold')
    sliderVals['glintMaxDist'] = cv2.getTrackbarPos('glintMaxDist', 'Threshold')
    sliderVals['glint&pubMINDist'] = cv2.getTrackbarPos('glint&pubMINDist', 'Threshold')
    sliderVals['glint&pubMAXDist'] = cv2.getTrackbarPos('glint&pubMAXDist', 'Threshold')
    # for hough
    sliderVals['Hough pupil size'] = cv2.getTrackbarPos('Hough pupil size', 'Threshold')
    sliderVals['Hough Iris size'] = cv2.getTrackbarPos('Hough Iris size', 'Threshold')
    sliderVals['Running'] = 1==cv2.getTrackbarPos('Stop/Start', 'Threshold')

    return sliderVals

def onSlidersChange(dummy=None):
	''' Handle updates when slides have changed.
	 This  function only updates the display when the video is put on pause'''
	global imgOrig;
	sv=getSliderVals()
	if(not sv['Running']): # if pause
		update(imgOrig)



#--------------------------
#         main
#--------------------------
run(inputFile)

#img = cv2.imread("Sequences/eye.png")
# img = np.ones()
#img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# print "y: " + y
#cv2.namedWindow("contour")
#cv2.imshow("contour", img)
#cv2.waitKey(0)

