import cv2
cv2.namedWindow("Krustjov")
cv2.destroyAllWindows()
from scipy import ndimage
import numpy as np
from pylab import *
from matplotlib import *
from matplotlib.pyplot import *
from scipy import *
import math
import SIGBTools
import time

rcParams['figure.figsize'] = 30,10

def make_homogen(point):
    return vstack((point, ones((1, point.shape[1]))))

def normalize(point):
    point /= point[-1]
    return point

def frameTrackingData2BoxData(data):
    #Convert a row of points into tuple of points for each rectangle
    pts= [ (int(data[i]),int(data[i+1])) for i in range(0,11,2) ]
    boxes = [];
    for i in range(0,7,2):
        box = tuple(pts[i:i+2])
        boxes.append(box)   
    return boxes


def simpleTextureMap():

    I1 = cv2.imread('data/Images/ITULogo.jpg')
    I2 = cv2.imread('data/Images/ITUMap.bmp')

    #Print Help
    H,Points  = SIGBTools.getHomographyFromMouse(I1,I2,4)
    h,w,d = I2.shape
    overlay = cv2.warpPerspective(I1, H,(w, h))
    M = cv2.addWeighted(I2, 0.5, overlay, 0.5,0)

    cv2.imshow("Overlayed Image",M)
    cv2.waitKey(0)

def showImageandPlot(N):
    #A simple attenmpt to get mouse inputs and display images using matplotlib
    I = cv2.imread('data/groundfloor.bmp')
    drawI = I.copy()
    #make figure and two subplots
    fig = figure(1) 
    ax1  = subplot(1,2,1) 
    ax2  = subplot(1,2,2) 
    ax1.imshow(I) 
    ax2.imshow(drawI)
    ax1.axis('image') 
    ax1.axis('off') 
    points = fig.ginput(5) 
    fig.hold('on')
    
    for p in points:
        #Draw on figure
        subplot(1,2,1)
        plot(p[0],p[1],'rx')
        #Draw in image
        cv2.circle(drawI,(int(p[0]),int(p[1])),2,(0,255,0),10)
    ax2.cla
    ax2.imshow(drawI)
    draw() #update display: updates are usually defered 
    show()
    savefig('data/somefig.jpg')
    cv2.imwrite("data/drawImage.jpg", drawI)


def texturemapGroundFloor():
    """
    Place the texture on every frame of the clip
    """
    fn = 'data/GroundFloorData/SunClipDS.avi'
    cap = cv2.VideoCapture(fn)

    texture = cv2.imread('data/Images/ITULogo.jpg')
    texture = cv2.pyrDown(texture)
    
    mTex,nTex,t = texture.shape
    
    running, imgOrig = cap.read()
    mI,nI,t = imgOrig.shape

    H,Points  = SIGBTools.getHomographyFromMouse(texture,imgOrig,-1)
    h,w,d = imgOrig.shape
    
    while(running):
        running, imgOrig = cap.read()
        if(running):
            h,w,d = imgOrig.shape
            overlay = cv2.warpPerspective(texture, H,(w, h))
            M = cv2.addWeighted(imgOrig, 0.5, overlay, 0.5,0)
            cv2.imshow("Overlayed",M)
            cv2.waitKey(1)
    

def texturemapGridSequence():
    """ Skeleton for texturemapping on a video sequence"""
    fn = 'data/GridVideos/grid1.mp4'
    cap = cv2.VideoCapture(fn)
    drawContours = True;

    texture = cv2.imread('data/Images/ITULogo.jpg')
    texture = cv2.pyrDown(texture)

    mTex,nTex,t = texture.shape

    # Use the corners of the texture
    srcPoints = [
            (float(0.0),float(0.0)),
            (float(nTex),0),
            (float(nTex),float(mTex)),
            (0,mTex)]
    
    #load Tracking data
    running, imgOrig = cap.read()
    mI,nI,t = imgOrig.shape

    cv2.imshow("win2",imgOrig)

    pattern_size = (9, 6)

    idx = [0,8,53,45]
    while(running):
    #load Tracking data
        running, imgOrig = cap.read()
        if(running):
            imgOrig = cv2.pyrDown(imgOrig)
            gray = cv2.cvtColor(imgOrig,cv2.COLOR_BGR2GRAY)

            m,n = gray.shape

            found, corners = cv2.findChessboardCorners(gray, pattern_size)
            if found:
                term = ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1 )
              #  cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)
              #  cv2.drawChessboardCorners(imgOrig, pattern_size, corners, found)
                # Get the points based on the chessboard
                dstPoints = []
                for t in idx:
                    dstPoints.append((int(corners[t,0,0]),int(corners[t,0,1])))
                    #cv2.circle(imgOrig,(int(corners[t,0,0]),int(corners[t,0,1])),10,(255,t,t))
                
                H = SIGBTools.estimateHomography(srcPoints,dstPoints)
                
                overlay = cv2.warpPerspective(texture, H,(n, m))
                
                M = cv2.addWeighted(imgOrig, 0.9, overlay, 0.9,0)
                
                cv2.imshow("win2",M)
            else:
                cv2.imshow("win2",imgOrig)
            cv2.waitKey(1)

def realisticTexturemap(scale=1,point=(200,200)):
    #Load in a homography Hg->m, found by selecting 4 corresponding points,
    #and saved from the floor-mapping method
    homo = np.load("Homography.good.npy")
    
    #A simple attenmpt to get mouse inputs and display images using matplotlib
    T = cv2.imread('data/Images/ITULogo.jpg')
    mp = cv2.imread('data/Images/Ground.jpg')
    M = cv2.imread('data/Images/ITUMap.bmp')

    T = copy(cv2.cvtColor(T,cv2.COLOR_BGR2RGB))
    mp = copy(cv2.cvtColor(mp,cv2.COLOR_BGR2RGB))

    #make figure and two subplots
    fig = figure(1) 
    ax1  = subplot(1,2,1) 
    ax2  = subplot(1,2,2) 
    ax3  = subplot(1,2,2) 
    ax1.imshow(mp) 
    ax2.imshow(T)
    ax1.axis('image') 
    ax1.axis('off') 
    # User selects a point
    point = fig.ginput(1)
    fig.hold('on')
    
    n,m,o = shape(mp)
    n1,m1,o1 = shape(T) 
    
    # Make homogeneous coordinate from selected point (Still in G)
    point = np.matrix([point[0][0],point[0][1],1]).T
    
    #Find the new point based on previously calculated homography
    #(the one saved to a file)
    pointPrime = homo * point
    pointPrime = normalize(pointPrime) # now a point in M (which is the same plane as T)
   
    # We need 4 four points in the destination image. (actually the map)
    # We "make up" these points, by choosing 4 corners of a (made up) rectangle
    # The points lie close to each other, to reduce risk of making a bad
    # homography. 

    aspect = m1 /n1 #aspect ration of texture /logo
    
    delta = 3 #side length of made up rectangle
    x = pointPrime[0][0]
    y = pointPrime[1][0]
    
    # New points calculated from aspect ratio and projected point
    # (These are the made up points) (origin is top-left, x axis is vertical)
    # (x,y)           |    (x,y+(delta*aspect)
    #-------------------------------------------------
    # (x+delta, y)    |    (x+delta, y+(delta*aspect)

    newPoints = [
                [x,y],   
                [x,y+(delta*aspect)],   
                [x+delta,y+(delta*aspect)],   
                [x+delta,y]   
            ]
    # Texture points are the just the corners of the texture
    TPoints = [
                [0,0],
                [0,n1],
                [m1,n1],
                [m1,0]
            ]

    # The center of the texture
    TCenterX = TPoints[0][0] + TPoints[3][0] / 2
    TCenterY = TPoints[0][1] + TPoints[1][1] / 2
    TCenter = (TCenterX,TCenterY)
   
    # A new homography is estimated based on the new points and the
    # texture points. This will go from Texture to Map
    H_MT = SIGBTools.estimateHomography(newPoints, TPoints)
     
    # The homography going from Groundfloor to Texture
    #Hg->m * Hm->t is Hg->t. we take the inverse of that, Ht->g
    H_TG = homo.dot(H_MT).I
    
    # phtg becomes the T's origin, projected to G
    phtg = normalize(H_TG.dot(np.matrix([0,0,1]).T))
    #sigurtVector (really?) is the translation vector from
    # the projected texture origin, to where the user clicked in G
    sigurtVector = point - phtg

    # Translation matrix a matrix that will apply sigurtVector as translation.
    # i.e. will translate the projected T, so that T's top-left corner is at the
    # clicked point
    sigurtMat = np.matrix([
            [1.,0.,sigurtVector[0][0]],
            [0.,1.,sigurtVector[1][0]],
            [0.,0.,1.]
            ]);

    # normalize (sigurtMat * (Ht->g * centerOfTInT)) is the center of T, projected and
    # moved so that T's origin would be at the clicked point in G
    # It is therefore also the vector from G's origin to the center of T in G
    GCenter = normalize(sigurtMat * H_TG * np.matrix([TCenter[0],TCenter[1],1]).T)

    # The vector from T's center (in G) to G's origin
    vectorToOrigin = [GCenter[0][0]*-1,GCenter[1][0]*-1]

    #The matrix that will move the projected texture's center to G's origin
    sigurtFlyt = np.matrix([
            [1,0,vectorToOrigin[0]],
            [0,1,vectorToOrigin[1]],
            [0,0,1]
        ]);

    #scaling matrix, as specified by method parameter
    sigurtScale = np.matrix([
            [scale,0,0],
            [0,scale,0],
            [0,0,1]
        ]);

    # (sigurtScale * sigurtFlyt) : make a matrix that moves T in G to G's origin
    # sigurtFlyt.I * (sigurtScale * sigurtFlyt) : this matrix will moove the
    # texture back again.
    scaleMat = sigurtFlyt.I * (sigurtScale * sigurtFlyt)

    cv2.circle(mp,(GCenter[0],GCenter[1]),10,(255,0,0))
    #(sigurtMat.dot(H_TG)) will project T to G, so that the corner of T is at
    #the clicked point
    #scaleMat.dot(sigurtMat.dot(H_TG) will do that, and move T to G's origin,
    #scale it and move it back
    warp = cv2.warpPerspective(T, scaleMat.dot(sigurtMat.dot(H_TG)), (m,n))

    mp = cv2.addWeighted(mp, 0.9, warp, 0.9, 0)
     
    ax1.imshow(mp)
    ax2.imshow(warp)
    draw() #update display: updates are usually defered 
    show()
    #savefig('data/somefig.jpg')
    #cv2.imwrite("drawImage.jpg", I)

def displayTrace(homo, pnts, space):
    a = [pnts[0][0],pnts[0][1],1]
    b = [pnts[1][0],pnts[1][1],1]
    a,b = homo.dot(a), homo.dot( b)
    a,b = normalize(a), normalize(b)

    #cv2.circle(space, (int(a[0]),int(a[1])),2,(0,0,0))
    #cv2.circle(space, (int(b[0]),int(b[1])),2,(0,0,0))
    cv2.circle(space, (int(b[0]),int(b[1]-((b[1]-a[1])/2))),2,(0,0,0))
    cv2.imshow("aux", space)

def showFloorTrackingData():
    #Load videodata
    fn = "data/GroundFloorData/SunClipDS.avi"
    I2 = cv2.imread('data/Images/ITUMap.bmp')
    cap = cv2.VideoCapture(fn)
    running, imgOrig = cap.read()
    #homo, pts = SIGBTools.getHomographyFromMouse(imgOrig, I2)
    homo = np.load("Homography.good.npy")
    #np.save("Homography", homo)
    
    #load Tracking data
    running, imgOrig = cap.read()
    dataFile = np.loadtxt("data/GroundFloorData/trackingdata.dat")
    m,n = dataFile.shape
    
    fig = figure()
    for k in range(m):
        running, imgOrig = cap.read() 
        if(running):
            boxes= frameTrackingData2BoxData(dataFile[k,:])
            boxColors = [(255,0,0),(0,255,0),(0,0,255)]
            for k in range(0,3):
                aBox = boxes[k]
                cv2.rectangle(imgOrig, aBox[0], aBox[1], boxColors[k])
            cv2.imshow("boxes",imgOrig);
            displayTrace(homo, [boxes[1][0], boxes[1][1]], I2)
            #time.sleep(0.001)
            l=cv2.waitKey(10)

def angle_cos(p0, p1, p2):
    d1, d2 = p0-p1, p2-p1
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def findSquares(img,minSize = 2000,maxAngle = 1):
    """ findSquares intend to locate rectangle in the image of minimum area, minSize, and maximum angle, maxAngle, between 
    sides"""
    squares = []
    contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
         cnt_len = cv2.arcLength(cnt, True)
         cnt = cv2.approxPolyDP(cnt, 0.08*cnt_len, True)
         if len(cnt) == 4 and cv2.contourArea(cnt) > minSize and cv2.isContourConvex(cnt):
             cnt = cnt.reshape(-1, 2)
             max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
             if max_cos < maxAngle:
                 squares.append(cnt)
    return squares

def DetectPlaneObject(I,minSize=1000):
      """ A simple attempt to detect rectangular 
      color regions in the image"""
      HSV = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
      h = HSV[:,:,0].astype('uint8')
      s = HSV[:,:,1].astype('uint8')
      v = HSV[:,:,2].astype('uint8')
      
      b = I[:,:,0].astype('uint8')
      g = I[:,:,1].astype('uint8')
      r = I[:,:,2].astype('uint8')
     
      # use red channel for detection.
      s = (255*(r>230)).astype('uint8')
      iShow = cv2.cvtColor(s, cv2.COLOR_GRAY2BGR)
      cv2.imshow('ColorDetection',iShow)
      squares = findSquares(s,minSize)
      return squares
  
def texturemapObjectSequence():
    """ Poor implementation of simple texturemap """
    fn = 'data/BookVideos/Seq3_scene.mp4'
    cap = cv2.VideoCapture(fn) 
    drawContours = True;
    
    texture = cv2.imread('data/images/ITULogo.jpg')
    #texture = cv2.transpose(texture)
    mTex,nTex,t = texture.shape
    
    #load Tracking data
    running, imgOrig = cap.read()
    mI,nI,t = imgOrig.shape
    
    while(running):
        for t in range(20):
            running, imgOrig = cap.read() 
        
        if(running):
            squares = DetectPlaneObject(imgOrig)
            
            for sqr in squares:
                 #Do texturemap here!!!!
                 #TODO
                 
                 if(drawContours):                
                     for p in sqr:
                         cv2.circle(imgOrig,(int(p[0]),int(p[1])),3,(255,0,0)) 
                 
            
            if(drawContours and len(squares)>0):    
                cv2.drawContours( imgOrig, squares, -1, (0, 255, 0), 3 )

            cv2.circle(imgOrig,(100,100),10,(255,0,0))
            cv2.imshow("Detection",imgOrig)
            cv2.waitKey(1)

#showFloorTrackingData()
#simpleTextureMap()
realisticTexturemap(20)
#texturemapGridSequence()
#texturemapGroundFloor()
# vim: ts=4:shiftwidth=4:expandtab:tw=80
