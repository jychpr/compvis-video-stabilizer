import numpy as np
import cv2
SMOOTHING_RADIUS = 100 #100-1000 optimal and quick to compute #steps 1000 / 10000 / 100000


### FUNCTION 
def movingAverage(curve, radius):
    window_size = 2 * radius + 1
    fltr = np.ones(window_size)/window_size 
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge') #padding
    curve_smoothed = np.convolve(curve_pad, fltr, mode='same')
    curve_smoothed = curve_smoothed[radius:-radius] #remove padding
    return curve_smoothed

def smooth(trajectory):
    smoothed_trajectory = np.copy(trajectory)
    for i in range(3):
        smoothed_trajectory[:,i] = movingAverage(trajectory[:,i], 
                                                        radius=SMOOTHING_RADIUS)
    return smoothed_trajectory

def fixBorder(frame):
    shp = frame.shape
    Trnsfrm = cv2.getRotationMatrix2D((shp[1]/2, shp[0]/2), 0, 1.05)
    frame = cv2.warpAffine(frame, Trnsfrm, (shp[1], shp[0]))
    return frame


### PREPROCESSING STAGE
#input video
vidcap = cv2.VideoCapture('mcp2_rz530.mp4') #insert filename input video here

#get frames and fps count
n_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = vidcap.get(cv2.CAP_PROP_FPS)
print("frame count in video : ", n_frames)

#get video dimension
width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print('width : ', width)
print('height : ', height)

#output video
fourcc = cv2.VideoWriter_fourcc(*'MPEG')
outvid = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height)) #insert filename output video here


### PROCESS STAGE
#define transformation array
transforray = np.zeros((n_frames-1,3), np.float64) #might try float32, float64

#greyscaling first or previous frame (?)
ret, prev = vidcap.read()
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

#recurring process
for i in range(n_frames-2):
    #determining strong corner on frame as points
    prev_point = cv2.goodFeaturesToTrack(prev_gray, 
                                            maxCorners=500,
                                            qualityLevel=0.01,
                                            minDistance=30,
                                            blockSize=3,
                                            useHarrisDetector=None)
    retval, crrnt = vidcap.read()
    if (retval == False):
        break
    crrnt_gray = cv2.cvtColor(crrnt, cv2.COLOR_BGR2GRAY)

    #track feature points
    crrnt_point, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, 
                                                            crrnt_gray, 
                                                            prev_point, 
                                                            None)
    index = np.where(status == 1)[0]
    prev_point = prev_point[index]
    crrnt_point = crrnt_point[index]
    assert prev_point.shape == crrnt_point.shape

    #create transformation matrix
    mtrx, inliers = cv2.estimateAffine2D(prev_point, crrnt_point,
                                    inliers=None,
                                    confidence=0.99, #0.95-0.99
                                    method=cv2.RANSAC, #ransac or LMEDS
                                    ransacReprojThreshold=8) #unknown

    #decompose matrix transformation
    dx = mtrx[0,2]
    dy = mtrx[1,2]
    da = np.arctan2(mtrx[1,0], mtrx[0,0])
    transforray[i] = [dx, dy, da]

    prev_gray = crrnt_gray

    print("Frame : " + str(i) + "/" + str(n_frames)
            + " - Tracked points : " + str(len(prev_point)))
            
trajectory = np.cumsum(transforray, axis=0)
smoothed_trajectory = smooth(trajectory)
difference = smoothed_trajectory - trajectory
transforray_smooth = transforray + difference

#apply transformation to video
vidcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
for i in range(n_frames-2):
    retval2, frame = vidcap.read()
    if (retval2 == False):
        break
    
    #extract transformation from new transformation array
    dx = transforray_smooth[i,0]
    dy = transforray_smooth[i,1]
    da = transforray_smooth[i,2]

    #remake transformation matrix according new values
    mtrx = np.zeros((2,3), np.float64)  #might try float32
    mtrx[0,0] = np.cos(da)
    mtrx[0,1] = -np.sin(da)
    mtrx[1,0] = np.sin(da)
    mtrx[1,1] = np.cos(da)
    mtrx[0,2] = dx
    mtrx[1,2] = dy

    frame_stabilized = cv2.warpAffine(frame, mtrx, (width, height))
    frame_stabilized = fixBorder(frame_stabilized)
    frame_out = cv2.hconcat([frame, frame_stabilized])
    #if(frame_out.shape[1] > 1920):
    #    frame_out = cv2.resize(frame_out, (frame_out.shape[1]/2, 
    #                                        frame_out.shape[0]/2))
    
    cv2.imshow("Before and After ", frame_out)
    cv2.waitKey(10)
outvid.write(frame_out)

vidcap.release()
outvid.release()

cv2.destroyAllWindows()


