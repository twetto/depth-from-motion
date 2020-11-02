import numpy as np
import cv2
from numpy import linalg as LA
from numba import jit
import cmapy
from timeit import default_timer as timer

@jit
def depth_perception(wall, K, Rt_rotate, Rt_translate, flow, noRotateFlow, depth, zd, diff):
    for point in wall:

        x, y = int(point[0]), int(point[1])
        if(np.mean(diff[y-hg:y+hg,x-hg:x+hg]) < 5): continue
        
        # rotation
        [ur, vr, wr] = K @ Rt_rotate @ np.array([point[4], point[5], point[6],1])
        ur = ur/wr
        vr = vr/wr
        wr = 1.0

        # translation
        [ut, vt, wt] = K @ Rt_translate @ np.array([point[4], point[5], point[6],1])
        ut = ut/wt
        vt = vt/wt
        wt = 1.0
        idealVector = np.array([ut-x, vt-y]).astype(np.float32)
        mag = LA.norm(idealVector)
        
        for i in range(y-hg,y+hg):
            for j in range(x-hg,x+hg):
                if(diff[i,j] >= 20 and mag > 0.1):
                    noRotateFlow[i,j,0] = flow[i,j,0] + (ur-x)
                    noRotateFlow[i,j,1] = flow[i,j,1] + (vr-y)
                    depth[i,j] = mag / np.sqrt(noRotateFlow[i,j,0]**2+noRotateFlow[i,j,1]**2) * 480

        '''
        flowVector = np.array([np.mean(noRotateFlow[y-40:y+40,x-40:x+40,0]),np.mean(noRotateFlow[y-40:y+40,x-40:x+40,1])])
        if(LA.norm(idealVector) > 0.1 and LA.norm(flowVector) > 3):
            uiVector = idealVector / LA.norm(idealVector)
            ufVector = flowVector / LA.norm(flowVector)
            outlier[y-40:y+40,x-40:x+40] = np.arccos(np.clip(np.dot(uiVector, ufVector), -1.0, 1.0))
        '''

    return depth

pose = open('201102/pose.txt', 'r')
wall = np.loadtxt('wall.txt')
cap = cv2.VideoCapture("201102/video.mkv")
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
width = int(width)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
height = int(height)
focalLength = min(height, width)/2/np.tan(35/180*np.pi)
hg = 40//2      # half grid
fps = cap.get(cv2.CAP_PROP_FPS)
print(width,height,fps,hg)
writer = cv2.VideoWriter("output.mkv", cv2.VideoWriter_fourcc(*'MJPG'),fps,(width,height))
writer_flow = cv2.VideoWriter("output_flow.mkv", cv2.VideoWriter_fourcc(*'MJPG'),fps,(width,height))
writer_nrflow = cv2.VideoWriter("output_noRotateFlow.mkv", cv2.VideoWriter_fourcc(*'MJPG'),fps,(width,height))
#writer_outlier = cv2.VideoWriter("output_outlier.mkv", cv2.VideoWriter_fourcc(*'X264'),fps,(640,480))
writer_depth = cv2.VideoWriter("output_depth.mkv", cv2.VideoWriter_fourcc(*'MJPG'),fps,(width,height))

dis = cv2.DISOpticalFlow_create(0)

K = np.array([[focalLength, 0.0, width/2+0.5],
             [0.0, focalLength, height/2+0.5],
             [0.0, 0.0, 1.0]])

ret, frame = cap.read()
hsv = np.zeros_like(frame)
hsv[...,1] = 255

curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
prev = curr.copy()

ptime, ptx, pty, ptz, prx, pry = 261.03900, 785.56394, 4.00000, -206.83414, np.radians(-1.35001), np.radians(176.10004)

start = timer()
frame_count = 0
time_flow = 0
time_rt = 0
time_depth = 0
for line in pose:
    time, tx_, ty_, tz_, rx_, ry_ = line.split()
    time = float(time)
    
    tx, ty, tz, rx, ry = float(tx_), float(ty_), float(tz_), np.radians(float(rx_)), np.radians(float(ry_))
    
    # calculate flow
    start_flow = timer()
    diff = cv2.absdiff(prev, curr)
    flow = dis.calc(prev, curr, None, )
    time_flow += timer() - start_flow
    #showFrame = frame.copy()
    noRotateFlow = np.zeros_like(flow)
    depth = np.zeros_like(curr).astype(np.float32)
    depth += 480
    #outlier = np.zeros_like(curr).astype(np.float32)
    
    start_rt = timer()
    # rotation in radius
    Rh = np.array([[1,            0,             0],
                  [ 0, np.cos(prx), -np.sin(prx)],
                  [ 0, np.sin(prx),  np.cos(prx)]])
    Ry = np.array([[np.cos(ry-pry), 0, np.sin(ry-pry)],
                  [          0,      1,               0],
                  [-np.sin(ry-pry), 0, np.cos(ry-pry)]])
    Rx = np.array([[1,           0,            0],
                  [ 0, np.cos(-rx), -np.sin(-rx)],
                  [ 0, np.sin(-rx),  np.cos(-rx)]])
    RI = np.eye(3, 3)

    # translation in pixels (1 meter = 16 pixels in Minecraft)
    xd, yd, zd = tx-ptx, ty-pty, tz-ptz
    xd *= 16
    yd *= 16
    zd *= 16
    zd = -zd
    translation = [np.sqrt(xd**2+yd**2+zd**2)*np.sin(np.arctan2(np.sqrt(xd**2+zd**2),yd)-rx)*np.cos(np.pi/2+np.arctan2(-xd,zd)+ry),
            np.sqrt(xd**2+yd**2+zd**2)*np.cos(np.arctan2(np.sqrt(xd**2+zd**2),yd)-rx),
            np.sqrt(xd**2+yd**2+zd**2)*np.sin(np.arctan2(np.sqrt(xd**2+zd**2),yd)-rx)*np.sin(np.pi/2+np.arctan2(-xd,zd)+ry)]
    translation = np.array(translation).reshape(3, 1)
    translation_O = [0, 0, 0]
    translation_O = np.array(translation_O).reshape(3, 1)

    # Rt = [R|t]
    Rt_rotate = np.concatenate((Rh @ Ry @ Rx, translation_O), axis=1)
    Rt_translate = np.concatenate((RI, translation), axis=1)
    time_rt += timer() - start_rt
    
    # get depth frame
    start_depth = timer()
    depth = depth_perception(wall, K, Rt_rotate, Rt_translate, flow, noRotateFlow, depth, zd, diff)
    time_depth += timer() - start_depth

    # some outputs
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    #hsv[...,2] = 6 * mag
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    cv2.imshow('Flow', bgr)
    writer_flow.write(bgr)

    mag, ang = cv2.cartToPolar(noRotateFlow[...,0], noRotateFlow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    #hsv[...,2] = 6 * mag
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    cv2.imshow('nrFlow', bgr)
    writer_nrflow.write(bgr)

    cv2.imshow("frame", frame)
    writer.write(frame)

    depth *= 255 / 480
    depth = 255 - depth
    depth = depth * (depth > 0)
    depth = depth.astype(np.uint8)
    depth = cv2.applyColorMap(depth, cmapy.cmap('viridis'))
    cv2.imshow("depth", depth)
    writer_depth.write(depth)
    
    '''
    #outlier = cv2.normalize(outlier,None,0,255,cv2.NORM_MINMAX)
    outlier *= 255 / np.pi
    outlier = outlier.astype(np.uint8)
    cv2.imshow("outlier", outlier)
    outlier = cv2.cvtColor(outlier, cv2.COLOR_GRAY2BGR)
    writer_outlier.write(outlier)
    '''
    
    prev = curr
    ptime, ptx, pty, ptz, prx, pry = time, tx, ty, tz, rx, ry
    cv2.waitKey(1)
    ret, frame = cap.read()
    end = timer()
    frame_count += 1
    if(frame_count % fps == 0): print(1/(end - start))
    start = end
    if(ret): curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else: print("Something went wrong with the video...")

cap.release()
writer.release()
writer_flow.release()
writer_nrflow.release()
writer_depth.release()
#writer_outlier.release()
cv2.destroyAllWindows()
print("flow time:")
print(time_flow)
print("rigid transform time:")
print(time_rt)
print("compensation & depth time:")
print(time_depth)
