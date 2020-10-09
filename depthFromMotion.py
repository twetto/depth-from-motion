import numpy as np
import cv2
from numpy import linalg as LA
from numba import jit
import cmapy
from timeit import default_timer as timer

@jit
def depth_perception(wall, K, Rt_rotate, Rt_translate, flow, noRotateFlow, depth, zd, diff):
    for point in wall:
        
        # rotation
        [u, v, w] = K @ Rt_rotate @ np.array([point[4], point[5], point[6],1])
        u = u/w
        v = v/w
        w = 1.0

        x, y = int(point[0]), int(point[1])
        noRotateFlow[y-hg:y+hg,x-hg:x+hg,0] = flow[y-hg:y+hg,x-hg:x+hg,0] + (u-x)
        noRotateFlow[y-hg:y+hg,x-hg:x+hg,1] = flow[y-hg:y+hg,x-hg:x+hg,1] + (v-y)
        noRotateFlow[y-hg:y+hg,x-hg:x+hg,0] *= diff[y-hg:y+hg,x-hg:x+hg] > 5
        noRotateFlow[y-hg:y+hg,x-hg:x+hg,1] *= diff[y-hg:y+hg,x-hg:x+hg] > 5

        # translation
        [u, v, w] = K @ Rt_translate @ np.array([point[4], point[5], point[6],1])
        u = u/w
        v = v/w
        w = 1.0
        idealVector = np.array([u-x, v-y]).astype(np.float32)
        
        mag = LA.norm(idealVector)
        if(mag > 0.001):
            depth[y-hg:y+hg,x-hg:x+hg] = mag / np.sqrt(noRotateFlow[y-hg:y+hg,x-hg:x+hg,0]**2+noRotateFlow[y-hg:y+hg,x-hg:x+hg,1]**2) * 480
        else:
            depth[y-hg:y+hg,x-hg:x+hg] = 480
        
        '''
        flowVector = np.array([np.mean(noRotateFlow[y-40:y+40,x-40:x+40,0]),np.mean(noRotateFlow[y-40:y+40,x-40:x+40,1])])
        if(LA.norm(idealVector) > 0.1 and LA.norm(flowVector) > 3):
            uiVector = idealVector / LA.norm(idealVector)
            ufVector = flowVector / LA.norm(flowVector)
            outlier[y-40:y+40,x-40:x+40] = np.arccos(np.clip(np.dot(uiVector, ufVector), -1.0, 1.0))
        '''

    return depth

pose = open('pose.txt', 'r')
wall = np.loadtxt('wall.txt')
cap = cv2.VideoCapture("2020-10-07_17-26-21.mkv")
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

video_skip = int(4.15*fps)
pose_skip = 91.498

dis = cv2.DISOpticalFlow_create(0)

K = np.array([[focalLength, 0.0, width/2],
             [0.0, focalLength, height/2],
             [0.0, 0.0, 1.0]])

# skip unwanted frames
for i in range(video_skip):
    ret, frame = cap.read()

ret, frame = cap.read()
hsv = np.zeros_like(frame)
hsv[...,1] = 255

curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
prev = curr.copy()

frame_count = 0
p_tx, p_ty, p_tz, p_rx, p_ry = 6.28410, 4.00000, 233.20225, np.radians(1.05000), np.radians(-177.14551)
start = timer()
for line in pose:
    time, tx_, ty_, tz_, rx_, ry_ = line.split()
    time = float(time)
    time -= pose_skip

    # skip unwanted poses
    if(time < 0): continue

    # skip misaligned poses
    if(abs(time - frame_count/fps) > 0.006): continue

    tx, ty, tz, rx, ry = float(tx_), float(ty_), float(tz_), np.radians(float(rx_)), np.radians(float(ry_))
    
    # calculate flow & others
    diff = cv2.absdiff(prev, curr)
    flow = dis.calc(prev, curr, None, )
    #flow[...,0] = flow[...,0] * (diff > 3)
    #flow[...,1] = flow[...,1] * (diff > 3)
    #showFrame = frame.copy()
    #attn = np.zeros_like(curr)
    noRotateFlow = np.zeros_like(flow)
    depth = np.zeros_like(curr).astype(np.float32)
    #outlier = np.zeros_like(curr).astype(np.float32)
    
    # rotation in radius
    Rh = np.array([[1,            0,             0],
                  [ 0, np.cos(p_rx), -np.sin(p_rx)],
                  [ 0, np.sin(p_rx),  np.cos(p_rx)]])
    Ry = np.array([[np.cos(ry-p_ry), 0, np.sin(ry-p_ry)],
                  [          0,      1,               0],
                  [-np.sin(ry-p_ry), 0, np.cos(ry-p_ry)]])
    Rx = np.array([[1,           0,            0],
                  [ 0, np.cos(-rx), -np.sin(-rx)],
                  [ 0, np.sin(-rx),  np.cos(-rx)]])
    RI = np.eye(3, 3)

    # translation in pixels
    xd, yd, zd = tx-p_tx, ty-p_ty, tz-p_tz
    #xd *= 100
    #yd *= 100
    #zd *= 100
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
    #print(Rt_translate)
    
    # Calculate displacement of the virtual wall inside the frame
    # with 120*120 pixel grids
    depth = depth_perception(wall, K, Rt_rotate, Rt_translate, flow, noRotateFlow, depth, zd, diff)
    '''
    for point in wall:
        
        # rotation
        [u, v, w] = K @ Rt_rotate @ np.array([point[4], point[5], point[6],1])
        u = u/w
        v = v/w
        w = 1.0

        x, y = int(point[0]), int(point[1])
        #noRotateFlow[y-hg:y+hg,x-hg:x+hg,0] = flow[y-hg:y+hg,x-hg:x+hg,0] + (u-x)
        #noRotateFlow[y-hg:y+hg,x-hg:x+hg,1] = flow[y-hg:y+hg,x-hg:x+hg,1] + (v-y)
        #noRotateFlow[y-hg:y+hg,x-hg:x+hg,0] *= diff[y-hg:y+hg,x-hg:x+hg] > 1
        #noRotateFlow[y-hg:y+hg,x-hg:x+hg,1] *= diff[y-hg:y+hg,x-hg:x+hg] > 1
        flow[y-hg:y+hg,x-hg:x+hg,0] = flow[y-hg:y+hg,x-hg:x+hg,0] + (u-x)
        flow[y-hg:y+hg,x-hg:x+hg,1] = flow[y-hg:y+hg,x-hg:x+hg,1] + (v-y)
        flow[y-hg:y+hg,x-hg:x+hg,0] *= diff[y-hg:y+hg,x-hg:x+hg] > 1
        flow[y-hg:y+hg,x-hg:x+hg,1] *= diff[y-hg:y+hg,x-hg:x+hg] > 1

        # translation
        [u, v, w] = K @ Rt_translate @ np.array([point[4], point[5], point[6],1])
        u = u/w
        v = v/w
        w = 1.0
        idealVector = np.array([u-x, v-y]).astype(np.float32)
        
        mag = LA.norm(idealVector)
        if(mag > 0.001):
            #depth[y-hg:y+hg,x-hg:x+hg] = mag / np.sqrt(noRotateFlow[y-hg:y+hg,x-hg:x+hg,0]**2+noRotateFlow[y-hg:y+hg,x-hg:x+hg,1]**2) * 480
            depth[y-hg:y+hg,x-hg:x+hg] = mag / np.sqrt(flow[y-hg:y+hg,x-hg:x+hg,0]**2+flow[y-hg:y+hg,x-hg:x+hg,1]**2) * 480
        else:
            depth[y-hg:y+hg,x-hg:x+hg] = 480
    '''


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

    #cv2.imshow("attn", attn)
    #attn = cv2.cvtColor(attn, cv2.COLOR_GRAY2BGR)
    #writer_attn.write(attn)
    
    #depth = cv2.normalize(depth,None,0,255,cv2.NORM_MINMAX)
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
    p_tx, p_ty, p_tz, p_rx, p_ry = tx, ty, tz, rx, ry
    cv2.waitKey(1)
    ret, frame = cap.read()
    frame_count += 1
    end = timer()
    if(frame_count % fps == 0): print(1/(end - start))
    start = end
    if(ret): curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        print("Something went wrong with the video...")
        break

cap.release()
writer.release()
writer_flow.release()
writer_nrflow.release()
writer_depth.release()
#writer_outlier.release()
cv2.destroyAllWindows()

