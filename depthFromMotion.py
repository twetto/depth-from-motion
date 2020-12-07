import numpy as np
import cv2
from numpy import linalg as LA
from scipy.spatial.transform import Rotation as R
from numba import jit
import cmapy
from timeit import default_timer as timer

@jit
def depth_perception(wall, K, Rt_rotate, Rt_translate, flow, noRotateFlow, depth, tz, diff):
    for point in wall:

        x, y = int(point[0]), int(point[1])
        
        # template wall ideal rotation flow
        [ur, vr, wr] = K @ Rt_rotate @ np.array([point[4], point[5], point[6],1])
        ur = ur/wr
        vr = vr/wr
        wr = 1.0

        # template wall translation flow
        [ut, vt, wt] = K @ Rt_translate @ np.array([point[4], point[5], point[6],1])
        ut = ut/wt
        vt = vt/wt
        wt = 1.0
        idealVector = np.array([ut-x, vt-y]).astype(np.float32)
        mag = np.sqrt(idealVector[0]**2 + idealVector[1]**2)
        
        for i in range(y-hg,y+hg):
            for j in range(x-hg,x+hg):
                if(diff[i,j] >= 20 and mag > 0.1):
                    
                    # rotation compensation
                    noRotateFlow[i,j,0] = flow[i,j,0] - (ur-x)
                    noRotateFlow[i,j,1] = flow[i,j,1] - (vr-y)
                    
                    # depth from motion
                    depth[i,j] = mag / np.sqrt(noRotateFlow[i,j,0]**2+noRotateFlow[i,j,1]**2) * 20

    return depth, noRotateFlow

def no_floor(depth, K, translation, rx, dfloor, srange, width, height):
    if (LA.norm(translation) > 0):
        
        # create floor from known distance and fixed sensing range
        translation = translation.flatten()
        n1 = translation / LA.norm(translation)
        origin = R.from_euler('x', rx).as_matrix() @ np.array([0, dfloor, 0])

        # make unit vector on XZ plane and perpendicular to translation vector
        not_n1 = R.from_euler('x', rx).as_matrix() @ np.array([0, 1, 0])
        if((abs(n1) == not_n1).all()):
            not_n1 = np.array([1.0, 0, 0])
        n2 = np.cross(n1, not_n1)
        n2 /= LA.norm(n2)

        # rotate n1 and n2 to align with YZ plane
        n3 = np.cross(n1, n2)
        n4 = np.cross(n3, np.array([1.0, 0, 0]))
        n4 /= LA.norm(n4)
        theta = np.arccos(np.clip(np.dot(n1, n4), -1.0, 1.0))
        if(n1[0] < 0): theta *= -1
        quat = np.array([np.sin(theta/2)*n3[0], np.sin(theta/2)*n3[1], np.sin(theta/2)*n3[2], np.cos(theta/2)])
        r = R.from_quat(quat).as_matrix()
        n1 = r @ n1
        n2 = r @ n2

        # create meshgrid
        meshx, meshz = 2, 5
        rectx = np.linspace(-srange, srange, meshx)
        rectz = np.linspace(0, srange, meshz)
        rectx, rectz = np.meshgrid(rectx, rectz)
        mx = srange / (meshx-1)
        mz = srange / (meshz-1) / 2

        # generate coordinates for floor
        X, Y, Z = [origin[i] + rectz * n1[i] + rectx * n2[i] for i in [0, 1, 2]]
        X, Y, Z = X.flatten(), Y.flatten(), Z.flatten()
        X = np.delete(X, np.where(Z <= 0))
        Y = np.delete(Y, np.where(Z <= 0))
        Z = np.delete(Z, np.where(Z <= 0))
        
        # floor masking
        masks = np.zeros_like(depth, dtype=np.uint8)
        twenty = masks + 20
        for x, y, z in zip(X, Y, Z):
            p = np.array([x, y, z])
            mask = np.array([p-n1*mz-n2*mx, p-n1*mz+n2*mx, p+n1*mz+n2*mx, p+n1*mz-n2*mx])
            if((mask[:,2] < 0).any()): continue
            u, v, w = K @ mask.T
            u /= w
            v /= w
            w = 1.0
            pts = np.array([u, v], dtype=np.int).T
            l, r, t, b = np.min(pts[:,0]), np.max(pts[:,0]), np.min(pts[:,1]), np.max(pts[:,1])
            l, r = np.clip(np.array([l, r]), 0, width)
            t, b = np.clip(np.array([t, b]), 0, height)
            if(l == r or t == b): continue
            sub_depth = np.abs(depth[t:b,l:r] - z) < 0.7
            masks[t:b,l:r] = sub_depth
        masks_inv = ~masks
        depth *= masks_inv > 0
        twenty *= masks > 0
        depth = depth + twenty
    
    return depth


pose = open('201105/pose.txt', 'r')
wall = np.loadtxt('wall.txt')
cap = cv2.VideoCapture("201105/video.mkv")
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
width = int(width)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
height = int(height)
focalLength = min(height, width)/2/np.tan(35/180*np.pi)
hg = 40//2      # half grid
fps = cap.get(cv2.CAP_PROP_FPS)
#writer = cv2.VideoWriter("output.mkv", cv2.VideoWriter_fourcc(*'MJPG'),fps,(width,height))
#writer_flow = cv2.VideoWriter("output_flow.mkv", cv2.VideoWriter_fourcc(*'MJPG'),fps,(width,height))
#writer_nrflow = cv2.VideoWriter("output_noRotateFlow.mkv", cv2.VideoWriter_fourcc(*'MJPG'),fps,(width,height))
#writer_depth = cv2.VideoWriter("output_depth.mkv", cv2.VideoWriter_fourcc(*'MJPG'),fps,(width,height))
#writer_obs = cv2.VideoWriter("output_obs.mkv", cv2.VideoWriter_fourcc(*'X264'),fps,(width,height))

dis = cv2.DISOpticalFlow_create(0)

K = np.array([[focalLength, 0.0, width/2],
             [0.0, focalLength, height/2],
             [0.0, 0.0, 1.0]])

ret, frame = cap.read()
hsv = np.zeros_like(frame)
hsv[...,1] = 255

curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
prev = curr.copy()

ptime, ptx, pty, ptz, prx, pry = 85.20800, -12.95121, 79.00000, 104.45334, np.radians(43.20006), np.radians(-145.04961)

srange = 3
dfloor = 1.6

start = timer()
frame_count = 0
time_flow = 0
time_rt = 0
time_depth = 0
time_contour = 0

for line in pose:
    time, tx_, ty_, tz_, rx_, ry_ = line.split()
    time = float(time)
    
    tx, ty, tz, rx, ry = float(tx_), float(ty_), float(tz_), np.radians(float(rx_)), np.radians(float(ry_))
    
    # calculate flow
    start_flow = timer()
    diff = cv2.absdiff(prev, curr)
    flow = dis.calc(prev, curr, None, )
    time_flow += timer() - start_flow
    noRotateFlow = np.zeros_like(flow)
    depth = np.zeros_like(curr).astype(np.float32)
    depth += 20
    
    start_rt = timer()
    # rotation in radians
    Rh = np.array([[1,            0,             0],
                  [ 0, np.cos(rx), -np.sin(rx)],
                  [ 0, np.sin(rx),  np.cos(rx)]])
    Ry = np.array([[np.cos(pry-ry), 0, np.sin(pry-ry)],
                  [          0,      1,               0],
                  [-np.sin(pry-ry), 0, np.cos(pry-ry)]])
    Rx = np.array([[1,           0,            0],
                  [ 0, np.cos(-prx), -np.sin(-prx)],
                  [ 0, np.sin(-prx),  np.cos(-prx)]])
    RI = np.eye(3, 3)

    # translation from world frame to camera frame
    xd, yd, zd = tx-ptx, ty-pty, tz-ptz
    xd = -xd
    yd = -yd
    translation = np.array([xd, yd, zd]).reshape(3, 1)
    translation = R.from_euler('x', rx).as_matrix() @ R.from_euler('y', -ry).as_matrix() @ translation
    translation_O = [0, 0, 0]
    translation_O = np.array(translation_O).reshape(3, 1)

    # Rt = [R|t]
    Rt_rotate = np.concatenate((Rh @ Ry @ Rx, translation_O), axis=1)
    Rt_translate = np.concatenate((RI, translation), axis=1)
    time_rt += timer() - start_rt
    
    # get depth frame
    start_depth = timer()
    depth, noRotateFlow = depth_perception(wall, K, Rt_rotate, Rt_translate, flow, noRotateFlow, depth, translation.flatten()[2], diff)
    time_depth += timer() - start_depth

    # filter out floor
    start_contour = timer()
    fdepth = no_floor(depth, K, translation, rx, dfloor, srange, width, height)

    # get contours
    fdepth = fdepth < 3
    fdepth = fdepth.astype(np.uint8)
    fdepth = cv2.dilate(fdepth, np.ones((5,5)))
    fdepth = cv2.erode(fdepth, np.ones((3,3)))
    contours, hierarchy = cv2.findContours(fdepth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    time_contour += timer() - start_contour

    # some outputs
    #mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    #hsv[...,0] = ang*180/np.pi/2
    #hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    #hsv[...,2] = 6 * mag
    #bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    #cv2.imshow('Flow', bgr)
    #writer_flow.write(bgr)

    #mag, ang = cv2.cartToPolar(noRotateFlow[...,0], noRotateFlow[...,1])
    #hsv[...,0] = ang*180/np.pi/2
    #hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    #hsv[...,2] = 6 * mag
    #bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    #cv2.imshow('nrFlow', bgr)
    #writer_nrflow.write(bgr)

    #cv2.imshow("frame", frame)
    #writer.write(frame)

    depth *= 255 / 20
    depth = 255 - depth
    depth = depth * (depth > 0)
    depth = depth.astype(np.uint8)
    depth = cv2.applyColorMap(depth, cmapy.cmap('viridis'))
    cv2.imshow("depth", depth)
    #writer_depth.write(depth)
    
    #fdepth *= 255 / 20
    #fdepth = 255 - fdepth
    #fdepth = fdepth * (fdepth > 0)
    #fdepth = fdepth.astype(np.uint8)
    #fdepth = cv2.applyColorMap(fdepth, cmapy.cmap('viridis'))
    #cv2.imshow("fdepth", fdepth)
    
    showFrame = frame.copy()
    for i in range(len(contours)):
        if(cv2.contourArea(contours[i]) > 2000):
            showFrame = cv2.drawContours(showFrame, contours, i, (0,0,255), -1)
    cv2.imshow("obstacles", showFrame)
    #writer_obs.write(showFrame)
    
    # keyframe selection (not necessary, uncomment if you want to)
    #if(abs(np.degrees(rx-prx)) > 1 or abs(np.degrees(ry-pry)) > 1 or np.sqrt((tx-ptx)**2+(ty-pty)**2+(tz-ptz)**2) > 0.05):
    #    prev = curr
    #    ptime, ptx, pty, ptz, prx, pry = time, tx, ty, tz, rx, ry
    prev = curr
    ptime, ptx, pty, ptz, prx, pry = time, tx, ty, tz, rx, ry
    
    cv2.waitKey(1)
    ret, frame = cap.read()
    end = timer()
    frame_count += 1
    if(frame_count % fps == 0):
        #print(1/(end - start))
        None
    start = end
    if(ret): curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        print("Video ends.")
        break

cap.release()
#writer.release()
#writer_flow.release()
#writer_nrflow.release()
#writer_depth.release()
#writer_obs.release()
cv2.destroyAllWindows()
print("flow time:")
print(time_flow)
print("rigid transform time:")
print(time_rt)
print("compensation & depth time:")
print(time_depth)
print("contouring time:")
print(time_contour)
