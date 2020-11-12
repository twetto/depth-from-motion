#!/usr/bin/python

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

width = 640
height = 480
grid = 40
focalLength = min(height, width)/2/np.tan(35/180*np.pi)
print(focalLength)

K = np.array([[focalLength, 0, width/2, 0],
             [0, focalLength, height/2, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]])
K_inv = LA.inv(K)
print(K_inv)

rotation = [0, 0, 0]
rx = rotation[0]
ry = rotation[1]
rz = rotation[2]
Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
              [ np.sin(rz),  np.cos(rz), 0],
              [          0,           0, 1]])
Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
              [          0, 1,          0],
              [-np.sin(ry), 0, np.cos(ry)]])
Rx = np.array([[1,          0,           0],
              [ 0, np.cos(rx), -np.sin(rx)],
              [ 0, np.sin(rx),  np.cos(rx)]])

translation = [0, 0, 0]
translation = np.array(translation).reshape(3, 1)

Rt = np.concatenate((Rz @ Ry @ Rx, translation), axis=1)
print(Rt)
Rt = np.concatenate((Rt, np.array([[0, 0, 0, 1]])), axis=0)

Rt_inv = LA.inv(Rt)
print(Rt_inv)

frameWallPair = np.zeros(height//grid*width//grid*4*2)
frameWallPair = frameWallPair.reshape(height//grid*width//grid, 4*2)

for u in range(grid//2, width, grid):
    for v in range(grid//2, height, grid):
        p_frame = np.array([u, v, 1, 1]).reshape(4, 1)
        p_wall = Rt_inv @ K_inv @ p_frame
        p_wall = p_wall / p_wall[2] * 20

        # generate ball-like wall
        '''
        a = p_wall[0] / p_wall[1]
        b = p_wall[0] / p_wall[2]
        p_wall[0] = np.sign(p_wall[0]) * np.sqrt(80000**2/(1+1/a**2+1/b**2))
        p_wall[1] = p_wall[0] / a
        p_wall[2] = p_wall[0] / b
        '''
        
        frameWallPair[u//grid*(height//grid)+v//grid] = np.concatenate((p_frame, p_wall), axis=None)

fig = plt.figure()
ax = Axes3D(fig)
frameWallPair = frameWallPair.transpose()
ax.scatter(frameWallPair[4], frameWallPair[5], frameWallPair[6], c='b')
ax.scatter(frameWallPair[0], frameWallPair[1], frameWallPair[2], c='r')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
axisEqual3D(ax)
plt.show()
frameWallPair = frameWallPair.transpose()
np.savetxt('wall.txt', frameWallPair)
