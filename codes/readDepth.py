import cv2
import numpy as np
import pyrealsense2 as rs
import os.path
from os import path
from PIL import Image
import imageio

from scipy.io import savemat, loadmat
from datagen import DataGen

import pickle


#load annot
jnt = []
batch = []
annot = []
mat_path = 'annotator\\test\\joint_data.mat'
if path.exists(mat_path):
    annot_data = loadmat(mat_path)
    data = annot_data['joint_uvd']
    if data.shape[0] == 1 and data.shape[1] > 0:
        for i in range(data.shape[1]):
            batch.append(data[0,i,:,:])
        # annot = annot_data['joint_uvd']
print(np.array(batch).shape)

depth_image = loadmat('test\\depth_1_0000007.mat')['1']

for joint in batch[6]:
  x = joint[0]
  y = joint[1]
  z = depth_image[y,x]
  print(z)
