import cv2
import numpy as np
import pyrealsense2 as rs
import os.path
from os import path
from PIL import Image
import imageio

from scipy.io import savemat, loadmat
from datagen import DataGen
i_frames = [
  88,# i2l.bag current 0
92,
127,
132,
136,
147,
172,
219,
283,
59, # i5l.bag, current 9
142,
172,
182,
183,
197,
202,
252,
257,
262,
266,
270,
332,
337,
362,
377,
382,
392,
397,
402,
32, # r2r.bag, current29
37,
62,
67,
73,
92,
113,
142,
147,
173,
177,
182,
187,
243,
247,
332,
367,
371,
52, # v2r.bag, current 47
57,
87,
92,
132,
168,
172,
177,
182,
252,
297,
302,
337,
63, # v4l.bag, current 60
87,
93,
167,
172,
200,
232,
260,
]
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

dataset_path = "test.bag"
pth = 'annotator\\test'
fr = 15
data = DataGen(dataset_path, fr, 640, 480)
left_stride = 160
stop = 5
jnt = []
jnt_im = []
key = None
im_number = 69
current = 60

imagefile = 'rgb_1_'+ str(im_number).zfill(7) +'.jpg'
image_RGB = imageio.imread(os.path.join(pth, imagefile))
last_fn = -1
for depth_frame, color_frame, depth_scale in data.generator():

  depth_image = np.asanyarray(depth_frame.get_data()).astype('float64')
  color_image_from_frame = np.asanyarray(color_frame.get_data()).astype('uint8')
  color_image = color_image_from_frame[:, left_stride:640, :]

  if False:
    if depth_frame.get_frame_number() == i_frames[current]:
      # stop = 0
      
      for joint in batch[current]:
        x = joint[0]
        y = joint[1]
        z = depth_frame.get_distance(int(x)+left_stride, int(y))
        z2 = depth_image[y, x+left_stride] * depth_scale
        print(z , ' == ', z2)
        cv2.circle(color_image, (x, y), 5, (255,0,0), 2)
        jnt.append([x, y, z])
      print('------------------------------', current+1)
      depth_image = depth_image[:,160:]
      savemat('test/depth_1_'+ str(current+1).zfill(7) +'.mat', {'1': depth_image*depth_scale})
      current+=1

  if True:
    # if depth_frame.get_frame_number() == 232:
    #   stop = 0
    for joint in batch[im_number-1]:
      x = joint[0]
      y = joint[1]
      z = depth_frame.get_distance(int(x), int(y))
      z2 = depth_image[x, y] * depth_scale
      # print(z , ' == ', z2)
      cv2.circle(color_image, (x, y), 5, (255,0,0), 2)
      jnt.append([x, y, z])

  cv2.imshow("Color Stream", color_image)
  cv2.imshow("RGB Stream", image_RGB)
  if key == ord('s'):
    print(depth_frame.get_frame_number())
    stop = 0
  if key == ord('c'):
    stop = 15

  key = cv2.waitKey(stop)

  if key == ord('a'):
    print(depth_frame.get_frame_number())
    print(jnt)
  if key == ord('q'):
    cv2.destroyAllWindows()
    break
  jnt = []