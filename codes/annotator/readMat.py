from scipy.io import loadmat
import numpy as np
import imageio
import os
import cv2

pth = "C:\\Users\\TBordac\\Downloads\\my_dataset\\"
annot_data = loadmat('joint_data.mat')
joints = annot_data['joint_uvd']
print(joints.shape)
# print(joints)

for sample_index in range(joints.shape[1]):
  kpanno = joints[0, sample_index, :, :]
  imagefile = 'rgb_1_'+ str(sample_index+1).zfill(7) +'.jpg'
  image_RGB = imageio.imread(os.path.join(pth, imagefile))

  image = np.zeros_like(image_RGB)
  image[:,:,0] = image_RGB[:,:,-1]
  image[:,:,1] = image_RGB[:,:,1]
  image[:,:,2] = image_RGB[:,:,0]

  for i in range(kpanno.shape[0]):
    x = kpanno[i, 0]
    y = kpanno[i, 1]
    image = cv2.circle(image, (int(x), int(y)), 5, (0,0,255), 2)
  cv2.imshow('Annotaded image', image)
  cv2.waitKey(0)