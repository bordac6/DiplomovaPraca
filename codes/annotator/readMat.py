from scipy.io import loadmat, savemat
import numpy as np
import imageio
import os
import cv2
from PIL import Image

pth = "rosbag_reader\\test"
mat_pth = "rosbag_reader\\test\\joint_test_data_with_depth.mat"
annot_data = loadmat(mat_pth)
joints = annot_data['joint_uvd']
print(joints.shape)
# print(joints)

# jnt = []
# batch = []
# annot = []

# data = annot_data['joint_uvd']
# if data.shape[0] == 1 and data.shape[1] > 0:
#     for i in range(data.shape[1]):
#       if i < 68:
#         batch.append(data[0,i,:,:])

for sample_index in range(69, joints.shape[1]):
  print(sample_index)
  kpanno = joints[0, sample_index, :, :]
  imagefile = 'rgb_1_'+ str(sample_index+1).zfill(7) +'.jpg'
  image_RGB = imageio.imread(os.path.join(pth, imagefile))

  image = np.zeros_like(image_RGB)
  image[:,:,0] = image_RGB[:,:,-1]
  image[:,:,1] = image_RGB[:,:,1]
  image[:,:,2] = image_RGB[:,:,0]
  # if sample_index > 67:
  #   orig_image = cv2.resize(image_RGB, dsize=(480, 480), interpolation=cv2.INTER_CUBIC)
  #   im = Image.fromarray(np.asanyarray(orig_image).astype('uint8'))
  #   im.save("bigger/rgb_1_"+ str(sample_index+1).zfill(7) +'.jpg')

  for i in range(kpanno.shape[0]):
    x = kpanno[i, 0]
    y = kpanno[i, 1]
    image = cv2.circle(image, (int(x), int(y)), 5, (0,0,255), 2)
    # if sample_index > 67:
    #   jnt.append(np.array([int((x*1.875)), int(y*1.875), 1]))

  # if sample_index > 67:
  #   batch.append(np.array(jnt))
  #   jnt = []
  cv2.imshow('Annotaded image', image)
  cv2.waitKey(0)
# annot = [np.array(batch)]
# savemat('joint_data_test.mat', {'joint_uvd': np.array(annot)})