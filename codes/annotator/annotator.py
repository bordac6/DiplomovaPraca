import cv2
import numpy as np
import pyrealsense2 as rs
import os.path
from os import path
from PIL import Image

from scipy.io import savemat, loadmat

my = 0
# path_ext = "C:\\Users\\TBordac\\Documents\\Workspace\\git\\rosbag_reader\\igor4r.bag"
train = "C:\\Users\\TBordac\\Documents\\Workspace\\git\\rosbag_reader\\igor2r.bag"
path_ext = "C:\\Users\\TBordac\\Documents\\Workspace\\FMFI\\DiplomovaPracaBackup\\codes\\rosbag_reader\\test_bag\\igor2l.bag"
# path_ext = "C:\\Users\\TBordac\\Documents\\Workspace\\git\\rosbag_reader\\viktor2l.bag"
# path_ext = "C:\\Users\\TBordac\\Documents\\Workspace\\git\\rosbag_reader\\viktor4r.bag"
# path_ext = "C:\\Users\\TBordac\\Documents\\Workspace\\git\\rosbag_reader\\rebecca2l.bag"
path_my = "C:\\Users\\TBordac\\Documents\\20191205_141123.bag"
sample_idx = 0

#annotatinos
# annot = [
#     [ # which camera
#         [ # which sample
#             [ # which keypoint
#                 [ # u,v,d
#                 ]
#             ]
#         ]
#     ]
# ]

jnt = []
batch = []
annot = []

if path.exists('test_joint_data.mat'):
    annot_data = loadmat('test_joint_data.mat')
    data = annot_data['joint_uvd']
    if data.shape[0] == 1 and data.shape[1] > 0:
        for i in range(data.shape[1]):
            batch.append(data[0,i,:,:])
        # annot = annot_data['joint_uvd']
print(np.array(batch).shape)

pth = path_my if my == 1 else path_ext
fr = 30 if my == 1 else 15
pipeline = rs.pipeline()
# Create a config object
config = rs.config()
# Tell config that we will use a recorded device from filem to be used by the pipeline through playback.
rs.config.enable_device_from_file(config, pth)
# Configure the pipeline to stream the depth stream
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, fr)
# Start streaming from file
pipeline.start(config)

# Create drawable window
def draw_circle(event,x,y,flags,param):
    global mouseX, mouseY, color_image_BGR
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(color_image_BGR,(x,y),5,(0,0,255),-1)
        cv2.imshow("Color Stream", color_image_BGR)
        mouseX,mouseY = x,y
        jnt.append(np.array([x, y, 1]))

i = 0
old = np.zeros(shape=(480, 480, 3))
video = np.zeros(shape=(600, 480, 480, 3))
# Streaming loop
while True:
    # Get frameset of depth
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data()).astype('uint8')
    color_image = color_image[:, 160:, :] #cv2.resize(color_image[:480, :480, :], dsize=(256, 256), interpolation=cv2.INTER_CUBIC)

    if i == 0:
        old = color_image
    elif np.sum(old - color_image) == 0:
        pipeline.stop()
        print(i)
        break
    video[i] = color_image
    i += 1

# Create opencv window to render image in
cv2.namedWindow("Color Stream", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("Color Stream",draw_circle)
for i in range(0, video.shape[0], 5):

    image = np.asanyarray(video[i]).astype('uint8')
    #switch RGB to BGR for cv2
    color_image_BGR = np.zeros_like(image)
    color_image_BGR[:,:,0] = image[:,:,-1]
    color_image_BGR[:,:,1] = image[:,:,1]
    color_image_BGR[:,:,2] = image[:,:,0]

    # Render image in opencv window
    cv2.imshow("Color Stream", color_image_BGR)
    key = cv2.waitKey(0)

    # if pressed escape exit program
    if key == 27:
        annot = [np.array(batch)]
        savemat('test_joint_data.mat', {'joint_uvd': np.array(annot)})
        cv2.destroyAllWindows()
        break
    elif key == ord('q'):
        cv2.destroyAllWindows()
        break
    elif key == ord('a'):
        verify_batch = batch[:]
        verify_batch.append(np.array(jnt))
        if len(np.array(verify_batch).shape) == 3:
            batch.append(np.array(jnt))
        else:
            print('Couldn`t add new annotations. Shape cann`t be {}'.format(np.array(verify_batch).shape))
        im = Image.fromarray(image)
        im.save("test/rgb_1_"+ str(len(batch)).zfill(7) +'.jpg')
        jnt = []
