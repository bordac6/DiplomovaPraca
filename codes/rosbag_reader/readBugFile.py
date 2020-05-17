import cv2
import numpy as np
import pyrealsense2 as rs
import os.path
from os import path
from PIL import Image

my = 0
pth = "train.bag" # recorded file from RS d435i 680x480
fr = 15 # frame rate used while recording

# print(path.exists(path_ext))

pipeline = rs.pipeline()

# Create a config object
config = rs.config()
# Tell config that we will use a recorded device from filem to be used by the pipeline through playback.
rs.config.enable_device_from_file(config, pth)
# Configure the pipeline to stream the depth stream
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fr)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, fr)

# Start streaming from file
pipeline.start(config)

# Create colorizer object
colorizer = rs.colorizer()

old = np.zeros(shape=(480, 480, 3))
depth_frames = [None]*126
color_frames = [None]*126
i = 0
# Streaming loop
while True:
    # Get frameset of depth
    frames = pipeline.wait_for_frames()

    #alignment
    align = rs.align(rs.stream.color)
    aligned_frames = align.process(frames)

    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if i > 125:
        pipeline.stop()
        print(i)
        break
    color_frames[i] = color_frame
    depth_frames[i] = depth_frame
    i += 1
    # depth_frame.get_distance(0,0)

# Create opencv window to render image in
cv2.namedWindow("Depth Stream", cv2.WINDOW_AUTOSIZE)
for j in range(len(color_frames)):
    depth_frame = depth_frames[j]
    color_frame = color_frames[j]

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data()).astype('uint8')

    # Get depth frame
    # depth_frame = frames.get_depth_frame()
    # color_frame = frames.get_color_frame()

    # Colorize depth frame to jet colormap
    # depth_color_frame = colorizer.colorize(frames.get_depth_frame())
    depth_color_frame = colorizer.colorize(depth_frame)

    # Convert depth_frame to numpy array to render image in opencv
    depth_color_image = np.asanyarray(depth_color_frame.get_data())

    #switch RGB to BGR for cv2
    color_image_RGB = np.zeros_like(color_image)
    color_image_RGB[:,:,0] = color_image[:,:,-1]
    color_image_RGB[:,:,1] = color_image[:,:,1]
    color_image_RGB[:,:,2] = color_image[:,:,0]

    # Render image in opencv window
    cv2.imshow("Depth Stream", depth_color_image)
    cv2.imshow("Coloer Stream", color_image_RGB)
    key = cv2.waitKey(100)

    # if pressed escape exit program
    if key == 27:
        imm = cv2.resize(color_image[:480, :480, :], dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
        im = Image.fromarray(imm)
        im.save("input_to_CNN_256.jpg")
        cv2.destroyAllWindows()
        break

