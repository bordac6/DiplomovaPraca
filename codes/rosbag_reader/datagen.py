import pyrealsense2 as rs
from os import path

class DataGen(object):

  def __init__(self, bag_file_path, frame_rate, width, height):
    self.bag_file_path = bag_file_path
    self.frame_rate = frame_rate
    self.width = width
    self.height = height

    if not path.exists(bag_file_path):
      raise FileExistsError('Can not find file specified at path: ' + bag_file_path)
    if bag_file_path[-4:] != '.bag':
      raise TypeError('File is not a .bag file')

  # Streaming loop
  def generator(self):
    pipeline = rs.pipeline()

    # Create a config object
    config = rs.config()
    # Tell config that we will use a recorded device from filem to be used by the pipeline through playback.
    rs.config.enable_device_from_file(config, self.bag_file_path)

    # Configure the pipeline to stream the depth stream
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, self.frame_rate)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, self.frame_rate)

    # Start streaming from file
    pipeline.start(config)
    while True:
      # Get frameset of depth
      frames = pipeline.wait_for_frames()

      #alignment
      align = rs.align(rs.stream.color)
      aligned_frames = align.process(frames)

      depth_frame = aligned_frames.get_depth_frame()
      color_frame = aligned_frames.get_color_frame()

      yield depth_frame, color_frame