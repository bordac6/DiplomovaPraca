from datagen import DataGen
from keras.optimizers import RMSprop
from keras.losses import mean_squared_error
from keras.models import model_from_json
from heatmap_process import post_process_heatmap
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

from PIL import Image

def load_model(modeljson, modelfile):
    with open(modeljson) as f:
        model = model_from_json(f.read())
    model.load_weights(modelfile)
    return model

def run_eval(model_json, model_weights, epoch, dataset_path):

    model = load_model(model_json, model_weights)
    model.compile(optimizer=RMSprop(lr=5e-4), loss=mean_squared_error, metrics=["accuracy"])

    data = DataGen(dataset_path, 15, 640, 480)
    scale = 4
    scale_origin = 7.5
    left_stride = 80
    index_frame = 0

    for depth_frame, color_frame in data.generator():
      depth_image = np.asanyarray(depth_frame.get_data())
      color_image_from_frame = np.asanyarray(color_frame.get_data()).astype('float64')
      color_image = cv2.resize(color_image_from_frame[:, left_stride:640-80, :], dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
      plot_image = cv2.resize(color_image_from_frame[:, left_stride:640-80, :], dsize=(256, 256), interpolation=cv2.INTER_CUBIC)

      #switch RGB to BGR for cv2
      color_image_BGR = np.zeros_like(color_image)
      color_image_BGR[:,:,0] = color_image[:,:,-1]
      color_image_BGR[:,:,1] = color_image[:,:,1]
      color_image_BGR[:,:,2] = color_image[:,:,0]

      color_image_BGR /= 255.0
      color_image /= 255.0

      color_mean = np.array([0.4486, 0.4269, 0.3987])
      for o in range(color_image.shape[-1]):
          color_image[:, :, o] -= color_mean[o]

      out = model.predict(color_image[np.newaxis, :, :, :])
      pred_kps = post_process_heatmap(out[-1][0])
      pred_kps = np.array(pred_kps)
      in_3d = []
      for j in range(out[-1].shape[-1]):
        color = (0, 0, 255)
        if j == 2 or j == 3:
          color = (255, 0, 0)
        x = int(pred_kps[j,0])
        y = int(pred_kps[j,1])
        z = depth_frame.get_distance(int((x*scale_origin)+left_stride), int(y*scale_origin))
        in_3d.append([x, y, z])
        # cv2.circle(color_image_BGR, (x*scale, y*scale), 5, color, 2)
        # cv2.circle(plot_image, (x*scale, y*scale), 5, color, 2)

      data_3d = np.array(in_3d)
      data_3d[:,:2] = data_3d[:,:2]*scale

      z = data_3d[:,2]
      if np.max(z) - np.median(z) > 30:
        z[np.argmax(z)] = np.mean(z[np.arange(z.shape[0]) != np.argmax(z)])
      data_3d[:,2] = z

      ## Plotting
      fig = plt.figure()
      ax = fig.add_subplot(122, projection='3d')
      ax.plot3D([0],[0],[0])
      ax.plot3D([256],[256],[2])
      for k in range(2, 11, 2):
        ax.plot3D(data_3d[k-2:k,0], data_3d[k-2:k,1], data_3d[k-2:k,2])
        ax.plot3D([data_3d[k-1,0]]+[data_3d[10,0]], [data_3d[k-1,1]]+[data_3d[10,1]], [data_3d[k-1,2]]+[data_3d[10,2]])

        start_point = (int(data_3d[k-2,0]), int(data_3d[k-2,1]))
        end_point = (int(data_3d[k-1,0]), int(data_3d[k-1,1]))
        thickness = 2
        color = (255,0,0) if k == 4 else (0,255,0)
        cv2.line(plot_image, start_point, end_point, color, thickness)
        cv2.line(color_image_BGR, start_point, end_point, color, thickness)

        start_point = (int(data_3d[k-1,0]), int(data_3d[k-1,1]))
        end_point = (int(data_3d[10,0]), int(data_3d[10,1]))
        color = (0,0,255)
        cv2.line(plot_image, start_point, end_point, color, thickness)
        cv2.line(color_image_BGR, start_point, end_point, color, thickness)

      ax.view_init(180, 90)
      ax.set_xlim(ax.get_xlim()[::-1])        # invert the axis
      ax.set_zlim(ax.get_zlim()[::-1])        # invert the axis

      ax.set_xlabel("x [px]")
      ax.set_ylabel("y [px]")
      ax.set_zlabel("z [m]")

      plt.subplot(1,2,1)
      plt.imshow(plot_image/255)
      plt.show()

      # #Render image in opencv window
      # cv2.imshow("Color Stream", color_image_BGR)
      # im = Image.fromarray(np.asanyarray(plot_image).astype('uint8'))
      # im.save("output_gif/rgb_1_"+ str(index_frame).zfill(7) +'.jpg')
      # index_frame += 1
      # key = cv2.waitKey(10)

      # #if pressed escape exit program
      # if key == 27:
      #     cv2.destroyAllWindows()
      #     break


if __name__ == "__main__":
  pth = 'C:\\Users\\TBordac\\Documents\\Workspace\\git\\PNNPPV_project\\trained_models\\my_final\\'
  # dataset_path = "test_bag\\viktor2r.bag"
  dataset_path = "C:\\Users\\TBordac\\Documents\\Workspace\\FMFI\\DiplomovaPracaBackup\\codes\\rosbag_reader\\test_bag\\rebecca2r.bag"
  run_eval(pth+"net_arch.json", pth+"weights_epoch{}.h5".format(219), 219, dataset_path)