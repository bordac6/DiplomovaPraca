import numpy as np
import matplotlib.pyplot as plt

from scipy.io import savemat, loadmat

import os
from matplotlib import font_manager as fm, rcParams

fpath = os.path.join("%APPDATA%\\Local\\Microsoft\\Windows\\Fonts", "cmu.serif-roman.ttf")
prop = fm.FontProperties(fname=fpath)
plt.rcParams.update({'font.size': 30, 'figure.dpi': 300})

def plot_all_acc(title):
  n_epoch = 51
  fig = plt.figure(figsize=(3, 3))
  # plt.subplot(121)
  plt.plot(np.arange(len(y5[:n_epoch])), np.array(y5[:n_epoch])*100, 'b', np.arange(len(y2[:n_epoch])), np.array(y2[:n_epoch])*100, 'r')
  plt.legend(['tolerancia 9px', 'tolerancia 3.6px'], bbox_to_anchor=(0.01, 1), loc='upper left', borderaxespad=0.1, prop=prop)
  plt.xlabel('Epocha', fontproperties=prop)
  plt.ylabel('Presnosť [%]', fontproperties=prop)
  # plt.title('celková presnosť', fontproperties=prop)
  ax = fig.gca()
  ax.set_xticks(np.arange(0, len(y5[:n_epoch]), 5))
  ax.set_xticklabels(np.arange(0, len(y5[:n_epoch]), 5), fontproperties=prop)
  # for i, label in enumerate(ax.xaxis.get_ticklabels()):
  #   if i%5 == 0:
  #     label.set_visible(True)
  #   else:
  #     label.set_visible(False)
  ax.set_yticks(np.arange(0, 101, 5))
  ax.set_yticklabels(np.arange(0, 101,5), fontproperties=prop)
  for i, label in enumerate(ax.yaxis.get_ticklabels()):
    if i%2 == 0:
      label.set_visible(True)
    else:
      label.set_visible(False)
  plt.tight_layout()
  plt.grid()

  fig = plt.figure(figsize=(3,3))
  # plt.subplot(121)
  plt.plot(np.array(joints_acc[:n_epoch]))
  print('e{}: {}'.format(np.argmax(joints_acc, axis=0), np.max(joints_acc)))
  print(np.argmax(np.array(joints_acc)[np.argmax(joints_acc, axis=0)]))
  
  # plt.legend(['k_palec', 'palec', 'k_ukazovák', 'ukazovák', 'k_prostredník', 'prostredník', 'k_prsteník', 'prsteník', 'k_malíček', 'malíček', 'zápästie'], 
  #   bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., prop=prop)
  
  plt.xlabel('Epocha', fontproperties=prop)
  plt.ylabel('Presnosť [%]', fontproperties=prop)
  # plt.title('presnosť pre jednotlivé kĺby', fontproperties=prop)

  joints = np.array(joints_acc)
  # ft = joints[:,[0,2,4,6,8]]
  ft = joints[:,:]
  mn = np.mean(ft, axis=1)
  # print('means: {}'.format(mn))
  # print('In Epoch: 49 was reachd acc: ', y5[48], ' with tolerance 0.5px. ', 'Acc with tolerance 0.2px: ', y2[48])
  # print('In Epoch: ', np.argmax(y5), ' was reachd maximum: ', np.max(y5), ' with tolerance 0.5px. ', 'Acc with tolerance 0.2px: ', y2[np.argmax(y5)])
  # print('In Epoch: ', np.argmax(y2), ' was reachd maximum: ', np.max(y2), ' with tolerance 0.2px.', 'Acc with tolerance 0.5px: ', y5[np.argmax(y2)])
  # name_joints = ['k_palec', 'palec', 'k_ukazovak', 'ukazovak', 'k_prostrednik', 'prostrednik', 'k_prstenik', 'prstenik', 'k_malicek', 'mallicek', 'zapastie']
  # for i in range(joints.shape[1]):
  #   print('In Epoch: 49 has ', name_joints[i], ' acc: ', joints[49, i], ' with tolerance 0.5px.')
  # print('===============================')
  # for i in range(joints.shape[1]):
  #   print('In Epoch: ', np.argmax(y5), ' has ', name_joints[i], ' acc: ', joints[np.argmax(y5), i], ' with tolerance 0.5px.')
  # print('===============================')
  # for i in range(joints.shape[1]):
  #   print('In Epoch: ', np.argmax(y2), ' has ', name_joints[i], ' acc: ', joints[np.argmax(y2), i], ' with tolerance 0.5px.')
  # print('===============================')

  # plt.suptitle(title, fontproperties=prop)
  ax = fig.gca()
  ax.set_xticks(np.arange(0, len(y5[:n_epoch]), 5))
  ax.set_xticklabels(np.arange(0, len(y5[:n_epoch]), 5), fontproperties=prop)
  # for i, label in enumerate(ax.xaxis.get_ticklabels()):
    # if i%5 == 0:
    #   label.set_visible(True)
    # else:
    #   label.set_visible(False)
  ax.set_yticks(np.arange(0, 101, 5))
  ax.set_yticklabels(np.arange(0, 101, 5), fontproperties=prop)
  for i, label in enumerate(ax.yaxis.get_ticklabels()):
    if i%2 == 0:
      label.set_visible(True)
    else:
      label.set_visible(False)
  plt.grid()
  plt.tight_layout()
  plt.show()

epochs_vals = []
open_path = 'val_nyu_32im_11hm_128ch_all'
with open(open_path+'.txt', 'r') as file:
  line = file.readline()
  while line:
    if line.startswith('Epoch'):
      epochs_vals.append(line.strip())
    else:
      epochs_vals[-1] += ' ' + line.strip()
    line = file.readline()

y5 = []
y2 = []
joints_acc = []
out = []

for i, epoch in enumerate(epochs_vals):
  data = epoch.split(':')
  joint_arr = data[-1][1:-1].split(' ')
  joint_arr = [float(joint) for joint in joint_arr if joint]
  acc_5 = float(data[-2])
  acc_2 = float(data[1])
  epoch_idx = data[0].split(' ')[-1]

  if epoch_idx == '0':
    if len(y5) > 1:
      plot_all_acc('Presnosť s učením končeka palca')
    y5 = []
    y2 = []
    joints_acc = []

  y5.append(acc_5)
  y2.append(acc_2)
  joints_acc.append(np.array(joint_arr)*100)
  tmp = [acc_2, acc_5]
  for i in joints_acc[-1]:
    tmp.append(i)
  out.append(np.array(tmp))
  # if epoch_idx == '50':
  #   break
# plot_all_acc('Presnosť s učením všetkých kĺbov')
plot_all_acc('Presnosť s učením všetkých kĺbov na vlastnom datasete')
# plot_all_acc('Presnosť s učením všetkých kĺbov bez rozdelenia na fázy')

# savemat(open_path+'.mat', {'item': np.array(out)})
