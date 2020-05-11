import numpy as np
import matplotlib.pyplot as plt

def plot_all_acc(title):
  n_epoch = 221
  fig = plt.figure()
  plt.subplot(121)
  plt.plot(np.arange(len(y5[:n_epoch])), np.array(y5[:n_epoch])*100, 'b', np.arange(len(y2[:n_epoch])), np.array(y2[:n_epoch])*100, 'r')
  plt.legend(['vzdialenosť 5mm', 'vzdialenosť 2mm'])
  plt.xlabel('Epocha')
  plt.ylabel('Presnosť [%]')
  plt.title('celková presnosť')
  ax = fig.gca()
  ax.set_xticks(np.arange(0, len(y5[:n_epoch]), 10))
  ax.set_yticks(np.arange(0, 101, 5))
  plt.grid()

  plt.subplot(122)
  plt.plot(np.array(joints_acc[:n_epoch]))
  print('e{}: {}'.format(np.argmax(joints_acc, axis=0), np.max(joints_acc)))
  print(np.argmax(np.array(joints_acc)[np.argmax(joints_acc, axis=0)]))
  plt.legend(['k_palec', 'palec', 'k_ukazovák', 'ukazovák', 'k_prostredník', 'prostredník', 'k_prsteník', 'prsteník', 'k_malíček', 'malíček', 'zápästie'], 
    bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
  plt.xlabel('Epocha')
  plt.ylabel('Presnosť [%]')
  plt.title('presnosť pre jednotlivé kĺby')

  joints = np.array(joints_acc)
  # ft = joints[:,[0,2,4,6,8]]
  ft = joints[:,:]
  mn = np.mean(ft, axis=1)
  # print('means: {}'.format(mn))
  print('Mean: ', np.max(mn), ' > ', np.max(y2))
  print('In Epoch: ', np.argmax(y2), ' ? ', y5[np.argmax(y2)])
  print('Max: ', np.max(mn), ' == ', np.max(y5))
  print('In Epoch: ', np.argmax(mn), ' == ', np.argmax(y5))
  print('Number of epochs: ', len(mn))

  plt.suptitle(title)
  ax = fig.gca()
  ax.set_xticks(np.arange(0, len(y5[:n_epoch]), 10))
  ax.set_yticks(np.arange(0, 101, 5))
  plt.grid()
  plt.show()

epochs_vals = []
with open('my_val.txt', 'r') as file:
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
  # if epoch_idx == '50':
  #   break
# plot_all_acc('Presnosť s učením všetkých kĺbov')
plot_all_acc('Presnosť s učením všetkých klbov na vlasnom datasete')
