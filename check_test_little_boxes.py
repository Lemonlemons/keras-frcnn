import csv
import matplotlib.pyplot as plt
import cv2
import pickle
import numpy as np
import time

def format_img(img, C):
	img_min_side = float(C.im_size)
	(height,width,_) = img.shape

	if width <= height:
		f = img_min_side/width
		new_height = int(f * height)
		new_width = int(img_min_side)
	else:
		f = img_min_side/height
		new_width = int(f * width)
		new_height = int(img_min_side)
	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	img = img[:, :, (2, 1, 0)]
	img = img.astype(np.float32)
	img[:, :, 0] -= C.img_channel_mean[0]
	img[:, :, 1] -= C.img_channel_mean[1]
	img[:, :, 2] -= C.img_channel_mean[2]
	img /= C.img_scaling_factor
	img = np.transpose(img, (2, 0, 1))
	img = np.expand_dims(img, axis=0)
	return img

with open('config.pickle', 'rb') as f_in:
	C = pickle.load(f_in)

# turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

with open('/home/ubuntu/Andrew_Projects/KaggleSeaLions/LionLocations.csv', 'rb') as csvfile:
  reader = csv.reader(csvfile)
  st = time.time()
  current_mini_image = ""
  for row in reader:
    if row[0] != current_mini_image:
      current_mini_image = row[0]
      img = cv2.imread('/home/ubuntu/Andrew_Projects/KaggleSeaLions/input/Test'+str(row[6])+'Mini/'+row[0])
      X = format_img(img, C)

      img_scaled = np.transpose(X.copy()[0, (2, 1, 0), :, :], (1, 2, 0)).copy()
      img_scaled[:, :, 0] += 123.68
      img_scaled[:, :, 1] += 116.779
      img_scaled[:, :, 2] += 103.939
      img_scaled = img_scaled.astype(np.uint8)

    x_start = int(row[1])
    x_end = int(row[3])
    y_start = int(row[2])
    y_end = int(row[4])
    base_image_id = int(row[5])
    lion_id = int(row[7])

    # height = y_end - y_start
    # width = x_end - x_start
    # print('height ' + str(height))
    # print('width ' + str(width))

    cropped_img = img_scaled[y_start:y_end, x_start:x_end, :]
    partial_file_name = '/home/ubuntu/Andrew_Projects/KaggleSeaLions/input/JustLions/'+str(base_image_id)+'_'+str(lion_id)+'.jpg'
    cv2.imwrite(partial_file_name, cropped_img)

    if lion_id % 100 == 0:
      print('completed: '+str(lion_id))
      print('Elapsed time = {}'.format(time.time() - st))

    # cv2.rectangle(img_scaled, (x_start, y_start), (x_end, y_end), (255,0,0), 2)
    #
    # plt.imshow(img_scaled)
    # plt.show()
