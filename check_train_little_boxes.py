import csv
import matplotlib.pyplot as plt
import cv2
import pickle
import numpy as np
import time

sea_lion_classes = ['adult_male', 'subadult_male', 'adult_female', 'juvenile']
lion_id = 0

with open('/home/ubuntu/Andrew_Projects/FRCNN/keras-frcnn/simple.csv', 'rb') as csvfile:
  reader = csv.reader(csvfile)
  st = time.time()
  current_mini_image = ""
  for row in reader:
    current_mini_image = row[0]
    img_scaled = cv2.imread(current_mini_image)

    x_start = int(row[1])
    x_end = int(row[3])
    y_start = int(row[2])
    y_end = int(row[4])

    # height = y_end - y_start
    # width = x_end - x_start
    # print('height ' + str(height))
    # print('width ' + str(width))

    lion_class_number = int(row[5])
    lion_class = sea_lion_classes[lion_class_number]

    cropped_img = img_scaled[y_start:y_end, x_start:x_end, :]
    partial_file_name = '/home/ubuntu/Andrew_Projects/KaggleSeaLions/input/TrainJustLions/'+lion_class+'/'+str(lion_id)+'.jpg'
    cv2.imwrite(partial_file_name, cropped_img)

    if lion_id % 100 == 0:
      print('completed: '+str(lion_id))
      print('Elapsed time = {}'.format(time.time() - st))

    lion_id += 1

    # cv2.rectangle(img_scaled, (x_start, y_start), (x_end, y_end), (255,0,0), 2)
    #
    # plt.imshow(img_scaled)
    # plt.show()
