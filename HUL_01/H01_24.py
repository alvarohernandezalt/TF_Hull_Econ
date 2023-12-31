import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import os

# Declare image folder and image name
image_name='prior-1-1.jpg'
image_dir='D:/zCODE/04_SD_Training/training/Prior'

# open image to numpy array and switch to RGB from BGR
img = cv2.imread(os.path.join(image_dir,image_name))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Save image data into npy file
np.save('images.npy', img)

# Import image data using numpy
images = np.load('images.npy')

# Convert numpy array into a tensorflow constant
images=tf.constant(images, dtype=tf.float32)
print(tf.shape(images))

# Normalize pixel values to [0,1] interval
images = images / 255.0
print(images.numpy())

# This also can be done with tf.division
# images = tf.division(images, 255.0)

# Import data and convert to tensorflow with pandas
data = np.load('data.csv')
data_tensorflow = tf.constant(data)

# Convert data to numpy array
datanumpy=np.array(data)


