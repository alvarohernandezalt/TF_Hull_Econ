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
