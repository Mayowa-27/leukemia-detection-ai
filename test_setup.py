import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
import cv2
from PIL import Image

print("TensorFlow version:", tf.__version__)
print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)
print("Matplotlib version:", matplotlib.__version__)
print("OpenCV version:", cv2.__version__)

gpus = tf.config.list_physical_devices('GPU')
print("GPU Available:", len(gpus) > 0)

print("All libraries imported successfully!")
