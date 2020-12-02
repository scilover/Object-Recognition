import os, cv2
import numpy as np
import tensorflow as tf


image_folder = r'G:\Data\coco2017\val2017'
imgs = os.listdir(image_folder)
shapes = []
for img in imgs:
    img = cv2.imread(os.path.join(image_folder, img))
    shapes.append(img.shape)

shapes = np.array(shapes)
print(max(shapes[:,0]))


