import cv2

from plot import showBox
from prediction import Predict
from preprocess import GetData, GetImage
from models import Yolo
import numpy as np
import tensorflow as tf
import os

# img_folder = r'G:\Data\yoloV4自己的数据集\JPEGImages\trainImages'
# label_folder = r'G:\Data\yoloV4自己的数据集\Annotations\trainLabels'
# data = GetData(img_folder, label_folder, batch_size=1, grid_size=(20,20))()
# n = 0
# for img, label in data:
#     n += 1
#     showBox(img=img[0], labels=label[0], name=n)

# test prediction
img_folder = r'G:\Data\yoloV4自己的数据集\JPEGImages\testImages'
label_folder = r'G:\Data\yoloV4自己的数据集\Annotations\testLabels'
#
#
# test_image = GetImage(img_folder)()
# test_image = np.array([img.numpy() for img in test_image])
model = Yolo((480,640), grid_size=(32, 32)).model()
model.load_weights('model_weights\DenseNet121-1024-Grid32-32\/field3-5-7-8-stride1pool/'
                   'beta-0.0067-alpha-1.0-lr-0.0001-epoch-336-testloss-0.00103-trainloss-0.00039-l1-0.00055-l2-0.00000-l3-0.00047-iouLoss-0.0579.h5')
# labels = model.predict(test_image)
# n = 0
# for img, label in zip(test_image, labels):
#     n += 1
#     showBox(img=img, labels=label, name=n)

data = GetData(img_folder, label_folder, (32,32), 32)()
for imgs, labels in data:
    predictions = model.predict(imgs)
    i = 0
    for img, label, prediction in zip(imgs, labels, predictions):
        i += 1
        showBox(img, label, prediction, i)


# for different input figure size
# img_folder = r'G:\Computer Vision\yolo\SimpleYolo_v2\figures\testImages'
# for img_name in os.listdir(img_folder):
#     path = os.path.join(img_folder, img_name)
#     if os.path.isfile(path):
#         img = tf.io.read_file(path)
#         img = tf.image.decode_jpeg(img, channels=3)
#         model = Yolo(img.shape[:-1], grid_size=(32, 32)).model()
#         model.load_weights('G:\Computer Vision\yolo\SimpleYolo_v2\model_weights\DenseNet121-1024-Grid32-32\/'
#                            'beta-0.005-alpha-2-lr-0.0001-epoch-457-testloss-0.00093-trainloss-0.00062-l1-0.00055-l2-0.00000-l3-0.00038.h5')
#         img = img[tf.newaxis, ...]
#         labels = model.predict(img)
#         for img, label in zip(img, labels):
#             showBox(img, label, img_name)



