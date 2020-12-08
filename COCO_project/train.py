from tensorflow.python.data.ops.dataset_ops import AUTOTUNE
import time
from models import Yolo
from collections import defaultdict
import tensorflow as tf
from preprocess import GetData, GetImage
from sklearn.preprocessing import OneHotEncoder
from pycocotools.coco import COCO

# import numpy as np
# from tensorflow.keras import backend as K
# from tensorflow.keras.layers import Lambda
import json, losses


def augment(image,label):
   # Random crop back to the original size
    image = tf.image.random_brightness(image, max_delta=0.3) # Random brightness
    image = tf.image.random_contrast(image, 0.2, 1.)
    return image, label

class Train(object):

    def __init__(self, model, epochs, train_data, test_data, beta, alpha, lr,
                 optimizer, loss, exist_thresh=0.9, iou_thresh=0.6, grid_size=(32, 32)):
        self.train_data = train_data
        self.test_data = test_data
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = epochs
        self.beta = beta
        self.alpha = alpha
        self.lr = lr
        self.exist_thresh = exist_thresh
        self.iou_thresh = iou_thresh
        self.grid_size = grid_size

        self.history = defaultdict(list)
        self.best_loss = 10.
        self.best_l1 = 10.
        self.best_l2 = 10.
        self.best_l3 = 10.
        self.best_iouLoss = 10.
        self.checkpoint = 0

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.l1 = tf.keras.metrics.Mean(name='l1')
        self.l2 = tf.keras.metrics.Mean(name='l2')
        self.l3 = tf.keras.metrics.Mean(name='l3')
        self.iouLoss = tf.keras.metrics.Mean(name='iouLoss')

        # squared_difference = tf.square(y_true - y_pred)
        # return tf.reduce_mean(squared_difference, axis=-1)  # Note the `axis=-1`

    # @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(images)
            loss, l1, l2, l3, iouLoss = self.loss(labels, predictions, self.exist_thresh, self.iou_thresh,
                                                   self.grid_size, self.beta, self.alpha)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            self.train_loss(loss)
            # train_accuracy(labels, predictions)

    # @tf.function
    def test_step(self, images, labels):
        predictions = self.model(images)
        loss, l1, l2, l3, iouLoss = self.loss(labels, predictions, self.exist_thresh, self.iou_thresh,
                                                   self.grid_size, self.beta, self.alpha)

        self.test_loss(loss)
        self.l1(l1)
        self.l2(l2)
        self.l3(l3)
        self.iouLoss(iouLoss)
        # self.test_accuracy(labels, predictions)


    def train(self):

        for epoch in range(self.epochs):
            # 在下一个epoch开始时，重置评估指标
            self.train_loss.reset_states()
            # # train_accuracy.reset_states()
            self.test_loss.reset_states()
            # self.test_accuracy.reset_states()
            self.l1.reset_states()
            self.l2.reset_states()
            self.l3.reset_states()
            self.iouLoss.reset_states()
            for images, labels in self.train_data:
                self.train_step(images, labels)

            for images, labels in self.test_data:
                self.test_step(images, labels)

            test_loss_val = float(self.test_loss.result().numpy())
            train_loss_val = float(self.train_loss.result().numpy())
            l1 = float(self.l1.result().numpy())
            l2 = float(self.l2.result().numpy())
            l3 = float(self.l3.result().numpy())
            iouLoss = float(self.iouLoss.result().numpy())
            self.history['train_loss'].append(train_loss_val)
            self.history['test_loss'].append(test_loss_val)
            self.history['l1'].append(l1)
            self.history['l2'].append(l2)
            self.history['l3'].append(l3)
            self.history['iouLoss'].append(iouLoss)


            if l1 < self.best_l1 or l3 < self.best_l3 or test_loss_val < self.best_loss or iouLoss < self.best_iouLoss:
                self.checkpoint = 0
                if iouLoss < 0.08:#l1 < 0.001 and l3 < 0.001:
                    model.save_weights('model_weights\DenseNet121-1024-Grid32-32'
                                       '\/beta-{}-alpha-{}-lr-{}-'
                                       'epoch-{}-testloss-{:.5f}-trainloss-{:.5f}-l1-{:.5f}-l2-{:.5f}-l3-{:.5f}-iouLoss-{:.4f}.h5'
                                       .format(self.beta, self.alpha, self.lr, epoch, test_loss_val, train_loss_val, l1, l2, l3, iouLoss))
                if l1 < self.best_l1:
                    self.best_l1 = l1
                if l2 < self.best_l2:
                    self.best_l2 = l2
                if l3 < self.best_l3:
                    self.best_l3 = l3
                if iouLoss < self.best_iouLoss:
                    self.best_iouLoss = iouLoss
                if test_loss_val < self.best_loss:
                    self.best_loss = test_loss_val
            else:
                self.checkpoint += 1
                if self.checkpoint > 100:
                    break

            template = 'Epoch: {}, Train loss: {}, Test Loss: {}, l1: {}, l2: {}, l3: {}, iouLoss: {}'#, Test Accuracy: {}'
            print(template.format(epoch + 1,
                                  self.train_loss.result(),
                                  self.test_loss.result(),
                                  self.l1.result(),
                                  self.l2.result(),
                                  self.l3.result(),
                                  self.iouLoss.result())
                  )
                                  # self.test_accuracy.result() * 100))
        time.sleep(20)
#advanced_activations.LeakyReLU(alpha=0.3)

if __name__ == '__main__':

    label_folder = r'G:\Data\coco2017\annotations_trainval2017'
    train_image_folder = r'G:\Data\coco2017\train2017'
    val_image_folder = r'G:\Data\coco2017\val2017'
    img_size = (320, 320)
    grid_size = (16, 16)
    batch_size = 16

    encoder = OneHotEncoder(categories='auto')
    annFile = '{}/annotations/instances_{}.json'.format(label_folder, 'val2017')
    coco = COCO(annFile)
    cats = coco.loadCats(coco.getCatIds())
    cats = [[cat['id']] for cat in cats]
    encoder.fit(cats)

    train_data = GetData(train_image_folder, label_folder, 'train2017', img_size, grid_size, batch_size, encoder)()
    val_data = GetData(val_image_folder, label_folder, 'val2017', img_size, grid_size, batch_size, encoder)()
    # train_data = (train_data
    #             .shuffle(1000)
    #             .map(augment, num_parallel_calls=AUTOTUNE)
    #             # .batch(batch_size=32)
    #             # .prefetch(AUTOTUNE)
    #               )


    betaRange = [0.01]
    lrRange = [1e-4]
    alphaRange = [100.0]
    for beta in betaRange:
        for lr in lrRange:
            for alpha in alphaRange:
                model = Yolo(img_size, grid_size).model()
                # model.load_weights('model_weights\DenseNet121-1024-Grid32-32\/'
                #                    'beta-0.0067-alpha-1.0-lr-0.0001-epoch-405-testloss-0.00058-trainloss-0.00034-l1-0.00018-l2-0.00000-l3-0.00039-iouLoss-0.0604.h5')
                hist = Train(model, 1000, train_data, val_data, beta, alpha, lr,
                             tf.keras.optimizers.Adam(lr),
                             losses.yolo_loss_v4)
                hist.train()
                with open('history/DenseNet121-1024-Grid32-32/history-beta-{}-alpha-{}-lr-{}.json'.format(beta,alpha,lr),'w') as obj:
                    json.dump(hist.history, obj)

    # model.save_weights('G:\Computer Vision\yolo\MyYolo\model_weights\weights.h5')