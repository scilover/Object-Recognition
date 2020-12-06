# from tensorflow.python.keras.losses import LossFunctionWrapper
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from prediction import Predict


class NonEquivBinaryCrossEntropy:
    def __init__(self, beta=0.01):

        self.beta = beta

    def __call__(self, y_true, y_pred):

        y_true = tf.clip_by_value(y_true, K.epsilon(), 1.- K.epsilon())
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.- K.epsilon())

        l1 = y_true * tf.math.log(y_pred)
        l1 += (1 - y_true) * tf.math.log(1 - y_pred) * self.beta
        l1 = - tf.reduce_mean(l1)

        # compFactor = 2 / (1 + self.beta)
        # compFactor = 300 / (2 + 298 * self.beta)

        return l1 #* compFactor


def yolo_loss_v1(y_true, y_pred, beta=0.95, alpha=0.5):

    mask = tf.where(y_true[...,1:] > 0, 1., 0.)
    mask = tf.concat([tf.ones([*y_true.shape[:-1],1]), mask], axis = -1)
    y_pred = tf.math.multiply(mask, y_pred)
    y_true = tf.dtypes.cast(y_true, y_pred.dtype)

    l1 = tf.keras.losses.BinaryCrossentropy()(y_true[..., 0:1], y_pred[..., 0:1])
    l2 = tf.keras.losses.BinaryCrossentropy()(y_true[...,1:3], y_pred[..., 1:3])
    l3 = tf.keras.losses.MeanSquaredError()(y_true[..., 3:], y_pred[..., 3:])
    loss = tf.add_n([l1, l2, l3])

    return loss, l1, l2, l3


def yolo_loss_v2(y_true, y_pred, beta=0.01, alpha=0.5):

    mask = np.zeros(y_true.shape)
    mask[...,0] = 1.
    index = np.where(y_true[...,0]==1)
    for i,j,k in zip(index[0], index[1], index[2]):
        mask[i,j,k] = 1.
    mask = tf.convert_to_tensor(mask, dtype=tf.float32)
    y_pred = tf.math.multiply(mask, y_pred)
    y_true = tf.dtypes.cast(y_true, y_pred.dtype)

    l1 = alpha * NonEquivBinaryCrossEntropy(beta)(y_true[...,0:1], y_pred[...,0:1])
    l2 = tf.keras.losses.BinaryCrossentropy()(y_true[...,1:3], y_pred[..., 1:3])
    l3 = tf.keras.losses.MeanSquaredError()(y_true[..., 3:], y_pred[..., 3:])
    loss = tf.add_n([l1, l2, l3])

    return loss, l1, l2, l3


def yolo_loss_v3(y_true, y_pred, exist_thresh, iou_thresh, grid_size, beta, alpha):

    y_true = tf.cast(y_true, y_pred.dtype)

    mask = np.zeros(y_true.shape)
    mask[...,0] = 1.
    index = tf.where(y_true[...,0]==1)
    iouLossList = [0]
    # for i,j,k in index.numpy():
    #     mask[i,j,k] = 1.
    #     iouLoss = 1.- Predict(exist_thresh, iou_thresh, grid_size).getIOUinTraining(j, k, y_true[i,j,k], y_pred[i,j,k])
    #     iouLossList.append(iouLoss)

    mask = tf.convert_to_tensor(mask, dtype=tf.float32)
    y_pred = tf.math.multiply(mask, y_pred)


    l1 = alpha * NonEquivBinaryCrossEntropy(beta)(y_true[...,0:1], y_pred[...,0:1])
    l2 = tf.keras.losses.CategoricalCrossentropy()(y_true[...,1:81], y_pred[..., 1:81])
    l3 = tf.keras.losses.MeanSquaredError()(y_true[..., -4:], y_pred[..., -4:])
    iouLoss = tf.reduce_mean(iouLossList)
    loss = tf.add_n([l1, l2, l3])

    return loss, l1, l2, l3, iouLoss
