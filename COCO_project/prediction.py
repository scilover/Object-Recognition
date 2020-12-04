import numpy as np
import tensorflow as tf
import math

class Predict:

    def __init__(self, exist_thresh=0.9, category_thresh=0.5, iou_thresh=0.6, grid_size=(32,32)):

        self.exist_thresh = exist_thresh
        self.category_thresh = category_thresh
        self.iou_thresh = iou_thresh
        self.grid_size = grid_size
        # self.ref_h = 50
        # self.ref_w = 50

    def getIndex(self, label):
        """
        label: 3-d output tensor (32,32,7)
        return :a sorted index list according to its confidence--- label[...,0]
        """
        indexes = tf.where(label[...,0] > self.exist_thresh)
        indexes = indexes.numpy().tolist()
        indexes_ = []
        for index in indexes:
            index1, index2 = index
            indexes_.append((index, label[index1, index2, 0]))
        if indexes_:  # if any target exists
            indexes_.sort(key=lambda x:x[1], reverse=False) # sort indexes according to its confidence
        indexes = [x[0] for x in indexes_]

        return indexes

    def getLocation(self, label):
        """
        transform label to locations

        :param label: the predicted label
        :return: the category c, the mid-point(x, y) and (h, w) in pixels.
        """
        locations = []
        indexes = self.getIndex(label) # get sorted indexes
        label = np.array(label)
        dh, dw = self.grid_size
        for index in indexes:
            c, x, y, h, w = label[index[0], index[1]][2:]
            x = int(dw * (index[1] + x))
            y = int(dh * (index[0] + y))
            h, w =  h * dh, w * dw
            # h = self.ref_h * math.exp(h)  # v2
            # w = self.ref_w * math.exp(h)  # v2
            locations.append([c, x, y, h, w])

        return locations
    # @classmethod
    def getIOU(self, box1, box2):
        """
        :param: box1, box2 should be two vectors(5-d) to be compared.
        :return: the intersection over union (IOU) result for the two vectors
        """
        c_, x_, y_, h_, w_ = box1 # predicted target vector
        c, x, y, h, w = box2

        if (c - 0.5) * (c_ - 0.5) > 0 and h > 0 and h_ > 0 and w_ > 0 and w > 0:
            intersection = max(0, (0.5 * (h + h_) - np.abs(y - y_))) * max(0, (0.5 * (w + w_) - np.abs(x - x_)))
            union = h_ * w_ + h * w - intersection
            iou = intersection / union
        else:
            iou = 0.
        return  iou


    def getBoxinTraining(self, index1, index2, label):

        c, x, y, h, w = label[2:]
        dh, dw = self.grid_size

        return [c, dw * (index2 + x), dh * (index1 + y), h * dh, w * dw] # index order is critical!!!

    def getIOUinTraining(self, index1, index2, label1, label2):

        return self.getIOU(self.getBoxinTraining(index1, index2, label1), self.getBoxinTraining(index1, index2, label2))

    def getBox(self, label):
        """
        :param label: 3-d label (32,32,7)
        :return: box param (c, x, y, h, w) in pixels
        """
        locations = self.getLocation(label)
        boxes = []
        while locations:
            location = locations.pop()
            overlap = False
            for box in boxes:
                iou = self.getIOU(location, box)
                if iou > self.iou_thresh:
                    overlap = True
                    break
            if not overlap:
                boxes.append(location)

        return boxes









