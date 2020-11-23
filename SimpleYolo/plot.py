import json,os
import matplotlib.pyplot as plt
import cv2
from prediction import Predict
import numpy as np
import tensorflow as tf

def showLoss(path, img_name):

    with open(path) as obj:
        hist = json.load(obj)

    plt.plot(hist['train_loss'], label='train_loss')
    plt.plot(hist['test_loss'], label='test_loss')
    plt.plot(hist['l1'], label='l1')
    plt.plot(hist['iouLoss'], label='iouLoss')
    # plt.plot(hist['l2'], label='l2')
    # plt.plot(hist['l3'], label='l3')
    plt.legend()

    plt.savefig('figures/{}.png'.format(img_name))
    plt.clf()

    plt.plot(hist['l2'], label='l2')
    plt.plot(hist['l3'], label='l3')
    plt.legend()
    plt.savefig('figures/{}-l2l3.png'.format(img_name))
    plt.clf()
    # plt.show()

def showBox(img, label, prediction, name):
    """
    img_tensor: 3-d img_tensor or array
    labels: predicted labels for single image (img_tensor)
    """
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    label = np.array(label)
    prediction = np.array(prediction)
    index = np.where(label[...,0]==1)
    iouList, boxList = [], []
    predict = Predict()
    for index1, index2 in zip(index[0], index[1]):
        iouList.append(predict.getIOUinTraining(index1, index2, prediction[index1, index2], label[index1, index2]))
        boxList.append(predict.getBoxinTraining(index1, index2, prediction[index1, index2]))

    # box = Predict(exist_thresh=0.8, iou_thresh=0.005, grid_size=(32,32)).getBox(label)
    for box, iou in zip(boxList, iouList):
        if box[0] < 0.5:
            cls = 'pingpong'
        else:
            cls = 'camera'
        _, x, y, h, w = box
        cv2.rectangle(img, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (60, 20, 220), 2)# bgr
        cv2.putText(img, cls+' '+str(round(iou,2)), (int(x-w/2), int(y-h/2)-5), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.8, (60, 20, 220), 2)
        # cv2.addText(img=img,text=cls,color=2)
    cv2.imwrite('boxes/box{}.png'.format(name),img)




if __name__ == '__main__':
    mainpath = 'history\DenseNet121-1024-Grid32-32'
    for img_name in os.listdir(mainpath):
        path = os.path.join(mainpath, img_name)
        if os.path.isfile(path):
            showLoss(path, img_name[8:-5])
