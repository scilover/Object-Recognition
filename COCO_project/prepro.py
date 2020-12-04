import os
import numpy as np
import tensorflow as tf
from pycocotools.coco import COCO

class GetData:

    def __init__(self, label_folder, data_type, img_size=(320, 320), grid_size=(16,16)):

        self.label_folder = label_folder
        self.data_type = data_type
        self.grid_size = grid_size
        self.img_size = img_size


    def _getDic(self):
        """
        :return: a list of 85 elements: the 1st one represents exist or not, the next 80 elements represent
                    object category and the last 4 elements represent resized box location (x, y, h, w).
        """

        annFile = '{}/annotations/instances_{}.json'.format(self.label_folder, self.data_type)

        coco = COCO(annFile)
        # cats = coco.loadCats(coco.getCatIds())
        anns = coco.loadAnns(coco.getAnnIds())
        imgs = coco.loadImgs(coco.getImgIds())

        # labels_ = {str(img['id']): [max(img['height'], img['width'])] for img in imgs}
        # categories = [[cat['id']] for cat in cats]
        # encoder = OneHotEncoder(categories='auto')
        # encoder.fit(categories)
        labels_ = {str(img['id']): [] for img in imgs}

        for ann in anns:
            id = str(ann['image_id'])
            if id in labels_:
                labels_[id].append([ann['category_id']] + ann['bbox'])
                # labels_[id].append([1.] + encoder.transform([[ann['category_id']]]).toarray()[0].tolist() + ann['bbox'])
        imgs = {str(img['id']): img['file_name'] for img in imgs}
        dic = {}
        for key, value in labels_.items():
            dic[imgs[key]] = value
            # tmp = np.array(value[1:]) * self.img_size[0] / value[0]
            # labels[imgs[key]] = tmp.tolist()
        labels = {}
        for img_name, objects in dic.items():
            labels[img_name] = np.zeros((m, n, 5), dtype=np.float32)
            for box in objects:
                # nx, ny, x_, y_, h_, w_ = self._calReletiveLocBox(box[-4:])
                labels[img_name][ny, nx] = box[:-4] + [x_, y_, h_, w_]

            labels[img_name] = labels[img_name].tolist()


        return labels
