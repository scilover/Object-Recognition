import os, json
import numpy as np
import tensorflow as tf
from pycocotools.coco import COCO
from sklearn.preprocessing import OneHotEncoder



class GetDataIndex:

    def __init__(self, label_folder, data_type, img_size=(320, 320), grid_size=(16,16)):

        self.label_folder = label_folder
        self.data_type = data_type
        self.grid_size = grid_size
        self.img_size = img_size

    def _calReletiveLocBox(self, box):
        """
        box: should be a list of 4 elements [x, y, h, w]
        return: box index and locations relative to corresponding grid cell.
        """
        x, y, h, w = box
        dh, dw = self.grid_size
        nx, ny = int(x // dw), int(y // dh)

        x_ = (x - dw * nx) / dw
        y_ = (y - dh * ny) / dh
        h_ = h / dh
        w_ = w / dw
        # h_ = np.log(h / self.ref_h)  # v2
        # w_ = np.log(w / self.ref_w)  # v2

        return [nx, ny, x_, y_, h_, w_]

    def __call__(self):
        """
        :return: a list of 7 elements: the 1st one represents object category and the next two represent
                nx and ny, the last 4 elements represent resized relative box location (x, y, h, w).
        """

        annFile = '{}/annotations/instances_{}.json'.format(self.label_folder, self.data_type)

        coco = COCO(annFile)
        # cats = coco.loadCats(coco.getCatIds())
        anns = coco.loadAnns(coco.getAnnIds())
        imgs = coco.loadImgs(coco.getImgIds())

        labels_ = {str(img['id']): [max(img['height'], img['width'])] for img in imgs}
        imgs = {img['file_name']: str(img['id']) for img in imgs}
        # categories = [[cat['id']] for cat in cats]
        # encoder = OneHotEncoder(categories='auto')
        # encoder.fit(categories)

        for ann in anns:
            id = str(ann['image_id'])
            if id in labels_:
                resize_ratio = self.img_size[0] / labels_[id][0]
                bbox = np.array(ann['bbox']) * resize_ratio
                xybbox = self._calReletiveLocBox(bbox.tolist())
                labels_[id].append([ann['category_id']] + xybbox)
                # labels_[id].append([1.] + encoder.transform([[ann['category_id']]]).toarray()[0].tolist() + ann['bbox'])

        data = []
        for img_name, img_id in imgs.items():

            data.append((img_name, np.array(labels_[img_id][1:])))

        return data

def generator(data, batch_size):
    for i in range(0, len(data), batch_size):
        if i + batch_size <= len(data):
            yield data[i: i+batch_size]
        else:
            yield data[i:]


class GetImage:

    def __init__(self, img_size=(320,320)):

        self.img_size = img_size

    def pad(self, image):
        h, w = image.shape[:2] # deal with channel-last occasion
        p = h - w
        if p > 0:
            image = tf.pad(image, [[0,0], [0,p], [0,0]])
        elif p < 0:
            image = tf.pad(image, [[0,-p], [0,0], [0,0]])
        return image


    def preprocess_image(self, image):
        image = tf.image.decode_jpeg(image, channels=3)
        # resize_ratio = self.img_size[0] / max(image.shape[:-1])
        image = self.pad(image)
        image = tf.image.resize(image, self.img_size)
        image = tf.cast(image, tf.uint8) # Note: [0,255] is uint8, not int8 !!!
        #image /= 255.0  # normalize to [0,1] range !!!Notice: it can not be 255.0 for incompatibility
        #image /= 255

        return image

    def load_and_preprocess_image(self, path):
        image = tf.io.read_file(path)
        return self.preprocess_image(image)

    # def __call__(self, *args, **kwargs):
    #
    #     path_ds = tf.data.Dataset.from_tensor_slices(self.all_image_paths)
    #     image_ds = path_ds.map(self.load_and_preprocess_image)
    #
    #     return image_ds

# class GetData:
#     """
#     get (image, label) pairs for training and test.
#
#     """
#     def __init__(self, image_folder, label_folder, data_type, grid_size=(32,32), batch_size=8):
#
#         self.image_folder = image_folder
#         self.label_folder = label_folder
#         self.data_type = data_type
#         self.batch_size = batch_size
#         self.grid_size = grid_size
#
#     def _devideLabels(self, labels, devide_index=3):
#         """
#
#         :param labels: input labels to be devided.
#         :param devide_index: devide labels in the last dimension
#         :return: devided labels
#         """
#
#         labels = np.array(labels)
#
#         return labels[...,0], labels[...,1:devide_index], labels[...,devide_index:]
#
#
#     def __call__(self, tensor_generator=True):
#
#         # labels = GetLabel(self.label_folder, self.data_type, grid_size=self.grid_size)()
#         # loader = GetImage(self.image_folder)
#         labels = GetLabel(self.label_folder, self.data_type)._getDic()
#         images, targets = [], []
#         for key, label in labels.items():
#             img_path = os.path.join(self.image_folder, key)
#             images.append([img_path])
#             # images.append(loader.load_and_preprocess_image(img_path))
#             targets.append(label)
#
#         if tensor_generator:
#             data = tf.data.Dataset.from_tensor_slices((images,targets)).batch(self.batch_size)
#             # labels = tf.data.Dataset.from_tensor_slices(labels).batch(self.batch_size)
#
#         else:
#             images = tf.convert_to_tensor(images)
#             labels = tf.convert_to_tensor(labels)
#             data = (images, labels)
#
#         # x_train, x_test, y_train, y_test = \
#         #     train_test_split(images, targets, test_size = test_size, random_state = random_state)
#
#         return data


if __name__ == '__main__':

    label_folder = r'G:\Data\coco2017\annotations_trainval2017'
    # labels = GetLabel(xml_folder)()
    # print(labels['010000.jpg'].shape)
    image_folder = r'G:\Data\coco2017\val2017'
    #image_ds = GetImage(image_folder)()
    data = GetDataIndex(label_folder, 'val2017', img_size=(320, 320), grid_size=(16,16))()
    # labels = GetLabel(label_folder, 'val2017')()
    # with open('val_labels.json', 'w') as obj:
    #     json.dump(labels, obj)
    print()
