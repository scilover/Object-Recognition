import os
import numpy as np
import tensorflow as tf
from pycocotools.coco import COCO
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle


class GetDataIndex:

    def __init__(self, label_folder, data_type, img_size, grid_size):

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
        anns = coco.loadAnns(coco.getAnnIds())
        imgs = coco.loadImgs(coco.getImgIds())

        labels_ = {str(img['id']): [max(img['height'], img['width'])] for img in imgs}
        imgs = {img['file_name']: str(img['id']) for img in imgs}

        for ann in anns:
            id = str(ann['image_id'])
            if id in labels_:
                resize_ratio = self.img_size[0] / labels_[id][0]
                bbox = np.array(ann['bbox']) * resize_ratio
                xybbox = self._calReletiveLocBox(bbox.tolist())
                labels_[id].append([ann['category_id']] + xybbox)

        data = []
        for img_name, img_id in imgs.items():

            data.append((img_name, labels_[img_id][1:]))

        return data # set random_seed to fix changes.


class GetImage:
    """
    image batch preprocess, __call__ returns image batch tensor.
    """

    def __init__(self, all_image_paths, img_size):

        self.img_size = img_size
        self.all_image_paths = all_image_paths

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
        image = self.pad(image)
        image = tf.image.resize(image, self.img_size)
        image = tf.cast(image, tf.uint8) # Note: [0,255] is uint8, not int8 !!!
        #image /= 255.0  # normalize to [0,1] range !!!Notice: it can not be 255.0 for incompatibility
        #image /= 255

        return image

    def load_and_preprocess_image(self, path):
        image = tf.io.read_file(path)
        return self.preprocess_image(image)

    def __call__(self, *args, **kwargs):

        image_ds = [self.load_and_preprocess_image(path) for path in self.all_image_paths]

        return tf.convert_to_tensor(image_ds)


class GetLabel:
    """
    label batch preprocess, __call__ returns label tensor batch with shape (batch_size, d, d, 85).
    """

    def __init__(self, label, encoder, output_size):

        self.label = label
        self.encoder = encoder
        self.output_size = output_size

    def convert_label_to_tensor(self, old_label):

        new_label = np.zeros((self.output_size))
        for label in old_label:
            c, nx, ny, x, y, h, w = label
            new_label[ny, nx] = [1.] + self.encoder.transform([[c]]).toarray()[0].tolist() + [x, y, h, w]

        return new_label

    def __call__(self, *args, **kwargs):

        new_labels = []

        for label in self.label:
            new_labels.append(self.convert_label_to_tensor(label))

        return tf.convert_to_tensor(new_labels)



class GetData:
    """
    get (image, label) pairs batch for training and test.

    """
    def __init__(self, data, image_folder, img_size, grid_size, batch_size, encoder):

        self.data = data
        self.batch_size = batch_size
        self.image_folder = image_folder
        self.data_generator = iter(self.generator())
        self.encoder = encoder
        d = int(img_size[0] / grid_size[0])
        self.output_size = (d, d, 85)
        self.img_size = img_size


    def generator(self):
        sample_number = 128#len(self.data)
        for i in range(0, sample_number, self.batch_size):
            if i + self.batch_size <= sample_number:
                data = self.data[i: i + self.batch_size]
            else:
                data =  self.data[i:]
            img_names = [i[0] for i in data]
            labels = [i[1] for i in data]

            batch_image_paths = [os.path.join(self.image_folder, img_name) for img_name in img_names]
            images = GetImage(batch_image_paths, self.img_size)()
            labels = GetLabel(labels, self.encoder, self.output_size)()

            yield images, labels

    def __call__(self):

        # serialized_features_dataset = tf.data.Dataset.from_generator(
        #     self.generator, output_types=tf.string, output_shapes=())

        return self.data_generator



if __name__ == '__main__':

    label_folder = r'G:\Data\coco2017\annotations_trainval2017'
    image_folder = r'G:\Data\coco2017\val2017'

    encoder = OneHotEncoder(categories='auto')
    annFile = '{}/annotations/instances_{}.json'.format(label_folder, 'val2017')
    coco = COCO(annFile)
    cats = coco.loadCats(coco.getCatIds())
    cats = [[cat['id']] for cat in cats]
    encoder.fit(cats)

    data = GetData(image_folder, label_folder, 'val2017', (320, 320), (16, 16), 8, encoder)()
    k = next(data)


    print()
