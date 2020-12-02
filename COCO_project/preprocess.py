import os
import numpy as np
import tensorflow as tf
from pycocotools.coco import COCO
from sklearn.preprocessing import OneHotEncoder

# from parser_xml import ParserXML


class GetLabel:

    def __init__(self, label_folder, data_type, img_size=(320, 320), grid_size=(32,32)):

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
        cats = coco.loadCats(coco.getCatIds())
        anns = coco.loadAnns(coco.getAnnIds())
        imgs = coco.loadImgs(coco.getImgIds())

        labels_ = {str(img['id']): [max(img['height'], img['width'])] for img in imgs}
        categories = [[cat['id']] for cat in cats]
        encoder = OneHotEncoder(categories='auto')
        encoder.fit(categories)

        for ann in anns:
            id = str(ann['image_id'])
            if id in labels_:
                labels_[id].append([1.] + encoder.transform([[ann['category_id']]]).toarray()[0].tolist() + ann['bbox'])
        imgs = {str(img['id']): img['file_name'] for img in imgs}
        labels = {}
        for key, value in labels_.items():
            tmp = np.array(value[1:]) * self.img_size[0] / value[0]
            # tmp = tmp.astype(np.float32)
            labels[imgs[key]] = tmp.tolist()


        return labels


    def _calAbsoluteLocBox(self, box):
        """
        :input box should be a list of [xmin, ymin, xmax, ymax]
        :return: absolute coordinate (x, y) of the object and box size (h, w)
        """
        xmin, ymin, xmax, ymax = box
        x = (xmin + xmax) // 2
        y = (ymin + ymax) // 2
        h = ymax - ymin
        w = xmax- xmin

        return [x, y, h, w]

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
        :return: labels in shape (m, n, 7)
        """
        dh, dw = self.grid_size
        H, W = self.img_size
        m, n = int(H/dh), int(W/dw)
        dic = self._getDic()

        labels = {}
        for img_name, objects in dic.items():
            labels[img_name] = np.zeros((m, n, 85), dtype=np.float32)
            for box in objects:
                # box = self._calAbsoluteLocBox(obj[5:])

                nx, ny, x_, y_, h_, w_ = self._calReletiveLocBox(box[-4:])
                labels[img_name][ny, nx] = box[:-4] + [x_, y_, h_, w_]

        return labels

class GetImage:

    def __init__(self, image_folder, img_size=(320,320)):

        self.all_image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder)]
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

        return image.numpy()

    def load_and_preprocess_image(self, path):
        image = tf.io.read_file(path)
        return self.preprocess_image(image)

    def __call__(self, *args, **kwargs):

        path_ds = tf.data.Dataset.from_tensor_slices(self.all_image_paths)
        image_ds = path_ds.map(self.load_and_preprocess_image)

        return image_ds

class GetData:
    """
    get (image, label) pairs for training and test.

    """
    def __init__(self, image_folder, label_folder, data_type, grid_size=(32,32), batch_size=8):

        self.image_folder = image_folder
        self.label_folder = label_folder
        self.data_type = data_type
        self.batch_size = batch_size
        self.grid_size = grid_size

    def _devideLabels(self, labels, devide_index=3):
        """

        :param labels: input labels to be devided.
        :param devide_index: devide labels in the last dimension
        :return: devided labels
        """

        labels = np.array(labels)

        return labels[...,0], labels[...,1:devide_index], labels[...,devide_index:]


    def __call__(self, tensor_generator=True):

        labels = GetLabel(self.label_folder, self.data_type, grid_size=self.grid_size)()
        loader = GetImage(self.image_folder)
        images, targets = [], []
        for key, label in labels.items():
            img_path = os.path.join(self.image_folder, key)
            images.append(loader.load_and_preprocess_image(img_path))
            targets.append(label)

        # label_exist, label_category, label_location = self._devideLabels(targets)
        if tensor_generator:
            data = tf.data.Dataset.from_tensor_slices((images,targets)).batch(self.batch_size)
            # labels = tf.data.Dataset.from_tensor_slices(labels).batch(self.batch_size)

        else:
            images = tf.convert_to_tensor(images)
            labels = tf.convert_to_tensor(labels)
            data = (images, labels)

        # x_train, x_test, y_train, y_test = \
        #     train_test_split(images, targets, test_size = test_size, random_state = random_state)

        # train_data = \
        #     tf.data.Dataset.from_tensor_slices((x_train, y_train_category, y_train_location)).batch(self.batch_size)
        # test_data = \
        #     tf.data.Dataset.from_tensor_slices((x_test, y_test_category, y_test_location)).batch(self.batch_size)

        return data


if __name__ == '__main__':

    label_folder = r'G:\Data\coco2017\annotations_trainval2017'
    # labels = GetLabel(xml_folder)()
    # print(labels['010000.jpg'].shape)
    image_folder = r'G:\Data\coco2017\val2017'
    #image_ds = GetImage(image_folder)()
    data = GetData(grid_size=(32,32), image_folder=image_folder, data_type='val2017', label_folder=label_folder)()

    print()
