import os
import numpy as np
import tensorflow as tf

from parser_xml import ParserXML


class GetLabel:

    def __init__(self, xml_folder, img_size=(480, 640), grid_size=(32,32)):

        self.xml_folder = xml_folder
        self.grid_size = grid_size
        self.img_size = img_size
        self.ref_h = 50
        self.ref_w = 50

    def _xml_parser(self):
        """
        :return: a list of 7 elements: object category and box location (xmin, ymin, xmax,ymax)
        """
        parser = ParserXML()
        for xml in os.listdir(self.xml_folder):
            xml_file = os.path.join(self.xml_folder, xml)
            parser._load_data(xml_file)
        return parser.data_dic

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
        dic = self._xml_parser()

        labels = {}
        # boxes = {} ##############
        for img_name, objects in dic.items():
            labels[img_name] = np.zeros((m, n, 7))
            # boxes[img_name] = []##########
            for obj in objects:
                box = self._calAbsoluteLocBox(obj[5:])
                nx, ny, x_, y_, h_, w_ = self._calReletiveLocBox(box)
                labels[img_name][ny, nx] = obj[:3] + [x_, y_, h_, w_]
                # boxes[img_name].append(box)##########

        return labels

class GetImage:

    def __init__(self, image_folder):

        self.all_image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder)]

    def preprocess_image(self, image):
        image = tf.image.decode_jpeg(image, channels=3)
        #image = tf.image.resize(image, [192, 192])
        #image /= 255.0  # normalize to [0,1] range !!!Notice: it can not be 255.0 for incompatibility
        #image /= 255

        return image

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
    def __init__(self, image_folder, label_folder, grid_size=(32,32), batch_size=32):

        self.image_folder = image_folder
        self.label_folder = label_folder
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


        labels = GetLabel(self.label_folder, grid_size=self.grid_size)()
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

    label_folder = r'G:\Data\yoloV4自己的数据集\Annotations\testLabels'
    # labels = GetLabel(xml_folder)()
    # print(labels['010000.jpg'].shape)
    image_folder = r'G:\Data\yoloV4自己的数据集\JPEGImages\testImages'
    #image_ds = GetImage(image_folder)()
    data = GetData(grid_size=(20,20), image_folder=image_folder, label_folder=label_folder)()
    print()
