"""
These utils were adapted from
https://github.com/joycex99/tiny-yolo-keras/blob/master/utils.py
"""

import numpy as np
import cv2
import csv
import pydicom
import copy

import keras


class BoundBox:
    def __init__(self, class_num):
        self.x, self.y, self.w, self.h, self.c = 0., 0., 0., 0., 0.
        self.probs = np.zeros((class_num,))

    def iou(self, box):
        intersection = self.intersect(box)
        union = self.w*self.h + box.w*box.h - intersection
        return intersection/union

    def intersect(self, box):
        width  = self.__overlap([self.x-self.w/2, self.x+self.w/2], [box.x-box.w/2, box.x+box.w/2])
        height = self.__overlap([self.y-self.h/2, self.y+self.h/2], [box.y-box.h/2, box.y+box.h/2])
        return width * height

    def __overlap(self, interval_a, interval_b):
        x1, x2 = interval_a
        x3, x4 = interval_b
        if x3 < x1:
            if x4 < x1:
                return 0
            else:
                return min(x2,x4) - x1
        else:
            if x2 < x3:
                return 0
            else:
                return min(x2,x4) - x3


def interpret_netout(image, netout, config):
    IMAGE_H = config['IMAGE_H']
    IMAGE_W = config['IMAGE_W']
    CLASS = config['CLASS']
    BOX = config['BOX']
    GRID_H = config['GRID_H']
    GRID_W = config['GRID_W']
    ANCHORS = config['ANCHORS']
    THRESHOLD = config['THRESHOLD']
    LABELS = config['LABELS']

    boxes = []

    # interpret the output by the network
    for row in range(GRID_H):
        for col in range(GRID_W):
            for b in range(BOX):
                box = BoundBox(CLASS)

                # first 5 weights for x, y, w, h and confidence
                box.x, box.y, box.w, box.h, box.c = netout[row,col,b,:5]

                box.x = (col + sigmoid(box.x)) / GRID_W
                box.y = (row + sigmoid(box.y)) / GRID_H
                box.w = ANCHORS[2 * b + 0] * np.exp(box.w) / GRID_W
                box.h = ANCHORS[2 * b + 1] * np.exp(box.h) / GRID_H
                box.c = sigmoid(box.c)

                # rest of weights for class likelihoods
                classes = netout[row,col,b,5:]
                box.probs = softmax(classes) * box.c
                box.probs *= box.probs > THRESHOLD

                boxes.append(box)

    # suppress non-maximal boxes
    for c in range(CLASS):
        sorted_indices = list(reversed(np.argsort([box.probs[c] for box in boxes])))

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].probs[c] == 0:
                continue
            else:
                for j in range(i+1, len(sorted_indices)):
                    index_j = sorted_indices[j]

                    if boxes[index_i].iou(boxes[index_j]) >= 0.4:
                        boxes[index_j].probs[c] = 0

    print("Number of initial boxes: {}".format(len(boxes)))

    # Switch channels and make img (416, 416, 3) shape
    image = image[0, :, :, :]
    image = image[..., ::-1]
    output_image = image.copy()

    # draw the boxes using a threshold
    for box in boxes:
        max_indx = np.argmax(box.probs)
        max_prob = box.probs[max_indx]
        # print("Highest box probability for box: {}".format(max_prob))


        if max_prob > THRESHOLD:
            xmin  = int((box.x - box.w/2) * image.shape[1])
            xmax  = int((box.x + box.w/2) * image.shape[1])
            ymin  = int((box.y - box.h/2) * image.shape[0])
            ymax  = int((box.y + box.h/2) * image.shape[0])


            cv2.rectangle(output_image, (xmin,ymin), (xmax,ymax), (0,0,0), 2)
            cv2.putText(output_image, LABELS[max_indx], (xmin, ymin - 12), 0, 
                    1e-3 * image.shape[0], (0,255,0), 2)

    return output_image

def sigmoid(x):
    return 1. / (1.  + np.exp(-x))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def parse_annotations(train_annotations_file):
    annotations = {}

    with open(train_annotations_file, 'r') as f:
        reader = csv.reader(f)

        header = next(reader)

        for row in reader:
            patient_id = row[0]

            if patient_id not in annotations:
                annotations[patient_id] = {}
                annotations[patient_id]['file'] = patient_id + '.dcm'
                annotations[patient_id]['width'] = 1024
                annotations[patient_id]['height'] = 1024
                annotations[patient_id]['object'] = []

            if row[1] != '':
                new_object = {}
                new_object['name'] = 1
                new_object['xmin'] = float(row[1])
                new_object['ymin'] = float(row[2])
                new_object['xmax'] = float(row[1]) + float(row[3])
                new_object['ymax'] = float(row[2]) + float(row[4])
                annotations[patient_id]['object'].append(new_object)

    return annotations

def aug_img(train_instance, train_img_dir):
    NORM_H = 416
    NORM_W = 416
    GRID_H = 13
    GRID_W = 13
    BOX = 5

    image_file_name = train_img_dir + train_instance['file']
    dicom_image = pydicom.read_file(image_file_name) 
    image = dicom_image.pixel_array 

    # # Make image have 3 channels and normalize
    img = np.stack((image,)*3, -1)
    # image = cv2.resize(image, (416, 416))
    # image = image / 255.
    all_obj = copy.deepcopy(train_instance['object'])

    # path = train_instance['filename']
    # all_obj = copy.deepcopy(train_instance['object'][:])
    # img = cv2.imread(img_dir + path + ".JPEG")
    h, w, c = img.shape

    # scale the image
    scale = np.random.uniform() / 10. + 1.
    img = cv2.resize(img, (0,0), fx = scale, fy = scale)

    # translate the image
    max_offx = (scale-1.) * w
    max_offy = (scale-1.) * h
    offx = int(np.random.uniform() * max_offx)
    offy = int(np.random.uniform() * max_offy)
    img = img[offy : (offy + h), offx : (offx + w)]

    # # flip the image
    # flip = np.random.binomial(1, .5)
    # if flip > 0.5: img = cv2.flip(img, 1)

    # re-color
    # t  = [np.random.uniform()]
    # t += [np.random.uniform()]
    # t += [np.random.uniform()]
    # t = np.array(t)

    # img = img * (1 + t)
    img = img / (255. * 2.)

    # resize the image to standard size
    img = cv2.resize(img, (NORM_H, NORM_W))
    img = img[:,:,::-1]

    # fix object's position and size
    for obj in all_obj:
        for attr in ['xmin', 'xmax']:
            obj[attr] = int(obj[attr] * scale - offx)
            obj[attr] = int(obj[attr] * float(NORM_W) / w)
            obj[attr] = max(min(obj[attr], NORM_W), 0)

        for attr in ['ymin', 'ymax']:
            obj[attr] = int(obj[attr] * scale - offy)
            obj[attr] = int(obj[attr] * float(NORM_H) / h)
            obj[attr] = max(min(obj[attr], NORM_H), 0)

        # if flip > 0.5:
            # xmin = obj['xmin']
            # obj['xmin'] = NORM_W - obj['xmax']
            # obj['xmax'] = NORM_W - xmin

    return img, all_obj

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, annotations, train_img_dir, batch_size=32, shuffle=False):
        'Initialization'
        self.annotations = annotations
        self.annotation_keys = list(sorted(annotations.keys()))
        self.train_img_dir = train_img_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.annotations) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_annotations_temp = [self.annotations[self.annotation_keys[k]]
                for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_annotations_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.annotations))

        # TODO implement the shuffle
        # if self.shuffle == True:
            # np.random.shuffle(self.indexes)

    def __data_generation(self, list_annotations_temp):
        'Generates data containing batch_size samples'
        NORM_H = 416
        NORM_W = 416
        GRID_H = 13
        GRID_W = 13
        BOX = 5

        # Initialization
        X = np.empty((self.batch_size, 416, 416, 3))
        y = np.empty((self.batch_size, 13, 13, 5, 6))

        # Generate data
        for i, annotation in enumerate(list_annotations_temp):
            # print("orig anno", annotation)
            img, all_obj = aug_img(annotation, self.train_img_dir)
            # min_point = (all_obj[0]['xmin'], all_obj[0]['ymin'])
            # max_point = (all_obj[0]['xmax'], all_obj[0]['ymax'])
            # img = cv2.rectangle(img.copy(), min_point, max_point, (255, 0, 0))
            # print("aug objs", all_obj)

            # image_file_name = self.train_img_dir + annotation['file']
            # dicom_image = pydicom.read_file(image_file_name) 
            # image = dicom_image.pixel_array 

            # Make image have 3 channels and normalize
            # image = np.stack((image,)*3, -1)
            # image = cv2.resize(image, (416, 416))
            # image = image / 255.
            # min_point = (int(annotation['object'][0]['xmin']),
                    # int(annotation['object'][0]['ymin']))
            # max_point = (int(annotation['object'][0]['xmax']),
                    # int(annotation['object'][0]['ymax']))
            # print("min point", min_point)
            # print("max point", max_point)
            # image = cv2.rectangle(image.copy(), min_point, max_point, (255, 0, 0))
            # print("image shape", image.shape)
            # cv2.imshow("orig img", image)
            # TODO should I make is from -1. to 1.?

            # Store sample
            # cv2.imshow("img", img)
            # cv2.waitKey(0)
            X[i,] = img

            # construct output from object's position and size
            for obj in all_obj:
                box = []
                center_x = .5*(obj['xmin'] + obj['xmax']) #xmin, xmax
                center_x = center_x / (float(NORM_W) / GRID_W)
                center_y = .5*(obj['ymin'] + obj['ymax']) #ymin, ymax
                center_y = center_y / (float(NORM_H) / GRID_H)

                grid_x = int(np.floor(center_x))
                grid_y = int(np.floor(center_y))

                if grid_x < GRID_W and grid_y < GRID_H:
                    # obj_idx = labels.index(obj['name'])
                    obj_idx = 0
                    box = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']]

                    y[i, grid_y, grid_x, :, 0:4] = BOX * [box]
                    y[i, grid_y, grid_x, :, 4  ] = BOX * [1.]
                    # y[i, grid_y, grid_x, :, 5: ] = BOX * [[0.]*CLASS]
                    y[i, grid_y, grid_x, :, 5+obj_idx] = 1.0

        return X, y
