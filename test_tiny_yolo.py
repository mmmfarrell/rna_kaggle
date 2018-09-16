import numpy as np
import cv2

from tiny_yolo import make_tiny_yolo_model
from image_prep import load_and_prep_test_image
from utils import interpret_netout, parse_annotations, DataGenerator

if __name__ == '__main__':

    # LABELS = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
            # 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
            # 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    # config = dict(
        # IMAGE_H = 416, 
        # IMAGE_W = 416,
        # GRID_H = 13,
        # GRID_W = 13,
        # BOX = 5,
        # CLASS = len(LABELS),
        # CLASS_WEIGHTS = np.ones(len(LABELS), dtype='float32'),
        # THRESHOLD = 0.2,
        # LABELS = LABELS,
        # OBJ_THRESHOLD = 0.3,#0.5
        # NMS_THRESHOLD = 0.3,#0.45
        # ANCHORS = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11,
            # 16.62, 10.52],
        # NO_OBJECT_SCALE = 1.0,
        # OBJECT_SCALE = 5.0,
        # COORD_SCALE = 1.0,
        # CLASS_SCALE = 1.0,
        # BATCH_SIZE = 2,
        # WARM_UP_BATCHES = 0,
        # TRUE_BOX_BUFFER = 50,
        # ORIG_SIZE = 1024
    # )

    # model = make_tiny_yolo_model(config, print_model_summary=True)

    # model.load_weights("weights/yolov2-tiny-voc.h5")

    # # Test image with loaded yolo weights
    # input_image = load_and_prep_test_image('person.jpg')
    # netout = model.predict(input_image)
    # output_image = interpret_netout(input_image, netout[0], config)

    # # cv2.imshow("output", output_image)
    # # cv2.waitKey(0)

    ####### RSNA Pneumonia Challenge ###############
    train_img_dir = \
        '/home/mmmfarrell/deeplearning/tensorflow/rna_kaggle/kaggle-data/stage_1_train_images/'
    train_annotations_file = \
        '/home/mmmfarrell/deeplearning/tensorflow/rna_kaggle/kaggle-data/stage_1_train_labels.csv'

    annotations = parse_annotations(train_annotations_file)

    print(len(annotations))

    data_gen = DataGenerator(annotations, train_img_dir)

    print(data_gen[0])
