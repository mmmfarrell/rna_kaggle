import numpy as np
import cv2

def load_and_prep_test_image(file_name):
    image = cv2.imread('./test_images/' + file_name)
    image = resize_and_normalize_image(image)

    return image


def resize_and_normalize_image(image):
    image = cv2.resize(image, (416, 416))
    image = image / 255.
    image = image[:, :, ::-1]
    image = np.expand_dims(image, 0)

    return image
