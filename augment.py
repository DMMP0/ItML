import os
import random

import cv2
import imageio.v2 as io
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage.transform import resize

# matplotlib.use('Agg')

dataset = "./dataset/gallery/"  # <--
augmented_dataset = "./dataset/augmented/"  # <--
shape = (128, 128, 3)


# print(categories)


def transform(kind, image):
    global shape
    # if image.shape != shape:
    #    return None
    if kind == 0:
        # flip vertically
        ni = image.copy()
        ni = np.flip(ni, 0)
        ni = np.ascontiguousarray(ni)
        return "flipped vertically", ni
    elif kind == 1:
        # random rotation
        return "random_rotated", ndimage.rotate(image, random.randint(0, 300))
    elif kind == 2:
        # random rotation without reshaping
        return "random_rotated_not_reshaped", ndimage.rotate(image, random.randint(0, 300), reshape=False)
    elif kind == 3:
        ni = image.copy()
        # Store height and width of the image
        height, width = ni.shape[:2]
        quarter_height, quarter_width = height / 4, width / 4
        T = np.float32([[1, 0, quarter_width], [0, 1, quarter_height]])
        # We use warpAffine to transform the image using the matrix, T
        return "translated", cv2.warpAffine(image, T, (width, height))
    elif kind == 4:
        # horizontal flip
        ni = image.copy()
        ni = np.flip(ni, 1)
        ni = np.ascontiguousarray(ni)
        return "flipped horizontally", ni


def augment_no_color():
    global augmented_dataset
    t = os.listdir(augmented_dataset)

    for image in t:
        i = io.imread(augmented_dataset + "/" + image)

        # random.shuffle(order)
        for j in range(0, 9):
            to_save = transform(j, i)
            # print(to_save)
            if to_save is not None:
                plt.imsave(augmented_dataset + "/" + to_save[0] + "_" + image, to_save[1], format="jpeg")


# reshape
def reshape():
    global dataset
    t = os.listdir(dataset)

    for image in t:
        i = io.imread(dataset + "/" + image)
        # random.shuffle(order)
        to_save = resize(i, shape,
                         anti_aliasing=True)
        if to_save is not None:
            plt.imsave(augmented_dataset + "/" + image, to_save, format="png")


# create dir
try:
    os.mkdir(augmented_dataset)
except:
    print("directories are already there")

# reshape

reshape()

# augment without color

print("working on augment")
augment_no_color()
