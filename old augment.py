import os
import random

import imageio.v2 as io
import matplotlib

from matplotlib import pyplot as plt
from skimage.transform import resize
import numpy as np
from scipy import ndimage

#matplotlib.use('Agg')

dataset = "./test/"
augmented_dataset = "./per_stefano/"
shape = (128, 128, 3)

categories = os.listdir(dataset)


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
        # noisy
        ni = image.copy()
        ni = ni + np.random.normal(0, 1, ni.shape)
        data = np.clip(ni, 0, 255)
        ni = data.astype(np.uint8)
        return "noisy", ni
    elif kind == 4:
        # horizontal flip
        ni = image.copy()
        ni = np.flip(ni, 1)
        ni = np.ascontiguousarray(ni)
        return "flipped horizontally", ni
    elif kind == 5:
        # little crop
        return "little_crop", resize(image[4:124, 4:124, :], shape,
                                     anti_aliasing=True)
    elif kind == 6:
        h1 = random.randint(0, 120)
        h2 = random.randint(0, 120)
        ni = image.copy()
        ni[h1:h1+20, h2:h2+20, 0] = 255  # white
        ni[h1:h1+20, h2:h2+20, 1] = 255  # white
        ni[h1:h1+20, h2:h2+20, 2] = 255  # white
        # white bars
        return "white_bars", ni
    elif kind == 7:
        # pixellated
        ni = image.copy()
        ni = ni[::3, ::3, :]
        ni = resize(ni, shape, anti_aliasing=True)
        return "pixel", ni


def change_color(kind, image):
    global shape
    # if image.shape != shape:
    #    return None
    if kind == 0:
        # channel 0
        return "first_channel", image[:, :, 0]
    elif kind == 1:
        # channel 1
        return "second_channel", image[:, :, 1]
    elif kind == 2:
        # channel 2
        return "last_channel", image[:, :, 2]
    elif kind == 3:
        # darker
        return "darkened", image[:, :, :] // 2
    elif kind == 4:
        # red
        ni = image.copy()
        ni[:, :, 1] = 0
        ni[:, :, 2] = 0
        return "red", ni
    elif kind == 5:
        # green
        ni = image.copy()
        ni[:, :, 0] = 0
        ni[:, :, 2] = 0
        return "green", ni
    elif kind == 6:
        # blue
        ni = image.copy()
        ni[:, :, 0] = 0
        ni[:, :, 1] = 0
        return "blue", ni
    elif kind == 7:
        # no red
        ni = image.copy()
        ni[:, :, 0] = 0
        return "no_red", ni
    elif kind == 8:
        # no green
        ni = image.copy()
        ni[:, :, 1] = 0
        return "no_green", ni
    elif kind == 9:
        # no blue
        ni = image.copy()
        ni[:, :, 2] = 0
        return "no_blue", ni
    elif kind == 10:
        # gray
        ni = image.copy()

        ni[:, :, 1] = ni[:, :, 0]
        ni[:, :, 2] = ni[:, :, 0]
        return "gray", ni
    elif kind == 11:
        # lighten
        light = 100
        return "lighten", image + np.minimum(255 - image, light)
    # elif kind == 12:
    #     # horizontal fade
    #     ls = np.linspace(255, 0, image.shape[1])
    #     horiz_fade = np.tile(ls, (image.shape[0], 1, 1))
    #     return "horizontal_fade", image - np.minimum(image, horiz_fade)


def augment_no_color(directory):
    global augmented_dataset
    #t = os.listdir(augmented_dataset + directory)
    t = os.listdir(directory)

    for image in t:
        #i = io.imread(augmented_dataset + directory + "/" + image)
        i = io.imread(directory+ "/" + image)
        # random.shuffle(order)
        for j in range(0, 9):
            to_save = transform(j, i)
            #print(to_save)
            if to_save is not None:
                #plt.imsave(augmented_dataset + directory + "/" + to_save[0] + "_" + image, to_save[1], format="jpeg")
                plt.imsave(augmented_dataset  + "/" + to_save[0] + "_" + image, to_save[1], format="jpeg")


def augment_color(directory):
    global augmented_dataset
    #t = os.listdir(augmented_dataset + directory)
    t = os.listdir(directory)

    for image in t:
        #i = io.imread(augmented_dataset + directory + "/" + image)
        i = io.imread( directory + "/" + image)
        # random.shuffle(order)
        for j in range(0, 12):
            to_save = change_color(j, i)
            # print("working on " + to_save[0])
            if to_save is not None:
                #plt.imsave(augmented_dataset + directory + "/" + to_save[0] + "_" + image, to_save[1], format="png")
                plt.imsave(augmented_dataset + "/" + to_save[0] + "_" + image, to_save[1], format="png")


# reshape
def reshape(directory):
    global dataset
#    t = os.listdir(dataset + directory)
    t = os.listdir( directory)

    for image in t:
#    for image in directory:
#        i = io.imread(dataset + directory + "/" + image)
        i = io.imread(directory + "/" + image)
        # random.shuffle(order)
        to_save = resize(i, shape,
                         anti_aliasing=True)
        if to_save is not None:
#            plt.imsave(augmented_dataset + directory + "/" + image, to_save, format="png")
            plt.imsave("./test2/" + image, to_save, format="png")


# create dir
try:
    os.mkdir(augmented_dataset)
    for cat in categories:
        os.mkdir(augmented_dataset + cat)
except:
    print("directories are already there")

# reshape
#for cat in categories:
print("reshaping " )#+ cat)
reshape(dataset)

# augment without color
#for cat in categories:
print("working on no color") # + cat + "(no color)")
augment_no_color("./test2/")

# augment with color
#for cat in categories:
print("working on color")# + cat + "(final)")
augment_color("./test2/")