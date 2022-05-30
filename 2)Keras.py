import json
import os
from os.path import exists

import PIL
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
from matplotlib import pyplot as plt
from scipy.spatial import distance

# redis
# github
# model_url = "https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2"
model_url = "https://tfhub.dev/google/efficientnet/b5/feature-vector/1"

gallery = "./challenge_test_data/gallery"
query = "./challenge_test_data/query"
IMAGE_SHAPE = (456, 456)
layer = hub.KerasLayer(model_url)
model = tf.keras.Sequential([layer])


def extract(file):
    file = Image.open(file).convert('L').resize(IMAGE_SHAPE)
    #                       grayscale
    # display(file)

    file = np.stack((file,) * 3, axis=-1)

    file = np.array(file) / 255.0

    embedding = model.predict(file[np.newaxis, ...])
    # print(embedding)
    imgnet_feature_np = np.array(embedding)
    flattended_feature = imgnet_feature_np.flatten()

    # print(len(flattended_feature))
    # print(flattended_feature)
    # print('-----------')
    return flattended_feature


def compute_similar_images(query_gallery_path, gallery_path, metric="cosine"):
    ris = list()
    gallery_imgs = os.listdir(gallery_path)
    query_imgs = os.listdir(query_gallery_path)
    q_features = list()
    g_features = list()
    for img_q in query_imgs:
        q_features.append((img_q, extract(query_gallery_path + "/" + img_q)))
    print("query done")
    for img_g in gallery_imgs:
        g_features.append((img_g, extract(gallery_path + "/" + img_g)))
    print("gallery done\nstarting distances")
    for im1 in q_features:
        tmp_lst = list()
        tmp_lst.append((im1[0], distance.cdist([im1[1]], [im1[1]], metric)[0]))
        for im2 in g_features:
            tmp_lst.append((im2[0], distance.cdist([im1[1]], [im2[1]], metric)[0]))

        df = pd.DataFrame(tmp_lst, columns=['Image', 'Distance'])
        df = df.sort_values(by="Distance")
        ris.append(df)
        print(im1[0] + " done\n")
    return ris


def jsonify(lists, top_k):
    ris = dict()
    ris["groupname"] = "Random Guys"  # nb: NOT REQUEST
    ris["images"] = dict()

    for lst in lists:
        ris["images"][lst[0]] = list()
        i = 0
        for el in lst:
            if i < top_k:
                ris["images"][lst[0]].append(el)
                i += 1
            else:
                break
    with open("result_only_pretrain.json", "w") as f:
        json.dump(ris, f)
    return ris


if exists("result_only_pretrain.json"):
    with open("result_only_pretrain.json", "r") as f:
        ris = json.load(f)
else:
    lst = compute_similar_images(query, gallery)
    print("all done")
    lsts = list()
    i = 0
    while i < len(lst):
        lsts.append(lst[i]["Image"])
        i += 1
    ris = jsonify(lsts, 10)

# print
# image similarity pytorch
d = ris['images']


def plot_em(dict_of_lists, gallery_path, query_path, size=(456, 456)):
    for key in dict_of_lists:

        ims = [Image.open(gallery_path + '/' + x).resize(size) for x in dict_of_lists[key][1:]]
        ims.insert(0, PIL.Image.open(query_path + "/" + key).resize(size))
        widths, heights = zip(*(i.size for i in ims))

        total_width = sum(widths)
        max_height = max(heights)

        new_im = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for im in ims:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]

        plt.imshow(new_im)
        plt.show()


plot_em(d, gallery, query)
