import copy
import os
from os.path import exists

import distance
import imagehash
import pandas as pd
from PIL import Image


class Hashes:

    def __init__(self, gallery_path: str, query_gallery_path, hashes_dataframe=None, ):
        self.query_gallery_path = query_gallery_path
        self.gallery_path = gallery_path

        self.hashes_dataframe = hashes_dataframe
        self.__hash_df_path = "./hashes_df.csv"

    def build_hashes_df(self, check=True):

        if check and exists(self.__hash_df_path):
            self.hashes_dataframe = pd.read_csv(self.__hash_df_path)
            print("Hash dataframe already exists as a file and has just been loaded!")
            return

        self.hashes_dataframe = pd.DataFrame(columns=['image', 'ahash', 'phash', 'dhash', 'whash', 'colorhash'])
        for img in os.listdir(self.gallery_path):
            file = Image.open(self.gallery_path+"/"+img).resize((64,64)).convert("L")

            data = {
                'image': img,
                'ahash': imagehash.average_hash(file),
                'phash': imagehash.phash(file),
                'dhash': imagehash.dhash(file),
                'whash': imagehash.whash(file),
                'colorhash': imagehash.colorhash(file),
            }

            self.hashes_dataframe.loc[len(self.hashes_dataframe)] = data

        print("Gallery hashes saved inside the class instance.\n")
        try:
            self.hashes_dataframe.to_csv("hashes_df.csv")
            print("Gallery hashes has been saved also as a file\n")
        except:
            print("error in saving hashes to file.\nNB: The dataframe is still present in the class instance!")

    def find_similar_images(self, query_image_name: str, top_n: int, method="avg", ) -> list:
        """Compare an unseen image to previously seen images and return
        a list of images ranked by their similarity according to the
        Hamming distance of:
            their average hash,
            perceptual hash,
            difference hash,
            wavelet hash,
            colorhash.

        NB: hashes dataframe must be present in the current instance

        :param query_image_name: name of the query image
        :param top_n: number of images to return
        :param method: rationale for calculating similarity score
            'avg' = average of hamming distances,
            'sum' = sum of hamming distances,
            'max' = highest of hamming distances ,
            'min' = lowest of of hamming distances,
            'ahash' = only the average hash,
            'phash' = only the perceptual hash,
            'dhash' = only the difference hash,
            'whash' = only the wavelet hash,
            'colorhash' = only the color hash


        :return
            list of tuples with images name and distance from the target image
            es:
            [('img1',0), ('img2',5), ...]

        """

        metrics = ['ahash', 'phash', 'dhash', 'whash', 'colorhash']
        file = Image.open(self.query_gallery_path + "//" + query_image_name)
        query_metrics = dict()
        #   create metrics for query image
        for metric in metrics:
            if metric == "ahash":
                query_metrics[metric] = str(imagehash.average_hash(file))
            elif metric == "phash":
                query_metrics[metric] = imagehash.phash(file)
            elif metric == "dhash":
                query_metrics[metric] = str(imagehash.dhash(file))
            elif metric == "whash":
                query_metrics[metric] = imagehash.whash(file)
            else:
                query_metrics["colorhash"] = str(imagehash.colorhash(file))

        tmp_df = copy.deepcopy(self.hashes_dataframe)

        # all hamming distances
        cols = list()
        for metric in metrics:
            cols.append(metric + "hamming_distance")
            tmp_df[metric + "hamming_distance"] = tmp_df.apply(
                lambda x: distance.hamming(str(x[metric]), str(query_metrics[metric])), axis=1)

        # final metric
        if method == "avg":
            tmp_df['final_hamming_distance'] = tmp_df[cols].sum(axis=1)
            tmp_df['final_hamming_distance'] /= len(metrics)
        elif method == "sum":
            tmp_df['final_hamming_distance'] = tmp_df[cols].sum(axis=1)
        elif method == "max":
            tmp_df['final_hamming_distance'] = tmp_df[cols].max(axis=1)
        elif method == "min":
            tmp_df['final_hamming_distance'] = tmp_df[cols].min(axis=1)
        else:
            tmp_df['final_hamming_distance'] = tmp_df[method]

        tmp_df = tmp_df.sort_values(by='final_hamming_distance', ascending=True)
        # ris = tmp_df.head(top_n)[["image", "final_hamming_distance"]].apply(tuple, axis=1).tolist()
        ris = tmp_df.head(top_n)["image"].tolist()
        return ris




