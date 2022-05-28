import json
import math
import os
from os.path import exists

import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from sklearn.neighbors import NearestNeighbors
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from tqdm import tqdm

import Autoencoders as A
import hash as H
from PIL import Image


class ImageSuggester():
    """The purpose of this class is to suggest k images from a gallery
    given a query image. It's implemented via a combination of Convolutional autoencoders,
    KNN and image hashing"""

    def __init__(self, gallery_path: str, query_gallery_path: str, device: str, flattened_embedding=None,
                 hash_dataframe=None, encoder=None, decoder=None):
        self.flattened_embedding = flattened_embedding
        self.hash_dataframe = hash_dataframe
        self.gallery_dir = gallery_path
        self.query_dir = query_gallery_path

        self.__ENCODER_PATH = "./encoder_model.pt"
        self.__DECODER_PATH = "./decoder_model.pt"
        self.__EMBEDDING_PATH = "./data_embedding.npy"
        self.__mapper = dict()
        self.__KEY_MAPPER_PATH = "JsonExampleResults/fembk.json"
        # encoder
        if encoder is None:
            self.encoder = A.ConvEncoder()
            if exists(self.__ENCODER_PATH):
                self.encoder.load_state_dict(torch.load(self.__ENCODER_PATH))
        else:
            self.encoder = encoder
        # decoder
        if decoder is None:
            self.decoder = A.ConvDecoder()
            if exists(self.__DECODER_PATH):
                self.decoder.load_state_dict(torch.load(self.__DECODER_PATH))
        else:
            self.decoder = decoder
        # embedding
        if flattened_embedding is None and exists(self.__EMBEDDING_PATH):
            self.flattened_embedding = np.load(self.__EMBEDDING_PATH)
        # device
        if device is None:
            self.device = "cpu"
        else:
            self.device = device

    def compute_similar_images(self, num_images: int, knn_metric="cosine", method="both"):
        """
        Given an image and number of similar images to search.
        Returns the num_images closest neares images.


        :param image_name: Name of the image whose similar images are to be found.
        :param num_images: Number of similar images to find.
        :param knn_metric: metric for knn when autoencoders are used
        :param method: can be:
            "auto-encoders" if only auto-encoders should be used.
            "hash" if only image hashing should be used
            "both" if both auto-encoders and image hashing should be used


        """

        query_images = os.listdir(self.query_dir)

        k = num_images

        if method == "hash":
            hasher = H.Hashes(self.gallery_dir, self.query_dir)
            hasher.build_hashes_df()

            hash_ris = dict()

            for image in query_images:
                hash_ris[image] = hasher.find_similar_images(image, k, method="min")
            return hash_ris
        if method == "auto-encoders":
            encoder_ris = dict()
            for image_name in query_images:
                list_ris = list()
                image = Image.open(self.query_dir + "//" + image_name).resize(A.IMAGE_RESIZE)

                image_tensor = T.ToTensor()(image)
                image_tensor = image_tensor.unsqueeze(0)

                with torch.no_grad():
                    image_embedding = self.encoder(image_tensor).cpu().detach().numpy()

                flat_embedding = image_embedding.reshape((image_embedding.shape[0], -1))

                knn = NearestNeighbors(n_neighbors=50, metric=knn_metric)
                knn.fit(self.flattened_embedding)

                distances, indices = knn.kneighbors(flat_embedding)

                indices_list = indices.tolist()[0]
                distances_list = distances.tolist()[0]

                assoc = dict(zip(indices_list, distances_list))
                # todo: fix for bad index
                for index in indices_list:
                    list_ris.append((self.__mapper[str(index)], assoc[index]))
                encoder_ris[image_name] = list_ris

            return encoder_ris

    def create_flattened_embedding(self, check=True, epochs=100):
        if check and exists("./data_embedding.npy"):
            print("Flattened embedding exists and is already loaded!")
            try:
                with open(self.__KEY_MAPPER_PATH, "r") as outfile:
                    self.__mapper = json.load(outfile)
            except:
                print("However something went wrong, reload with check=false please")
            return

        transforms = T.Compose([T.ToTensor()])  # Normalize the pixels and convert to tensor.
        full_dataset = A.FolderDataset(self.gallery_dir, transforms)  # Create folder dataset.

        # must be integers
        train_size = int(len(full_dataset) * 0.7)
        val_size = len(full_dataset) - train_size

        # Split data to train and test.pt
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        # Create the train dataloader
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # Create the validation dataloader
        val_loader = DataLoader(val_dataset, batch_size=32)

        # Create the full dataloader
        full_loader = DataLoader(full_dataset, batch_size=32)
        self.__mapper = dict(enumerate(full_dataset.get_all()))
        with open(self.__KEY_MAPPER_PATH, "w") as outfile:
            json.dump(self.__mapper, outfile)

        loss_fn = nn.MSELoss()  # We use Mean squared loss which computes difference between two images.

        # Shift models to GPU
        self.encoder.to(self.device)
        self.decoder.to(self.device)

        # Both the encoder and decoder parameters
        autoencoder_params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        optimizer = optim.Adam(autoencoder_params, lr=1e-3)  # Adam Optimizer
        max_loss = math.inf

        # Train

        # Training Loop
        for epoch in tqdm(range(epochs)):
            train_loss = A.train_step(self.encoder, self.decoder, train_loader, loss_fn, optimizer, device=self.device)
            print(f"Epochs = {epoch}, Training Loss : {train_loss}")
            val_loss = A.val_step(self.encoder, self.decoder, val_loader, loss_fn, device=self.device)
            print(f"Epochs = {epoch}, Validation Loss : {val_loss}")
            # Simple Best Model saving
            if val_loss < max_loss:
                max_loss = val_loss  # update max loss
                print("Validation Loss decreased, saving new best model")
                torch.save(self.encoder.state_dict(), "encoder_model.pt")
                torch.save(self.decoder.state_dict(), "decoder_model.pt")
        # Save the feature representations.
        embedding_shape = (1, 256, 8, 8)  # This we know from our encoder

        # We need feature representations for complete dataset not just train and validation.

        # Hence we use full loader here.
        embedding = A.create_embedding(self.encoder, full_loader, embedding_shape, self.device)

        # Convert embedding to numpy and save them
        numpy_embedding = embedding.cpu().detach().numpy()
        num_images = numpy_embedding.shape[0]

        # Save the embeddings for complete dataset, not just train
        flattened_embedding = numpy_embedding.reshape((num_images, -1))
        np.save("data_embedding.npy", flattened_embedding)
        self.flattened_embedding = flattened_embedding


# create a list of lists
image_reccomendation = dict()
gallery = "./challenge_test_data/gallery"
query = "./challenge_test_data/query"

s = ImageSuggester(gallery, query, "cuda")

# s.create_flattened_embedding()
ris = s.compute_similar_images(10, method="hash")
# print(ris)
final_ris = {"groupname": "Random Guys",
             'images': ris}

with open("JsonExampleResults/hash.json", "w") as f:
    json.dump(final_ris, f)

# print
d = final_ris['images']


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
