import json
import os

import PIL
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
import torch
from tqdm import tqdm
from torchvision import models
import pandas as pd
import numpy as np

# needed input dimensions for the CNN
inputDim = (224, 224)
inputDir = "./challenge_test_data/gallery"
inputDirCNN = "inputImagesCNN"
query_dir = "./challenge_test_data/query"

query_list = os.listdir(query_dir)

os.makedirs(inputDirCNN, exist_ok=True)

transformationForCNNInput = transforms.Compose(
    [transforms.Resize(inputDim), transforms.Grayscale(num_output_channels=3)])

for imageName in os.listdir(inputDir):
    I = Image.open(os.path.join(inputDir, imageName))
    newI = transformationForCNNInput(I)
    #
    # copy the rotation information metadata from original image and save, else your transformed images may be rotated
    newI.save(os.path.join(inputDirCNN, imageName))

    newI.close()
    I.close()

for imageName in query_list:
    I = Image.open(os.path.join(query_dir, imageName))
    newI = transformationForCNNInput(I)
    #
    # copy the rotation information metadata from original image and save, else your transformed images may be rotated
    newI.save(os.path.join(inputDirCNN, imageName))

    newI.close()
    I.close()


class Img2VecResnet18():
    def __init__(self):
        self.device = torch.device("cuda:0")
        self.numberFeatures = 256
        self.modelName = "resnet-18"
        self.model, self.featureLayer = self.getFeatureLayer()
        self.model = self.model.to(self.device)
        self.model.eval()
        self.toTensor = transforms.ToTensor()

        # normalize the resized images as expected by resnet18
        # [0.485, 0.456, 0.406] --> normalized mean value of ImageNet, [0.229, 0.224, 0.225] std of ImageNet
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def getVec(self, img):
        image = self.normalize(self.toTensor(img)).unsqueeze(0).to(self.device)

        embedding = torch.zeros(1, self.numberFeatures, 6, 6)

        def copyData(m, i, o): embedding.copy_(o.data)

        h = self.featureLayer.register_forward_hook(copyData)
        self.model(image)
        h.remove()

        return embedding.numpy()[0, :, 0, 0]

    def getFeatureLayer(self):
        cnnModel = models.alexnet(pretrained=True)
        layer = cnnModel._modules.get('avgpool')
        self.layer_output_size = 512

        return cnnModel, layer


# generate vectors for all the images in the set
img2vec = Img2VecResnet18()

allVectors = {}
print("Converting images to feature vectors:")
for image in tqdm(os.listdir("inputImagesCNN")):
    I = Image.open(os.path.join("inputImagesCNN", image))
    vec = img2vec.getVec(I)
    allVectors[image] = vec
    I.close()

for image in query_list:
    I = Image.open(os.path.join("inputImagesCNN", image))
    vec = img2vec.getVec(I)
    allVectors[image] = vec
    I.close()


# now let us define a function that calculates the cosine similarity entries in the similarity matrix
# print(allVectors["n01443537_21198.JPEG"])

def getSimilarityMatrix(vectors):
    v = np.array(list(vectors.values())).T
    sim = np.inner(v.T, v.T) / (
            (np.linalg.norm(v, axis=0).reshape(-1, 1)) * ((np.linalg.norm(v, axis=0).reshape(-1, 1)).T))
    keys = list(vectors.keys())
    matrix = pd.DataFrame(sim, columns=keys, index=keys)

    return matrix


similarityMatrix = getSimilarityMatrix(allVectors)
similarityMatrix.drop(columns=query_list, axis=1, inplace=True)
k = 11  # the number of top similar images to be stored


similarNames = pd.DataFrame(index=similarityMatrix.index, columns=range(k))
# similarValues = pd.DataFrame(index=similarityMatrix.index, columns=range(k))

for j in tqdm(range(similarityMatrix.shape[0])):
    kSimilar = similarityMatrix.iloc[j, :].sort_values(ascending=False).head(k)
    similarNames.iloc[j, :] = list(kSimilar.index)
    # similarValues.iloc[j, :] = kSimilar.values



def jsonify(q, df):
    ris = dict()
    ris["groupname"] = "request"
    ris["images"] = dict()

    for img in q:
        ris["images"][img] = df.loc[[img]].values.tolist()[0][1:]
    with open("JsonExampleResults/result_pytorch.json", "w") as f:
        json.dump(ris, f)
    return ris


to_plot = jsonify(query_list, similarNames)
to_plot = to_plot["images"]


def plot_em_all(dict_of_lists, gallery_path, query_path, size=(224, 224)):
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


plot_em_all(to_plot, inputDir, query_dir)
