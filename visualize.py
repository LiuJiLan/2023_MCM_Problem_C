import os.path
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.manifold import TSNE


from model import *
from main import load_file, WEIGHT_PATH, BATCH_SIZE, MANUAL_SEED

import torch
from torch.utils.data import random_split, DataLoader

STAT_NAME = "1_1.7400736396666616e-05.pth"

stat_pth = os.path.join(WEIGHT_PATH, STAT_NAME)
checkpoint = torch.load(stat_pth)

# def run_model(one_hot):
#     model = Model()
#     model.eval()
#     model.load_state_dict(checkpoint['net'])
#     with torch.no_grad():
#         return model(one_hot.view(1, 5, 26))

if __name__ == "__main__":
    dataset = load_file()

    train_data, test_data = random_split(dataset, [0.8, 0.2],
                                         generator=torch.Generator().manual_seed(MANUAL_SEED))
    # print("Train Length: {}".format(len(train_data)))
    # print("Test Length: {}".format(len(test_data)))

    # 时序分析是不能打乱
    # train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    model = Model()
    model.eval()
    model.load_state_dict(checkpoint['net'])

    embeddings = []
    labels = []

    with torch.no_grad():
        for one_hot, tries in test_dataloader:
            output = model.embedding(one_hot)
            output = model.embedding(one_hot)
            for i in range(one_hot.shape[0]):
                embeddings.append(output[i].numpy().reshape(1, -1))
                labels.append(tries[i].numpy().reshape(1, -1))

    # Concatenate the embedding vectors and their corresponding labels
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    print(embeddings.shape)
    print(labels.shape)

    # Apply t-SNE to reduce the dimensionality of the embedding vectors
    tsne = TSNE(n_components=2, perplexity=30, verbose=2)
    embeddings_tsne = tsne.fit_transform(embeddings)

    # Visualize the embedding vectors using matplotlib
    # plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=labels, cmap="viridis")
    plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=labels, cmap="viridis")
    plt.colorbar()
    plt.show()