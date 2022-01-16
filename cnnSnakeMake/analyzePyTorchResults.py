#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
import glob
import torch
from torch_geometric.utils.convert import from_networkx
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
from torch_geometric.nn import GCNConv, GINConv, GATv2Conv
import random
import math
from sklearn.manifold import TSNE
import matplotlib.colors
import time
from sklearn.model_selection import KFold
from models import *
from sys import argv
import pickle as pkl

seed = 2
torch.manual_seed(seed)
np.random.seed(seed)

def testCNNs(networksFile, featuresFile, modelConfigFile):
    dataState = pkl.load(open(networksFile+'_'+featuresFile+'.p','rb'))
    train_loaders = dataState['train_loaders']
    test_loaders = dataState['test_loaders']
    dataList = dataState['dataList']

    configD = pkl.load(open(modelConfigFile, 'rb'))
    mName = configD['modelName']
    mParams = configD['params']
    epochs = configD['epochs']
    learningRate = configD['lRate']

    ### CV Models (Just plots the average for now)
    if mName == 'Linear NN'
    print("\n\n"+mName,epochs)
    models = []
    for i in range(len(train_loaders)):
        #There was some debate online about if deepcopy works, so let's just make new ones
        #dataList[0] just shows the model the type of data shape it's looking at
        models.append(LinearNN(dataList[0],mParams))

    elif mName == 'Simple GCN'
    print("\n\n"+mName,epochs)
    models = []
    for i in range(len(train_loaders)):
        models.append(SimpleGCN(dataList[0],mParams))

    elif mName == 'Deep GCN'
    print("\n\n"+mName,epochs)
    models = []
    for i in range(len(train_loaders)):
        models.append(DeepGCN(dataList[0],mParams))

    elif mName == 'GATCONV'
    print("\n\n"+mName,epochs)
    models = []
    for i in range(len(train_loaders)):
        models.append(GATCONV(dataList[0],mParams))

    elif mName == 'PANCONV'
    print("\n\n"+mName,epochs)
    models = []
    for i in range(len(train_loaders)):
        models.append(GATCONV(dataList[0],mParams))

    elif mName == 'GIN2'
    print("\n\n"+mName,epochs)
    models = []
    for i in range(len(train_loaders)):
        models.append(GIN2(dataList[0],mParams))

    else:
        print('Invalid Model!')
    perfDF = perfDF.append(evalModelCV(models, train_loaders, test_loaders, mName,epochs))
    perfDF = perfDF.reset_index()
    perfDF.to_pickle(modelConfigFile+"_performance.p")
    return

    sns.lineplot(data=perfDF, x='Epoch', y='Accuracy', style='Data', hue='Model')
    plt.show()
    sns.relplot(data=perfDF, x='Epoch', y='Accuracy', hue='Data', col='Model', units='Fold', estimator=None, kind='line')
    plt.show()
    perfDF = perfDF.sort_values(by='Accuracy', ascending=False)[perfDF.Epoch >= epochs]
    perfDF = perfDF[perfDF.Data == "Testing Set"]
    print(perfDF)
    return

# Function from tutorial notebook 2
# Modified argument name, added legend (hard-coded for inviable class label), add PCA init and seed, perplexity
# https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html gives the default perplexity of 30.0
def visualize(embedding, color, tsne_perplexity=30.0):
    z = TSNE(n_components=2, init='pca', random_state=seed, perplexity=tsne_perplexity).fit_transform(embedding.detach().cpu().numpy())

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    # Add legend using https://stackoverflow.com/a/58516451
    scatter = plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2", norm=matplotlib.colors.Normalize(vmin=0,vmax=5))
    plt.legend(handles=scatter.legend_elements()[0], labels=locList)
    plt.show()

def train(loader, model, optimizer, criterion):
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    for data in loader:  # Iterate in batches over the training dataset.
         out,e = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
         loss = criterion(out, data.y)  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.


def test(loader, model):
     model.eval()

     correct = 0
     total = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         out,e = model(data.x, data.edge_index, data.batch)
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
         total += len(data.y)
     return correct / total  # Derive ratio of correct predictions.

def getEmbedding(loader, model):
    model.eval()
    embeddingAll = None
    yAll = None
    for data in loader:  # Iterate in batches over the training/test dataset.
         out,e = model(data.x, data.edge_index, data.batch)
         if (embeddingAll == None):
            embeddingAll = e
            yAll = data.y
         else:
            embeddingAll = torch.cat((embeddingAll,e), dim=0)
            yAll = torch.cat((yAll,data.y), dim=0)
    return embeddingAll, yAll

def evalModelCV(models, train_loaders, test_loaders, mName, epochs = 1000, lr=0.001):
    optimizers = []
    criteria = []
    for i in range(len(train_loaders)):
        optimizers.append(torch.optim.Adam(models[i].parameters(), lr=0.001))
        criteria.append(torch.nn.CrossEntropyLoss())

    train_acc_list = []
    test_acc_list = []
    perfDict = dict()
    perfDict["Epoch"] = []
    perfDict["Accuracy"] = []
    perfDict["Data"] = []
    perfDict["Fold"] = []
    perfDict["Model"] = []
    previous_time = time.time()

    for epoch in range(1, epochs+1):
        train_acc_total = 0.0
        test_acc_total = 0.0
        for i in range(len(train_loaders)):
            train(train_loaders[i], models[i], optimizers[i], criteria[i])
            train_acc = test(train_loaders[i], models[i])
            test_acc = test(test_loaders[i], models[i])
            train_acc_total += train_acc
            test_acc_total += test_acc
            if epoch % 25 == 0:
                perfDict["Epoch"].append(epoch)
                perfDict["Accuracy"].append(train_acc)
                perfDict["Fold"].append(i)
                perfDict["Data"].append("Training Set")
                perfDict["Model"].append(mName)

                perfDict["Epoch"].append(epoch)
                perfDict["Accuracy"].append(test_acc)
                perfDict["Fold"].append(i)
                perfDict["Data"].append("Testing Set")
                perfDict["Model"].append(mName)
        train_acc_list.append(train_acc_total/len(train_loaders))
        test_acc_list.append(test_acc_total/len(train_loaders))

        curr_time = time.time()
        if curr_time - previous_time > 2:
            print(f'Epoch: {epoch:03d}, Train Acc: {train_acc_list[-1]:.4f}, Test Acc: {test_acc_list[-1]:.4f}')
            previous_time = time.time()
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc_list[-1]:.4f}, Test Acc: {test_acc_list[-1]:.4f}')
    perfDF = pd.DataFrame.from_dict(perfDict)
    #sns.lineplot(data=perfDF, x='Epoch', y='Accuracy', units='Fold', hue='Data', estimator=None)
    #plt.title(mName,epochs)
    #plt.show()
    return perfDF

if __name__ == "__main__":
    networksFile = argv[1]
    featuresFile = argv[2]
    modelConfigFile = argv[3]

    #networksFile = 'allDevReactomePathsCom.txt'
    networksFile = 'allDevReactomePaths.txt'
    featuresFile = '../scripts/exploratoryScripts/comPPINodes.tsv'
    #featuresFile = '../data/uniprotKeywords/mergedKeyWords_5.tsv'
    testCNNs(networksFile, featuresFile, modelConfigFile)


