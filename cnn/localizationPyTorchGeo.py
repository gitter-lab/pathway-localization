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

seed = 2
torch.manual_seed(seed)
np.random.seed(seed)

def testCNNs(networksFile, featuresFile):
    #Load in the reactome networks as a dictionary of dataFrames

    #We will not add pathways with fewer nodes than this
    MIN_NETWORK_SIZE_THRESHOLD = 4
    allPathDFs = dict()
    allPathNets = dict()

    locList = ["cytosol","extracellular","membrane","nucleus","secretory-pathway","mitochondrion"]
    locDict = {"cytosol":0,"extracellular":1,"membrane":2,"mitochondrion":5,"nucleus":3,"secretory-pathway":4}

    colNames = ["Interactor1", "Edge Type", "Interactor2", "Location"]
    for line in open(networksFile, "r"):
        pName = line.strip().split("/")[-1]
        pathDF = pd.read_csv(line.strip(), sep="\t", header=None, names=colNames, dtype=str)
        pathDF["loc_feat"] = pathDF['Location'].map(locDict)
        if len(pathDF) >= MIN_NETWORK_SIZE_THRESHOLD:
            allPathDFs[pName] = pathDF
            allPathNets[pName] = nx.from_pandas_edgelist(pathDF, source='Interactor1',
                                                         target='Interactor2', edge_attr= ['Edge Type','Location','loc_feat'])
    print("Loaded in %d pathways" %len(allPathDFs))


    #Load in all comPPI Data as a dataframe too
    featuresDF = pd.read_csv(featuresFile, sep="\t", index_col="uniprot")
    featuresDict = featuresDF.to_dict('index')

    #Make a uniform attribute row for misses
    uniform = dict()
    for feat in featuresDF.columns:
        uniform[feat] = 1.0/(len(featuresDF.columns))


    #Now merge it to the created networks
    for p in allPathNets:
        net = allPathNets[p]
        #Give missing data a uniform distribution
        for k in net.nodes():
            if k not in featuresDict:
                featuresDict[k] = uniform
        nx.set_node_attributes(net, featuresDict)

    #Look at one of the networks
    testP = allPathNets['R-HSA-375165.txt']
    print(f'graph has {testP.number_of_nodes()} nodes and {testP.number_of_edges()} undirected edges')
    i=1
    print("\n3 Nodes:")
    for k,v in testP.nodes(data=True):
        print(k,v)
        i+=1
        if i>3: break

    print("\n3 Edges:")
    for k1,k2,v in testP.edges(data=True):
        print(k1,k2,v)
        i+=1
        if i>6: break


    #So I'll want a graph object for each pathway as a networkx graph
    dataList = []
    nameMap = dict()
    for p in allPathNets:
        nameMap[p] = len(dataList)
        graphData = from_networkx(allPathNets[p], group_node_attrs=all, group_edge_attrs=['loc_feat'])
        graphData['y'] = graphData.edge_attr.squeeze(1)
        graphData.num_classes = len(locList)
        dataList.append(graphData)


    # ### Train-Test Split With Mini-Batching
    nFolds = 5
    kf = KFold(n_splits = nFolds)
    np.random.shuffle(dataList)

    train_sets = []
    test_sets = []
    train_loaders = []
    test_loaders = []

    for tr_ind, te_ind in kf.split(dataList):
        trains = []
        tests = []
        for ind in tr_ind:
            trains.append(dataList[ind])
        for ind in te_ind:
            tests.append(dataList[ind])

        train_sets.append(trains)
        test_sets.append(tests)

        train_loader = DataLoader(trains, batch_size=64, shuffle=False)
        test_loader = DataLoader(tests, batch_size=64, shuffle=False)

        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    # ### CV Models (Just plots the average for now)
    mName = 'Linear NN'
    print("\n\n"+mName)
    models = []
    for i in range(len(train_loaders)):
        #There was some debate online about if deepcopy works, so let's just make new ones
        models.append(LinearNN(dataList[0]))
    perfDF = evalModelCV(models, train_loaders, test_loaders, mName)


    mName = 'Simple GCN'
    print("\n\n"+mName)
    models = []
    for i in range(len(train_loaders)):
        #There was some debate online about if deepcopy works, so let's just make new ones
        models.append(SimpleGCN(dataList[0]))
    perfDF = perfDF.append(evalModelCV(models, train_loaders, test_loaders, mName))


    mName = 'Deep GCN'
    print("\n\n"+mName)
    models = []
    for i in range(len(train_loaders)):
        #There was some debate online about if deepcopy works, so let's just make new ones
        models.append(DeepGCN(dataList[0]))
    perfDF = perfDF.append(evalModelCV(models, train_loaders, test_loaders, mName))


    mName = 'GATCONV'
    print("\n\n"+mName)
    models = []
    for i in range(len(train_loaders)):
        #There was some debate online about if deepcopy works, so let's just make new ones
        models.append(GATCONV(dataList[0]))
    perfDF = perfDF.append(evalModelCV(models, train_loaders, test_loaders, mName))


    mName = 'GIN2'
    print("\n\n"+mName)
    models = []
    for i in range(len(train_loaders)):
        #There was some debate online about if deepcopy works, so let's just make new ones
        models.append(GIN2(dataList[0]))
    perfDF = perfDF.append(evalModelCV(models, train_loaders, test_loaders, mName))
    perfDF = perfDF.reset_index()
    print(perfDF)
    perfDF.to_pickle("geoPerformance.pkl")
    sns.lineplot(data=perfDF, x='Epoch', y='Accuracy', style='Data', hue='Model')
    plt.show()
    sns.relplot(data=perfDF, x='Epoch', y='Accuracy', hue='Data', col='Model', units='Fold', estimator=None, kind='line')
    plt.show()
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

def evalModelCV(models, train_loaders, test_loaders, mName):
    optimizers = []
    criteria = []
    for i in range(len(train_loaders)):
        optimizers.append(torch.optim.Adam(models[i].parameters(), lr=0.0001))
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

    for epoch in range(1, 2000):
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
    #plt.title(mName)
    #plt.show()
    return perfDF

if __name__ == "__main__":
    #networksFile = 'allDevReactomePathsCom.txt'
    networksFile = 'allDevReactomePaths.txt'
    featuresFile = '../scripts/exploratoryScripts/comPPINodes.tsv'
    #featuresFile = '../data/uniprotKeywords/mergedKeyWords_5.tsv'
    testCNNs(networksFile, featuresFile)


# ### Single Fold Models (old)
#em, yAll = getEmbedding(test_loaders[0], models[0])
#visualize(em, yAll, tsne_perplexity=30.0)
#
#em2, yAll2 = getEmbedding(train_loaders[0], models[0])
#visualize(em2, yAll2, tsne_perplexity=30.0)
#
#em3 = torch.cat((em,em2), dim=0)
#yAll3 = torch.cat((yAll,yAll2), dim=0)
#visualize(em3, yAll3, tsne_perplexity=30.0)
