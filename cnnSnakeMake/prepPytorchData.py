#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
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
import os.path

seed = 24
torch.manual_seed(seed)
np.random.seed(seed)

def getCNNData(networksFile, featuresFile, outFile):
    #Load in the reactome networks as a dictionary of dataFrames

    #We will not add pathways with fewer nodes than this
    MIN_NETWORK_SIZE_THRESHOLD = 4
    EVIDENCE_THRESHOLD_LOWER = 0.0
    EVIDENCE_THRESHOLD_UPPER = 9999

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
    featuresDF = featuresDF.where(featuresDF > EVIDENCE_THRESHOLD_LOWER, 0)
    featuresDF = featuresDF.where(featuresDF < EVIDENCE_THRESHOLD_UPPER, 0)
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

        train_loader = DataLoader(trains, batch_size=512, shuffle=False)
        test_loader = DataLoader(tests, batch_size=512, shuffle=False)

        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    dataState = dict()
    dataState['train_loaders'] = train_loaders
    dataState['test_loaders'] = test_loaders
    dataState['dataList'] = dataList
    print("Saving ",outFile)
    torch.save(dataState, outFile)
    return

if __name__ == "__main__":
    networksFile = argv[1]
    featuresFile = argv[2]
    outFile = argv[3]

    #networksFile = 'allDevReactomePathsCom.txt'
    #networksFile = 'allDevReactomePaths.txt'
    #featuresFile = '../scripts/exploratoryScripts/comPPINodes.tsv'
    #featuresFile = '../data/uniprotKeywords/mergedKeyWords_5.tsv'
    getCNNData(networksFile, featuresFile, outFile)


