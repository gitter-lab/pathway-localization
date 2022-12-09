#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import networkx as nx
import numpy as np
from torch_geometric.utils.convert import from_networkx
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
from models import *
from sys import argv

seed = 24
torch.manual_seed(seed)
np.random.seed(seed) #TODO numpy now reccomends using generators to create random states

def getCNNData(networks_file, features_file, out_file):
    allPathNets, allPathDFs, featuresDF, locList, locDict, pOrder = loc_data_to_tables(networks_file, features_file)

    #So I'll want a graph object for each pathway as a networkx graph
    dataList = []
    nameMap = dict()
    for p in pOrder:
        nameMap[p] = len(dataList)
        graphData = from_networkx(allPathNets[p], group_node_attrs=all, group_edge_attrs=['loc_feat'])
        graphData['y'] = graphData.edge_attr.squeeze(1)
        graphData.num_classes = len(locList)
        dataList.append(graphData)

    # ### Train-Test Split With Mini-Batching
    nFolds = 5
    kf = KFold(n_splits = nFolds)

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
    print("Saving ",out_file)
    torch.save(dataState, out_file)
    return

def loc_data_to_tables(networks_file, features_file):
    #Load in the reactome networks as a dictionary of dataFrames

    #We will not add pathways with fewer nodes than this
    MIN_NETWORK_SIZE_THRESHOLD = 4

    allPathDFs = dict()
    allPathNets = dict()

    locList = ["cytosol","extracellular","membrane","nucleus","secretory-pathway","mitochondrion"]
    locDict = {"cytosol":0,"extracellular":1,"membrane":2,"mitochondrion":5,"nucleus":3,"secretory-pathway":4}

    colNames = ["Interactor1", "Edge Type", "Interactor2", "Location"]
    for line in open(networks_file, "r"):
        pName = line.strip().split("/")[-1]
        pathDF = pd.read_csv(line.strip(), sep="\t", header=None, names=colNames, dtype=str)
        pathDF["loc_feat"] = pathDF['Location'].map(locDict)
        if len(pathDF) >= MIN_NETWORK_SIZE_THRESHOLD:
            allPathDFs[pName] = pathDF
            allPathNets[pName] = nx.from_pandas_edgelist(pathDF, source='Interactor1',
                                                         target='Interactor2', edge_attr= ['Edge Type','Location','loc_feat'])
    print("Loaded in %d pathways" %len(allPathDFs))


    #Load in all comPPI Data as a dataframe too
    featuresDF = pd.read_csv(features_file, sep="\t", index_col="uniprot")
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

    #Make sure we have a canonical ordering for all models
    pOrder = list(allPathNets.keys())
    np.random.shuffle(pOrder)

    return allPathNets, allPathDFs, featuresDF, locList, locDict, pOrder

if __name__ == "__main__":
    networks_file = argv[1]
    features_file = argv[2]
    outFile = argv[3]
    getCNNData(networks_file, features_file, outFile)


