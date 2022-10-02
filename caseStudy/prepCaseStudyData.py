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

def getCaseStudyData(networks_file, features_file, out_file, markers_file, name_map_file, all_folds, pred_data):
    allPathNets, allPathDFs, featuresDF, locDict, pOrder = pcsf_paths_to_tables(networks_file, features_file, markers_file, name_map_file, pred_data, all_folds)

    #So I'll want a graph object for each pathway as a networkx graph
    dataList = []
    nameMap = dict()
    for p in pOrder:
        nameMap[p] = len(dataList)
        graphData = from_networkx(allPathNets[p], group_node_attrs=all, group_edge_attrs=['loc_feat','is_marker','is_pred'])
        graphData['y'] = graphData.edge_attr[:,0].long()
        graphData['is_marker'] = graphData.edge_attr[:,1].gt(0)
        graphData['is_pred'] = graphData.edge_attr[:,2].gt(0)
        graphData['name'] = p
        graphData.num_classes = len(locDict)
        dataList.append(graphData)

    #Train-Test Split With Mini-Batching
    nFolds = 5
    kf = KFold(n_splits = nFolds)

    train_loaders = []
    test_loaders = []

    #Used when we skip training and use a pretrained model
    if all_folds=="all":
        trains = []
        tests = []
        for i in range(len(dataList)):
            trains.append(dataList[i])
            tests.append(dataList[i])

        train_loader = DataLoader(trains, batch_size=512, shuffle=False)
        test_loader = DataLoader(tests, batch_size=512, shuffle=False)

        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    #Here we hardcoded the dataset indices. This is not great, but it works.
    elif 'egf' in pred_data:
        trains = []
        tests = []
        for i in range(453):
            trains.append(dataList[i])
        for i in range(453,len(dataList)):
            tests.append(dataList[i])
        train_loader = DataLoader(trains, batch_size=512, shuffle=False)
        test_loader = DataLoader(tests, batch_size=512, shuffle=False)

        train_loaders.append(train_loader)
        test_loaders.append(test_loader)
    elif 'same' in pred_data:
        trains = []
        tests = []
        for i in range(492):
            trains.append(dataList[i])
        for i in range(492,len(dataList)):
            tests.append(dataList[i])
        train_loader = DataLoader(trains, batch_size=512, shuffle=False)
        test_loader = DataLoader(tests, batch_size=512, shuffle=False)

        train_loaders.append(train_loader)
        test_loaders.append(test_loader)
    elif all_folds=="half":
        trains = []
        tests = []
        for i in range(int(len(dataList)/2)):
            trains.append(dataList[i])
        for i in range(int(len(dataList)/2),len(dataList)):
            tests.append(dataList[i])
        train_loader = DataLoader(trains, batch_size=512, shuffle=False)
        test_loader = DataLoader(tests, batch_size=512, shuffle=False)

        train_loaders.append(train_loader)
        test_loaders.append(test_loader)
    elif all_folds=="folds":
        for tr_ind, te_ind in kf.split(dataList):
            trains = []
            tests = []
            for ind in tr_ind:
                trains.append(dataList[ind])
            for ind in te_ind:
                tests.append(dataList[ind])

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

def pcsf_paths_to_tables(networks_file, features_file, markers_file, name_map_file, pred_data, all_folds):
    #Load in the reactome networks as a dictionary of dataFrames

    #We will not add pathways with fewer nodes than this
    MIN_NETWORK_SIZE_THRESHOLD = 4

    allPathDFs = dict()
    allPathNets = dict()

    locList = ["cytosol","extracellular","membrane","nucleus","secretory-pathway","mitochondrion"]
    #Misses will be -1
    locDict = {"cytosol":0,"extracellular":1,"membrane":2,"mitochondrion":5,"nucleus":3,"secretory-pathway":4,"none":-1}

    colNames = ["Interactor1", "Interactor2", "Included"]
    nameMap = dict()
    for line in open(name_map_file,'r'):
        lineList = line.strip().split('\t')
        nameMap[lineList[0]] = lineList[1].split('_')[0]

    markerDict = dict()
    for line in open(markers_file, "r"):
        lineList=line.strip().split(',')
        markerDict[lineList[0]] = lineList[2]

    predDataDict = dict()
    for line in open(pred_data, "r"):
        lineList=line.strip().split(',')
        if lineList[1]=='NA':
            continue
        predDataDict[lineList[0]] = lineList[1]

    predDataDictTMT = dict()
    if 'egf' in pred_data or 'same' in pred_data:
        pred_dataTMT = 'data/tmtLocs120hpi.csv'
        for line in open(pred_dataTMT, "r"):
            lineList=line.strip().split(',')
            if lineList[1]=='NA':
                continue
            predDataDictTMT[lineList[0]] = lineList[1]

    pathsOrder = []
    for line in open(networks_file, "r"):
        pName = line.strip().split("/")[-1]
        pathDF = pd.read_csv(line.strip(), sep=" ", header=None, names=colNames, dtype=str)
        if len(pathDF) >= MIN_NETWORK_SIZE_THRESHOLD:
            allPathDFs[pName] = pathDF
            pathsOrder.append(pName)

    totalE = 0
    misses = 0
    totalPred = 0
    for p in allPathDFs:
        locs = []
        isMarker = []
        isPred = []
        eNameInd = []
        eNameNames = []
        pathDF = allPathDFs[p]
        for index,row in pathDF.iterrows():
            loc = ""
            i1 = row["Interactor1"]
            i2 = row["Interactor2"]
            eName = "_".join(sorted([i1,i2]))
            eNameNames.append(eName)
            eNameInd.append(len(eNameNames)-1)
            if (i1 in markerDict) and (i2 in markerDict):
                loc1 = markerDict[i1]
                loc2 = markerDict[i2]
                totalE += 1
                if loc1 != loc2:
                    misses += 1
                loc = loc1
            elif i1 in markerDict:
                totalE += 1
                loc = markerDict[i1]
            elif i2 in markerDict:
                totalE += 1
                loc = markerDict[i2]
            else:
                loc = "none"
            locs.append(loc)
            if loc != "none":
                isMarker.append(True)
            else:
                #Marker no longer does anything since we just use pretrained models for the timeseries data
                isMarker.append(True)
            if (loc=='none'):
                if 'egf' in p or '24' in p:
                    curPredData = predDataDict
                else:
                    curPredData = predDataDictTMT
                if (i1 in curPredData) and (i2 in curPredData):
                    loc1 = curPredData[i1]
                    loc2 = curPredData[i2]
                    loc = loc1
                elif i1 in curPredData:
                    loc = curPredData[i1]
                elif i2 in curPredData:
                    loc = curPredData[i2]
                else:
                    loc = "none"
                locs[-1] = loc
                if loc != "none":
                    isPred.append(True)
                    totalPred += 1
                else:
                    isPred.append(False)
            else:
                isPred.append(False)
        pathDF['Location'] = locs
        pathDF["Interactor1"] = pathDF["Interactor1"].map(nameMap)
        pathDF["Interactor2"] = pathDF["Interactor2"].map(nameMap)
        pathDF["loc_feat"] = pathDF['Location'].map(locDict)
        pathDF['is_marker'] = isMarker
        pathDF['is_pred'] = isPred
        pathDF['e_name'] = eNameNames
        allPathNets[p] = nx.from_pandas_edgelist(pathDF, source='Interactor1',
                                                 target='Interactor2', edge_attr= ['Location','e_name','loc_feat','is_marker','is_pred'])

    print("Loaded in %d pathways with %f percent misses and %d predictions % edges" %(len(allPathDFs), 100*float(misses)/totalE, totalPred, totalE))


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
    if all_folds=='half':
        pOrder = pathsOrder
    else:
        pOrder = list(allPathNets.keys())
        np.random.shuffle(pOrder)

    return allPathNets, allPathDFs, featuresDF, locDict, pOrder

if __name__ == "__main__":
    networks_file = argv[1]
    features_file = argv[2]
    outFile = argv[3]
    markers_file = argv[4]
    name_map_file = argv[5]
    all_folds = argv[6]
    pred_data = argv[7]

    getCaseStudyData(networks_file, features_file, outFile, markers_file, name_map_file, all_folds, pred_data)


