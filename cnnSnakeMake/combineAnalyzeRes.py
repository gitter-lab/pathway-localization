#!/usr/bin/env python
# coding: utf-8
import pandas as pd
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
from sklearn.model_selection import KFold
from models import *
from sys import argv
import pickle as pkl
import os.path
from scipy.stats import sem
from ax.service.ax_client import AxClient
import multiprocessing
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score,matthews_corrcoef,f1_score,balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

seed = 24
torch.manual_seed(seed)
np.random.seed(seed)

def evalCNNs(inFile):

    resultsData = torch.load(inFile)
    parameterization = resultsData['parameters']
    dataFile = parameterization["dataFile"]

    mName = parameterization["mName"]
    epochs = parameterization["epochs"]
    learningRate = parameterization["lRate"]

    dataState = torch.load(dataFile)
    train_loaders = dataState['train_loaders']
    test_loaders = dataState['test_loaders']
    dataList = dataState['dataList']

    if mName == 'LinearNN':
        models = []
        for i in range(len(train_loaders)):
            #There was some debate online about if deepcopy works, so let's just make new ones
            #dataList[0] just shows the model the type of data shape it's looking at
            models.append(LinearNN(dataList[0],parameterization))

    elif mName == 'SimpleGCN':
        models = []
        for i in range(len(train_loaders)):
            models.append(SimpleGCN(dataList[0],parameterization))

    elif mName == 'GATCONV':
        models = []
        for i in range(len(train_loaders)):
            models.append(GATCONV(dataList[0],parameterization))

    elif mName == 'GIN2':
        models = []
        for i in range(len(train_loaders)):
            models.append(GIN2(dataList[0],parameterization))
    else:
        print('Invalid Model!')
        return
    predList,yList = evalModels(models, train_loaders, test_loaders, mName, parameterization, resultsData,learningRate)
    metrics, mergedMetrics = getMetricLists(predList, yList)
    num_inst = len(yList)
    metrics['model'] = [mName]*num_inst

    dataFList = dataFile.split('-')
    pathwaySet = dataFList[0].split('/')[-1]
    features = dataFList[1][:-2] #get rid of '.p'
    metrics['data'] = [pathwaySet]*num_inst
    metrics['features'] = [features]*num_inst

    mergedMetrics['model'] = [mName]
    mergedMetrics['data'] = [pathwaySet]
    mergedMetrics['features'] = [features]

    return metrics,mergedMetrics

def evalPGM(inFile):
    locList = ["cytosol","extracellular","membrane","nucleus","secretory-pathway","mitochondrion"]
    locDict = {"cytosol":0,"extracellular":1,"membrane":2,"mitochondrion":5,"nucleus":3,"secretory-pathway":4}
    colNames = ["Interactor1", "Edge Type", "Interactor2", "Location"]

    resultsData = torch.load(inFile)
    inF = resultsData['inF']
    mName = resultsData['mName']
    netF = resultsData['netF']

    allPathDF = dict()
    for line in open(netF, "r"):
        pName = line.strip().split("/")[-1]
        pathDF = pd.read_csv(line.strip(), sep="\t", header=None, names=colNames, dtype=str)
        pathDF["edgeStr"] = pathDF["Interactor1"] + "_" + pathDF["Interactor2"]
        allPathDF[pName] = pathDF
    allPredDict = dict()
    allYDict = dict()
    curPred = []
    curY=[]
    pName = ""
    for line in open(inF):
        lineList = line.strip().split()
        if len(lineList)<3:
            #New pathway
            if len(curPred)>0:
                allPredDict[pName] = curPred
                allYDict[pName] = curY
            pName = line.strip().split("/")[-1]
            pName = "_".join(pName.split("_")[1:])
            curPred = []
            curY = []
        else:
            i1 = lineList[0]
            i2 = lineList[1]
            pred = lineList[2]
            edgeStr = i1+"_"+i2
            y = allPathDF[pName][allPathDF[pName]['edgeStr']==edgeStr].iloc[0]["Location"]
            curPred.append(locDict[pred])
            curY.append(locDict[y])
    yList = []
    predList = []
    for p in allYDict:
        yList.append(allYDict[p])
        predList.append(allPredDict[p])
    metrics, mergedMetrics = getMetricLists(predList, yList)

    num_inst = len(yList)
    metrics['model'] = [mName]*num_inst

    dataFList = inFile.split('-')
    pathwaySet = dataFList[1]
    features = dataFList[2][:-2] #get rid of '.p'
    metrics['data'] = [pathwaySet]*num_inst
    metrics['features'] = [features]*num_inst

    mergedMetrics['model'] = [mName]
    mergedMetrics['data'] = [pathwaySet]
    mergedMetrics['features'] = [features]
    return metrics, mergedMetrics

def evalSKModel(inFile):
    locList = ["cytosol","extracellular","membrane","nucleus","secretory-pathway","mitochondrion"]
    locDict = {"cytosol":0,"extracellular":1,"membrane":2,"mitochondrion":5,"nucleus":3,"secretory-pathway":4}
    colNames = ["Interactor1", "Edge Type", "Interactor2", "Location"]

    resultsData = torch.load(inFile)
    mName = resultsData['model']
    preds = resultsData['predictions']
    yAll = resultsData['y_all']
    netInd = resultsData['network_index']

    predList = []
    yList = []
    for net in netInd:
        curY = yAll[netInd[net]]
        curPred = preds[netInd[net]]
        yList.append(curY)
        if len(np.unique(curY))==6:
            print(net)
        predList.append(curPred)

    metrics, mergedMetrics = getMetricLists(predList, yList)

    num_inst = len(yList)
    metrics['model'] = [mName]*num_inst

    dataFList = inFile.split('-')
    pathwaySet = dataFList[1]
    features = dataFList[2][:-2] #get rid of '.p'
    metrics['data'] = [pathwaySet]*num_inst
    metrics['features'] = [features]*num_inst

    mergedMetrics['model'] = [mName]
    mergedMetrics['data'] = [pathwaySet]
    mergedMetrics['features'] = [features]
    return metrics, mergedMetrics

def getOuts(loader, model):
     model.eval()
     correct = 0
     total = 0
     predList = []
     yList = []
     for data in loader:
         out,e = model(data.x, data.edge_index, data.batch)
         pred = out.argmax(dim=1)  # Use the class with highest probability.

         maxBatch = torch.max(data.batch)
         for i in range(maxBatch+1):
            newT = torch.flatten(torch.index_select(data.edge_index,0,torch.as_tensor([0])))
            edgeBatch = torch.index_select(data.batch,0,newT)
            predB = torch.masked_select(pred, (edgeBatch==i))
            yB = torch.masked_select(data.y, (edgeBatch==i))
            predList.append(predB)
            yList.append(yB)
     return predList,yList

def getMetricLists(predList,yList):
    mccList = []
    accList = []
    f1List = []
    baccList = []
    sizeList = []
    locCount = []
    predCount = []
    totalPred = None
    totalY = None
    for i in range(len(predList)):
        try:
            #If this is a tensor convert it
            pred = predList[i].numpy()
            y = yList[i].numpy()
        except AttributeError:
            pred = predList[i]
            y = yList[i]
        if totalPred is None:
            totalPred = pred
            totalY = y
        else:
            totalPred = np.concatenate((totalPred, pred))
            totalY = np.concatenate((totalY, y))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)
            mcc = matthews_corrcoef(y,pred)
            f1 = f1_score(y, pred, average='micro')
            acc = accuracy_score(y, pred)
            bacc = balanced_accuracy_score(y, pred)
            mccList.append(mcc)
            f1List.append(f1)
            accList.append(acc)
            baccList.append(bacc)
            sizeList.append(len(y))
            locCount.append(len(np.unique(y)))
            predCount.append(len(np.unique(pred)))

    netOut = {'mcc':mccList, 'f1':f1List, 'acc':accList, 'bal_acc':baccList, 'p_size':sizeList, 'Unique Localizations': locCount, 'Predicted Unique Localizations':predCount}
    allOut = {'mcc':[matthews_corrcoef(totalY, totalPred)],
              'f1':[f1_score(totalY, totalPred, average='micro')],
              'acc':[accuracy_score(totalY, totalPred)],
              'bal_acc':[balanced_accuracy_score(totalY, totalPred)],
              'p_size':[len(totalY)],
              'Unique Localizations':[len(np.unique(totalY))],
              'Predicted Unique Localizations':[len(np.unique(totalPred))]}
    return netOut, allOut


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

def evalModels(models, train_loaders, test_loaders, mName, parameters, resultsData, lr):
    #optimizers = []
    losses = resultsData['losses']
    for i in range(len(train_loaders)):
        #optimizers.append(torch.optim.Adam(models[i].parameters(), lr=lr))
        #optimizers[i].load_state_dict(resultsData['optimizer_states'][i])
        model_states = resultsData['model_states']

        models[i].load_state_dict(model_states[i])
    yList = []
    predList = []
    for i in range(len(train_loaders)):
        preds, ys = getOuts(test_loaders[i], models[i])
        predList += preds
        yList += ys
    return predList,yList

if __name__ == "__main__":
    fList = []
    for f in argv[1:]:
        fList.append(f)
    perfs = dict()
    perfsMerged = dict()
    for f in fList:
        print("Loading "+f+"...")
        metrics = None
        mergedMetrics = None
        if f[:3]=="pgm":
            metrics, mergedMetrics = evalPGM(f)
        elif f[:3]=="sk_":
            metrics, mergedMetrics = evalSKModel(f)
        else:
            metrics, mergedMetrics = evalCNNs(f)
        numInst = 0
        for m in metrics:
            if not m in perfs:
                perfs[m] = []
                perfsMerged[m] = []
            perfs[m] += metrics[m]
            perfsMerged[m] += mergedMetrics[m]
            numInst = len(metrics[m])
    metricDF = pd.DataFrame.from_dict(perfs)
    metricMergedDF = pd.DataFrame.from_dict(perfsMerged)

    torch.save({'metrics':metricDF, 'mergedMetrics':metricMergedDF},'results/allRes.p')





