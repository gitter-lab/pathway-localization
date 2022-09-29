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
    predList,yList,missingList = evalModels(models, train_loaders, test_loaders, mName, parameterization, resultsData,learningRate)
    metrics, mergedMetrics = getMetricLists(predList, yList, missingList)
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

    dataFList = inFile.split('-')
    pathwaySet = dataFList[1]
    features = dataFList[2][:-2] #get rid of '.p'
    featuresFName = 'data/'+features+'.tsv'
    featuresDF = pd.read_csv(featuresFName, sep="\t", index_col="uniprot")
    featuresDict = featuresDF.to_dict('index')

    allPathDF = dict()
    for line in open(netF, "r"):
        pName = line.strip().split("/")[-1]
        pathDF = pd.read_csv(line.strip(), sep="\t", header=None, names=colNames, dtype=str)
        pathDF["edgeStr"] = pathDF["Interactor1"] + "_" + pathDF["Interactor2"]
        allPathDF[pName] = pathDF
    allPredDict = dict()
    allYDict = dict()
    allMisses = dict()
    curPred = []
    curY=[]
    curMiss = []
    pName = ""
    for line in open(inF):
        lineList = line.strip().split()
        if len(lineList)<3:
            #New pathway
            if len(curPred)>0:
                allPredDict[pName] = curPred
                allYDict[pName] = curY
                allMisses[pName] = curMiss
            pName = line.strip().split("/")[-1]
            pName = "_".join(pName.split("_")[1:])
            curPred = []
            curY = []
            curMiss = []
        else:
            i1 = lineList[0]
            i2 = lineList[1]
            numMiss = 0
            if i1 not in featuresDict:
                numMiss+=1
            if i2 not in featuresDict:
                numMiss+=1
            curMiss.append(numMiss)
            pred = lineList[2]
            edgeStr = i1+"_"+i2
            y = allPathDF[pName][allPathDF[pName]['edgeStr']==edgeStr].iloc[0]["Location"]
            curPred.append(locDict[pred])
            curY.append(locDict[y])
    yList = []
    predList = []
    missList = []
    for p in allYDict:
        yList.append(allYDict[p])
        predList.append(allPredDict[p])
        missList.append(np.array(allMisses[p]))
    metrics, mergedMetrics = getMetricLists(predList, yList, missList)

    num_inst = len(yList)
    metrics['model'] = [mName]*num_inst

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
    xAll = resultsData['x_all']

    x_1 = xAll[:,:6]
    x_2 = xAll[:,6:]
    var_x1 = np.var(x_1, axis=1)
    var_x2 = np.var(x_2, axis=1)
    singleMiss = (var_x1==0) | (var_x2==0)
    doubleMiss = (var_x1==0) & (var_x2==0)
    missingList = singleMiss.astype(int) + doubleMiss.astype(int)

    netInd = resultsData['network_index']

    predList = []
    yList = []
    missList = []
    for net in netInd:
        curY = yAll[netInd[net]]
        curPred = preds[netInd[net]]
        curMiss = missingList[netInd[net]]
        yList.append(curY)
        predList.append(curPred)
        missList.append(curMiss)
    metrics, mergedMetrics = getMetricLists(predList, yList, missList)

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
     missingList = []
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

            #Find missing data by looking for variance of 0
            x1 = data.x[data.edge_index[0]]
            x2 = data.x[data.edge_index[1]]
            x1_b = x1[(edgeBatch==i),:]
            x2_b = x2[(edgeBatch==i),:]
            var_x1 = torch.var(x1_b, 1, False)
            var_x2 = torch.var(x2_b, 1, False)
            oneMissing = (var_x1==0) | (var_x2==0)
            allMissing = (var_x1==0) & (var_x2==0)
            missVec = (oneMissing.type(torch.int) + allMissing.type(torch.int)).numpy()
            missingList.append(missVec)

     return predList,yList,missingList

def getMetricLists(predList,yList,missingList=None):
    mccList = []
    accList = []
    f1List = []
    baccList = []
    sizeList = []
    locCount = []
    predCount = []
    missList = []
    totalPred = None
    totalY = None
    totalMiss = None
    totalSize = None
    totalPredCount = None
    totalLocCount = None
    for i in range(len(predList)):
        try:
            pred = predList[i].numpy()
            y = yList[i].numpy()
        except AttributeError:
            pred = predList[i]
            y = yList[i]
        if totalPred is None:
            totalPred = pred
            totalY = y
            totalSize = [len(y)]*len(y)
            totalMiss = missingList[i]
            totalLocCount = [len(np.unique(y))]*len(y)
            totalPredCount = [len(np.unique(pred))]*len(y)
        else:
            totalPred = np.concatenate((totalPred, pred))
            totalY = np.concatenate((totalY, y))
            totalSize = np.concatenate((totalSize, [len(y)]*len(y)))
            totalMiss = np.concatenate((totalMiss, missingList[i]))
            totalLocCount = np.concatenate((totalLocCount, [len(np.unique(y))]*len(y)))
            totalPredCount = np.concatenate((totalPredCount, [len(np.unique(pred))]*len(y)))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)
            mcc = matthews_corrcoef(y,pred)
            f1 = f1_score(y, pred, average='macro')
            acc = accuracy_score(y, pred)
            bacc = balanced_accuracy_score(y, pred)
            frac_missing = len(missingList[i][missingList[i]==2])/float(len(missingList[i]))
            mccList.append(mcc)
            f1List.append(f1)
            accList.append(acc)
            baccList.append(bacc)
            sizeList.append(len(y))
            locCount.append(len(np.unique(y)))
            predCount.append(len(np.unique(pred)))
            missList.append(frac_missing)
    #Main result, performance stratified by network
    netOut = {'mcc':mccList, 'f1':f1List, 'acc':accList, 'bal_acc':baccList, 'p_size':sizeList, 'Unique Localizations': locCount, 'Predicted Unique Localizations':predCount,'missing_features':missList}

    #Also calculate all edges merged together to make sure tiny networks aren't dominating results
    allOut = {'mcc':[matthews_corrcoef(totalY, totalPred)],
              'f1':[f1_score(totalY, totalPred, average='macro')],
              'acc':[accuracy_score(totalY, totalPred)],
              'bal_acc':[balanced_accuracy_score(totalY, totalPred)],
              'p_size':[totalSize],
              'missing_features':[totalMiss],
              'allY':[totalY],
              'allPred':[totalPred],
              'Unique Localizations':[totalLocCount],
              'Predicted Unique Localizations':[totalPredCount]}
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
    losses = resultsData['losses']
    for i in range(len(train_loaders)):
        model_states = resultsData['model_states']

        models[i].load_state_dict(model_states[i])
    yList = []
    predList = []
    missingList = []
    for i in range(len(train_loaders)):
        preds, ys, missList = getOuts(test_loaders[i], models[i])
        predList += preds
        yList += ys
        missingList += missList
    return predList,yList,missingList

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
        for m in metrics:
            if not m in perfs:
                perfs[m] = []
                perfsMerged[m] = []
            perfs[m] += metrics[m]
            perfsMerged[m] += mergedMetrics[m]
        for m in ['allY','allPred']:
            if not m in perfsMerged:
                perfsMerged[m] = []
            perfsMerged[m] += mergedMetrics[m]
    metricDF = pd.DataFrame.from_dict(perfs)
    metricMergedDF = pd.DataFrame.from_dict(perfsMerged)

    torch.save({'metrics':metricDF, 'mergedMetrics':metricMergedDF},'results/allRes.p')





