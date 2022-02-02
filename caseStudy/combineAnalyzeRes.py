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

    ### CV Models (Just plots the average for now)
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

    elif mName == 'PANCONV':
        models = []
        for i in range(len(train_loaders)):
            models.append(PANCONV(dataList[0],parameterization))

    elif mName == 'GIN2':
        models = []
        for i in range(len(train_loaders)):
            models.append(GIN2(dataList[0],parameterization))
    else:
        print('Invalid Model!')
        return
    predList,yList,eNames = evalModels(models, train_loaders, test_loaders, mName, parameterization, resultsData,learningRate)
    metrics, mergedMetrics = getMetricLists(predList, yList,eNames)
    namedDF = matchNamesPreds(predList,yList,eNames)
    num_inst = len(yList)
    metrics['model'] = [mName]*num_inst
    dataFList = dataFile.split('-')
    pathwaySet = dataFList[0].split('/')[-1]+'hpi'
    features = dataFList[1][:-2] #get rid of '.p'
    metrics['Timepoint'] = [pathwaySet]*num_inst
    metrics['features'] = [features]*num_inst

    mergedMetrics['model'] = [mName]
    mergedMetrics['Timepoint'] = [pathwaySet]
    mergedMetrics['features'] = [features]

    return metrics,mergedMetrics,namedDF

def matchNamesPreds(predList,yList,eNames):
    nameAll = []
    yAll = []
    predAll = []
    eDict = dict()
    numMiss = 0
    numTotal = 0
    for i in range(len(predList)):
        try:
            #If this is a tensor convert it
            pred = list(predList[i].numpy())
            y = list(yList[i].numpy())
        except AttributeError:
            pred = list(predList[i])
            y = list(yList[i])
        names = list(eNames[i])
        #print(names)
        nameAll += names
        yAll += y
        predAll += pred
        for j in range(len(names)):
            n = names[j]
            if n in eDict:
                numTotal += 1
                if pred[j] != eDict[n]:
                    numMiss+=1
            eDict[n] = pred[j]
    print(float(numMiss)/numTotal)
    eDF = pd.DataFrame.from_dict({'Name':nameAll,'Predicted Location':predAll,'Location':yAll})
    eDF["Name"] = eDF["Name"].astype(str)
    eDF = eDF.drop_duplicates(subset='Name',ignore_index=True)
    print(eDF.dtypes)
    return eDF




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
    metrics['Timepoint'] = [pathwaySet]*num_inst
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
        predList.append(curPred)

    metrics, mergedMetrics = getMetricLists(predList, yList)

    num_inst = len(yList)
    metrics['model'] = [mName]*num_inst

    dataFList = inFile.split('-')
    pathwaySet = dataFList[1]
    features = dataFList[2][:-2] #get rid of '.p'
    metrics['Timepoint'] = [pathwaySet]*num_inst
    metrics['features'] = [features]*num_inst

    mergedMetrics['model'] = [mName]
    mergedMetrics['Timepoint'] = [pathwaySet]
    mergedMetrics['features'] = [features]
    return metrics, mergedMetrics

def getOuts(loader, model):
     model.eval()
     correct = 0
     total = 0
     predList = []
     yList = []
     eNames = []
     for data in loader:
         out,e = model(data.x, data.edge_index, data.batch)
         pred = out.argmax(dim=1)  # Use the class with highest probability.

         maxBatch = torch.max(data.batch)
         for i in range(maxBatch+1):
            newT = torch.flatten(torch.index_select(data.edge_index,0,torch.as_tensor([0])))
            edgeBatch = torch.index_select(data.batch,0,newT)
            predB = torch.masked_select(pred, torch.logical_and(edgeBatch==i,data.is_pred))
            yB = torch.masked_select(data.y, torch.logical_and(edgeBatch==i,data.is_pred))
            e_name = np.concatenate(data.e_name, axis=0)[torch.logical_and(edgeBatch==i,data.is_pred).numpy()]
            #e_name = torch.masked_select(data.e_name, torch.logical_and(edgeBatch==i,data.is_pred))
            predList.append(predB)
            yList.append(yB)
            eNames.append(e_name)
     return predList,yList,eNames

def getMetricLists(predList,yList,eNames):
    mccList = []
    accList = []
    f1List = []
    baccList = []
    sizeList = []
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

    netOut = {'mcc':mccList, 'f1':f1List, 'Accuracy':accList, 'Balanced Accuracy':baccList, 'p_size':sizeList}
    allOut = {'mcc':[matthews_corrcoef(totalY, totalPred)],
              'f1':[f1_score(totalY, totalPred, average='micro')],
              'Accuracy':[accuracy_score(totalY, totalPred)],
              'Balanced Accuracy':[balanced_accuracy_score(totalY, totalPred)],
              'p_size':[len(totalY)]}
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
    eNames = []
    for i in range(len(train_loaders)):
        preds, ys, e_name = getOuts(test_loaders[i], models[i])
        predList += preds
        yList += ys
        eNames += e_name
    return predList,yList,eNames

def plotMetric(metricDF, metric_name, x_val = None, hue=None, title="", ax=None):
    if "Merged" in title:
        sns.barplot(x=x_val,y=metric_name, hue=hue, data=metricDF, ax=ax)
    else:
        sns.boxplot(x=x_val,y=metric_name, hue=hue, data=metricDF, ax=ax)
    plt.title(title)

if __name__ == "__main__":
    fList = []
    for f in argv[1:]:
        fList.append(f)
    perfs = dict()
    perfsMerged = dict()
    namedList = []
    for f in fList:
        print("Loading "+f+"...")
        metrics = None
        mergedMetrics = None
        if f[:3]=="pgm":
            metrics, mergedMetrics = evalPGM(f)
        elif f[:3]=="sk_":
            metrics, mergedMetrics = evalSKModel(f)
        else:
            metrics, mergedMetrics, namedDF = evalCNNs(f)
            namedList.append(namedDF)
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

    sns.set_theme(style="white", context='paper')
    sns.set(font_scale=1.4)
    sns.set_palette('flare')
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    #plotMetric(metricMergedDF, 'Balanced Accuracy', x_val='model', hue='features')
    #f, (ax1, ax2) = plt.subplots(1,2)
    #plotMetric(metricDF, 'Accuracy', x_val='Timepoint', hue=None, ax=ax1)
    #plotMetric(metricDF, 'Balanced Accuracy', x_val='Timepoint', hue=None, ax=ax2)
    #plt.show()

    #Calculate translocation events
    if len(namedList) != 2:
        print("This is not what I thought")
    joinedDF = namedList[0].merge(namedList[1], on='Name', how='inner',suffixes=('_24','_120'))
    print(len(namedList[1]))
    print(len(namedList[0]))
    print(len(joinedDF))
    joinedDF['locChanged'] = joinedDF['Location_24']!=joinedDF['Location_120']
    joinedDF['pred_locChanged'] = joinedDF['Predicted Location_24']!=joinedDF['Predicted Location_120']
    print("Localization Change")
    print(len(joinedDF[joinedDF['pred_locChanged']]))
    print(len(joinedDF[joinedDF['locChanged']]))
    print(len(joinedDF[joinedDF[['locChanged','pred_locChanged']].all(axis='columns')]))

    print("No Localization Change")
    print(len(joinedDF[~joinedDF['pred_locChanged']]))
    print(len(joinedDF[~joinedDF['locChanged']]))
    print(len(joinedDF[~joinedDF[['locChanged','pred_locChanged']].any(axis='columns')]))

    print("Localization Change But also we got the right loc")
    joinedDF['correct_24'] = joinedDF['Predicted Location_120']==joinedDF['Location_120']
    joinedDF['correct_120'] = joinedDF['Location_24']==joinedDF['Predicted Location_24']
    joinedDF['correct_all'] = joinedDF[['correct_24','correct_120']].all(axis='columns')
    joinedDF['correct_all_loc'] = joinedDF[['correct_all','locChanged','pred_locChanged']].all(axis='columns')

    print(len(joinedDF[joinedDF['correct_all_loc']]))
    print(len(joinedDF[joinedDF['correct_all']]))
    changedDF = joinedDF[joinedDF['locChanged']]


    #torch.save({'metrics':metricDF},'results/allRes.p')





