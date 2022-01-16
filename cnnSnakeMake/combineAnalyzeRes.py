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
from sklearn.metrics import accuracy_score,matthews_corrcoef,f1_score
import matplotlib.pyplot as plt
import seaborn as sns

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

    device='cpu'

    ### CV Models (Just plots the average for now)
    if mName == 'LinearNN':
        models = []
        for i in range(len(train_loaders)):
            #There was some debate online about if deepcopy works, so let's just make new ones
            #dataList[0] just shows the model the type of data shape it's looking at
            models.append(LinearNN(dataList[0],parameterization).to(device))

    elif mName == 'SimpleGCN':
        models = []
        for i in range(len(train_loaders)):
            models.append(SimpleGCN(dataList[0],parameterization).to(device))

    elif mName == 'GATCONV':
        models = []
        for i in range(len(train_loaders)):
            models.append(GATCONV(dataList[0],parameterization).to(device))

    elif mName == 'PANCONV':
        models = []
        for i in range(len(train_loaders)):
            models.append(PANCONV(dataList[0],parameterization).to(device))

    elif mName == 'GIN2':
        models = []
        for i in range(len(train_loaders)):
            models.append(GIN2(dataList[0],parameterization).to(device))
    else:
        print('Invalid Model!')
        return
    predList,yList = evalModels(models, train_loaders, test_loaders,device, mName, parameterization, resultsData,learningRate)
    metrics = getMetricLists(predList, yList)
    return metrics

def getOuts(loader, model, device):
     model.eval()
     correct = 0
     total = 0
     predList = []
     yList = []
     for data in loader:  # Iterate in batches over the training/test dataset.
         data.to(device)
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
     return predList,yList  # Derive ratio of correct predictions.

def getMetricLists(predList,yList):
    mccList = []
    accList = []
    f1List = []
    for i in range(len(predList)):
        pred = predList[i].numpy()
        y = yList[i].numpy()
        mcc = matthews_corrcoef(y,pred)
        f1 = f1_score(y, pred, average='micro')
        acc = accuracy_score(y, pred)
        mccList.append(mcc)
        f1List.append(f1)
        accList.append(acc)
    return {'mcc':mccList, 'f1':f1List, 'acc':accList}

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

def evalModels(models, train_loaders, test_loaders,device, mName, parameters, resultsData, lr):
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
        preds, ys = getOuts(test_loaders[i], models[i], device)
        predList += preds
        yList += ys
    return predList,yList


if __name__ == "__main__":
    fList = []
    for f in argv[1:]:
        fList.append(f)
    perfs = dict()
    perfs['model']=[]
    for f in fList:
        metrics = evalCNNs(f)
        numInst = 0
        for m in metrics:
            if not m in perfs:
                perfs[m] = []
            perfs[m] += metrics[m]
            numInst = len(metrics[m])
        perfs['model']+=[f]*numInst
    metricDF = pd.DataFrame.from_dict(perfs)
    #sns.boxplot(x='model',y='mcc',data=metricDF)
    #plt.show()
    torch.save({'metrics':metricDF},'results/allRes.p')





