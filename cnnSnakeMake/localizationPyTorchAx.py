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

seed = 24
torch.manual_seed(seed)
np.random.seed(seed)

def testCNNs(parameterization):
    dataFile = parameterization["dataFile"]

    mName = parameterization["mName"]
    epochs = parameterization["epochs"]
    learningRate = parameterization["lRate"]

    dataState = torch.load(dataFile)
    train_loaders = dataState['train_loaders']
    test_loaders = dataState['test_loaders']
    dataList = dataState['dataList']


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = 'cpu'
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

    #elif mName == 'NODE2VEC'
    #print("\n\n"+mName,epochs)
    #models = []
    #for i in range(len(train_loaders)):
    #    models.append(NODE2VEC(dataList[0],parameterization))

    else:
        print('Invalid Model!')
        return

    accuracy = evalModelCV(models, train_loaders, test_loaders,device, mName,epochs,learningRate)
    return accuracy

def train(loader, model, optimizer, criterion,device):
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    for data in loader:  # Iterate in batches over the training dataset.
         data.to(device)
         out,e = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
         loss = criterion(out, data.y)  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.

def testTraining(loader, model,device):
     model.eval()
     correct = 0
     total = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         data.to(device)
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

def evalModelCV(models, train_loaders, test_loaders,device, mName, epochs = 1000, lr=0.001):
    optimizers = []
    losses = []
    checkpoint_interval = 50
    start_epoch = 1
    for i in range(len(train_loaders)):
        optimizers.append(torch.optim.Adam(models[i].parameters(), lr=lr))
        losses.append(torch.nn.CrossEntropyLoss())

    train_acc_list = []
    test_acc_list = []
    perfDict = dict()
    perfDict["Epoch"] = []
    perfDict["Accuracy"] = []
    perfDict["Data"] = []
    perfDict["Fold"] = []
    perfDict["Model"] = []

    #Load from checkpoint
    #for i in range(0,epochs,checkpoint_interval):
    #    if os.path.exists(modelFile+"_checkpoint_0_"+str(i)):
    #        for f in range(len(train_loaders)):
    #            checkpoint = torch.load(modelFile+"_checkpoint_"+str(f)+"_"+str(i))
    #            models[f].load_state_dict(checkpoint['model_state_dict'])
    #            optimizers[f].load_state_dict(checkpoint['optimizer_state_dict'])
    #            start_epoch = checkpoint['epoch']
    #            losses[f] = checkpoint['loss']
    #        perfDict = torch.load(modelFile+"_checkpoint_perf_"+str(epoch))['perDF']

    for epoch in range(start_epoch, epochs+1):
        train_acc_total = 0.0
        test_acc_total = 0.0
        for i in range(len(train_loaders)):
            train(train_loaders[i], models[i], optimizers[i], losses[i], device)
            train_acc = testTraining(train_loaders[i], models[i], device)
            test_acc = testTraining(test_loaders[i], models[i], device)
            train_acc_total += train_acc
            test_acc_total += test_acc
            #if epoch % 10 == 0:
            #    perfDict["Epoch"].append(epoch)
            #    perfDict["Accuracy"].append(train_acc)
            #    perfDict["Fold"].append(i)
            #    perfDict["Data"].append("Training Set")
            #    perfDict["Model"].append(mName)

            #    perfDict["Epoch"].append(epoch)
            #    perfDict["Accuracy"].append(test_acc)
            #    perfDict["Fold"].append(i)
            #    perfDict["Data"].append("Testing Set")
            #    perfDict["Model"].append(mName)
        train_acc_list.append(train_acc_total/len(train_loaders))
        test_acc_list.append(test_acc_total/len(train_loaders))

        if (epoch%checkpoint_interval)==0:
            print(f'Epoch: {epoch:03d}, Train Acc: {train_acc_list[-1]:.4f}, Test Acc: {test_acc_list[-1]:.4f}')
        #    for i in range(len(train_loaders):
        #        torch.save({
        #        'epoch': epoch,
        #        'model_state_dict': models[i].state_dict(),
        #        'optimizer_state_dict': optimizer[i].state_dict(),
        #        'loss': losses[i],
        #        }, modelFile+"_checkpoint_"+str(fold)+"_"+str(epoch))
        #    torch.save({'perfDict': perfDict},modelFile+"_checkpoint_perf_"+str(epoch))

    #model.eval()
    #outputs = dict()
    #perfDF = pd.DataFrame.from_dict(perfDict)
    #for i in range(len(train_loaders)):
    #    outputs[fold] = []
    #    for data in loader:  # Iterate in batches over the training/test dataset.
    #         out,e = model(data.x, data.edge_index, data.batch)
    #         outputs[fold].append((out,data.y))
    #         torch.save({'outputs':outputs,'perfDF':perfDF},modelFile+"_finalOutput.p")
    #print(np.mean(test_acc_list),sem(test_acc_list))
    return {'accuracy': (np.mean(test_acc_list),sem(test_acc_list))}

def runTrial(client):
    parameters, trial_index = ax_client.get_next_trial()
    ax_client.complete_trial(trial_index=trial_index, raw_data=testCNNs(parameters))
    return

if __name__ == "__main__":
    #networksFile = argv[1]
    #featuresFile = argv[2]
    mName = argv[1]
    dataFile = argv[2]
    outF = argv[3]

    #networksFile = 'allDevReactomePathsCom.txt'
    #networksFile = 'allDevReactomePaths.txt'
    #featuresFile = '../scripts/exploratoryScripts/comPPINodes.tsv'
    #featuresFile = '../data/uniprotKeywords/mergedKeyWords_5.tsv'

    ax_client = AxClient()

    parameters=[
        {"name": "lRate", "type": "fixed", "value": 0.00005},
        {"name": "epochs", "type": "fixed", "value_type": "int", "value": 2000},
        {"name": "dataFile", "type": "fixed", "value_type": "str", "value": dataFile},
        {"name": "mName", "type": "fixed", "value_type": "str", "value": mName}]
    if mName == 'LinearNN':
        parameters.append({"name": "dim", "type": "range", "value_type": "int", "bounds": [24, 128]})
    elif mName == 'SimpleGCN':
        parameters.append({"name": "dim", "type": "range", "value_type": "int", "bounds": [24, 128]})
        parameters.append({"name": "c_depth", "type": "range", "value_type": "int", "bounds": [1, 10]})
        parameters.append({"name": "l_depth", "type": "range", "value_type": "int", "bounds": [1, 10]})
    elif mName == 'GATCONV':
        parameters.append({"name": "dim", "type": "range", "value_type": "int", "bounds": [24, 128]})
        parameters.append({"name": "c_depth", "type": "range", "value_type": "int", "bounds": [1, 10]})
        parameters.append({"name": "num_heads", "type": "range", "value_type": "int", "bounds": [1, 10]})
    elif mName == 'PANCONV':
        parameters.append({"name": "dim", "type": "range", "value_type": "int", "bounds": [24, 128]})
        parameters.append({"name": "c_depth", "type": "range", "value_type": "int", "bounds": [1, 10]})
        parameters.append({"name": "filters", "type": "range", "value_type": "int", "bounds": [1, 10]})
    elif mName == 'GIN2':
        parameters.append({"name": "dim", "type": "range", "value_type": "int", "bounds": [24, 128]})
        parameters.append({"name": "c_depth", "type": "range", "value_type": "int", "bounds": [1, 10]})

    ax_client.create_experiment(
        name=mName,
        parameters=parameters,
        objective_name="accuracy",
        minimize=False)

    startI = 0
    for i in range(29,-1,-1):
        curPath = outF+"_iter"+str(i)
        if os.path.exists(curPath):
            startI = i+1
            ax_client = AxClient.load_from_json_file(curPath)
            break
    for i in range(startI,30):
        parameters, trial_index = ax_client.get_next_trial()
        # Local evaluation here can be replaced with deployment to external system.
        ax_client.complete_trial(trial_index=trial_index, raw_data=testCNNs(parameters))
        ax_client.save_to_json_file(filepath=outF+"_iter"+str(i))
    #num_cores = multiprocessing.cpu_count()
    #print(num_cores)
    #Parallel(n_jobs=num_cores-1)(delayed(runTrial)(ax_client) for i in range(10))
    #Parallel(n_jobs=num_cores-1)(delayed(runTrial)(ax_client) for i in range(3))
    #Parallel(n_jobs=num_cores-1)(delayed(runTrial)(ax_client) for i in range(3))
    best_parameters, values = ax_client.get_best_parameters()
    ax_client.save_to_json_file(filepath=outF)
    print(best_parameters)



