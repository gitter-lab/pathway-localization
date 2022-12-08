#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from models import *
from sys import argv
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

    #Add dummy classifier
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dumPredList = []
    for y in yList:
        dummy_clf.fit(y,y)
        dumPredList.append(dummy_clf.predict(y))
    dumMetrics, dumMergedMetrics = getMetricLists(dumPredList,yList, eNames)
    dumMetrics['model'] = ['Baseline']*num_inst

    dumMetrics['Timepoint'] = [pathwaySet]*num_inst
    dumMetrics['features'] = [features]*num_inst

    dumMergedMetrics['model'] = ['Baseline']
    dumMergedMetrics['Timepoint'] = [pathwaySet]
    dumMergedMetrics['features'] = [features]
    namedDF = matchNamesPreds(dumPredList,yList,eNames)

    return metrics, mergedMetrics, namedDF, dumMetrics, dumMergedMetrics

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
            f1 = f1_score(y, pred, average='weighted')
            acc = accuracy_score(y, pred)
            bacc = balanced_accuracy_score(y, pred)
            mccList.append(mcc)
            f1List.append(f1)
            accList.append(acc)
            baccList.append(bacc)
            sizeList.append(len(y))

    netOut = {'mcc':mccList, 'f1':f1List, 'Accuracy':accList, 'Balanced Accuracy':baccList, 'p_size':sizeList}
    allOut = {'mcc':[matthews_corrcoef(totalY, totalPred)],
              'f1':[f1_score(totalY, totalPred, average='macro')],
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
    losses = resultsData['losses']
    for i in range(len(train_loaders)):
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
        order = ["Pathway Database","Diff. Experiment","Same Experiment","Baseline"]
        sns.boxplot(x=x_val,y=metric_name, hue=hue, data=metricDF, ax=ax,order=order)
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
        dumMetrics = None
        mergedMetrics = None
        dumMergedMetrics = None
        if f[:3]=="pgm":
            metrics, mergedMetrics = evalPGM(f)
        elif f[:3]=="sk_":
            metrics, mergedMetrics = evalSKModel(f)
        else:
            metrics, mergedMetrics, namedDF, dumMetrics, dumMergedMetrics = evalCNNs(f)
            namedList.append(namedDF)
        numInst = 0
        for m in metrics:
            if not m in perfs:
                perfs[m] = []
                perfsMerged[m] = []
            perfs[m] += metrics[m]
            perfsMerged[m] += mergedMetrics[m]
            perfs[m] += dumMetrics[m]
            perfsMerged[m] += dumMergedMetrics[m]
            numInst = len(metrics[m])
    metricDF = pd.DataFrame.from_dict(perfs)
    metricMergedDF = pd.DataFrame.from_dict(perfsMerged)

    sns.set_theme(style="white", context='paper')
    sns.set(font_scale=2.0)
    sns.set_palette('crest')
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["figure.figsize"] = (12,5)

    plt.rcParams['font.sans-serif'] = "Oswald"
    plt.rcParams['font.family'] = "sans-serif"

    metricDF = metricDF[(metricDF.model!='Baseline') | ((metricDF.Timepoint!='120hpi') & (metricDF.Timepoint!='samehpi'))]

    #Give things nice names
    metricDF.Timepoint[metricDF.Timepoint=='120hpi'] = "Pathway Database"
    metricDF.Timepoint[metricDF.Timepoint=='samehpi'] = "Same Experiment"
    metricDF.Timepoint[metricDF.Timepoint=='egf120hpi'] = "Diff. Experiment"
    metricDF.Timepoint[metricDF.model=='Baseline'] = "Baseline"

    print(metricDF.groupby(["Timepoint"]).mean())
    f, (ax1) = plt.subplots(1,1)
    plotMetric(metricDF, 'f1', x_val='Timepoint', ax=ax1)
    ax1.set_ylabel('F1 Score')
    ax1.set_xlabel('Training Data')
    ax1.set_ylim(bottom=0)
    ax1.set_ylim(top=1)
    plt.show()

    torch.save({'metrics':metricDF},'results/allRes.p')





