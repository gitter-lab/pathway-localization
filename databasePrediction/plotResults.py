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
from sklearn.metrics import accuracy_score,matthews_corrcoef,f1_score,balanced_accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

seed = 24
torch.manual_seed(seed)
np.random.seed(seed)

def plotMetric(metricDF, metric_name, x_val = None, hue=None, title="", ax=None, merged=False):
    fig=None
    if merged:
        fig = sns.barplot(x=x_val,y=metric_name, hue=hue, data=metricDF, ax=ax)
    else:
        fig = sns.boxplot(x=x_val,y=metric_name, hue=hue, data=metricDF, ax=ax)
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.35), ncol=4)
    plt.title(title)

def stratByMissing(df):
    order = ['FullyConnectedNN','GCN','GAT','GIN','Logit','RF','TrainedPGM','NaivePGM']
    locDict = {"cytosol":0,"extracellular":1,"membrane":2,"mitochondrion":5,"nucleus":3,"secretory-pathway":4}
    for i in range(6):
        locDict[i] = i
    allData = dict()
    allData['Model'] = []
    allData['Data'] = []
    allData['Features'] = []
    allData['Pathway Size'] = []
    allData['Missing Node Data'] = []
    allData['Localization'] = []
    allData['Predicted Localization'] = []
    allData['Unique Localizations'] = []
    for index, row in df.iterrows():
        model = row['Model']
        data = row['Data']
        feats = row['Features']
        sizeVec = row['p_size']
        missingVec = row['missing_features']
        yVec = row['allY']
        predVec = row['allPred']
        uLocVec = row['Unique Localizations']

        for i in range(len(yVec)):
            allData['Model'].append(model)
            allData['Data'].append(data)
            allData['Features'].append(feats)
            allData['Pathway Size'].append(sizeVec[i])
            allData['Missing Node Data'].append(missingVec[i])
            allData['Localization'].append(locDict[yVec[i]])
            allData['Predicted Localization'].append(locDict[predVec[i]])
            allData['Unique Localizations'].append(uLocVec[i])

    edgeDF = pd.DataFrame.from_dict(allData)
    print(edgeDF['Features'].unique())
    edgeDF = edgeDF[edgeDF['Features']=='ComPPI']
    groups = edgeDF.groupby(['Model','Data','Missing Node Data'])
    res = groups.apply(lambda group: f1_score(group['Localization'],group['Predicted Localization'], average='macro'))
    #res = groups.apply(lambda group: balanced_accuracy_score(group['Localization'],group['Predicted Localization']))
    res = res.reset_index()
    res = res.rename({0:'F1 Score'},axis=1)
    f,allAx = plt.subplots(1,2,sharey=True)
    i=0
    for data in res['Data'].unique():
        subRes = res[res['Data']==data]
        sns.barplot(x="Model",y='F1 Score',hue="Missing Node Data",data=subRes,ax=allAx[i],order=order)
        allAx[i].set_title(data)
        i+=1
    allAx[0].get_legend().remove()
    allAx[1].set_ylabel('')
    allAx[1].legend(ncol=3, title="Amount of Missing Data")
    plt.show()

    res['Missing Data Amount'] = res['Data'] +": "+ res['Missing Node Data'].astype(str)
    bar_pal = sns.color_palette('icefire',8)
    bar_pal[3] = bar_pal[-1]
    bar_pal[4] = bar_pal[-2]
    bar_pal[5] = bar_pal[-3]
    sns.barplot(x="Model",y='F1 Score',hue="Missing Data Amount",data=res,order=order,palette=bar_pal)
    plt.legend(ncol=2, title="Amount of Missing Data")
    plt.show()

    edgeDF = edgeDF[edgeDF['Model']=='GAT']
    pbDF = edgeDF[edgeDF['Data']=='PathBank']
    reactomeDF = edgeDF[edgeDF['Data']=='Reactome']
    f,allAx = plt.subplots(4,2,sharex=False, sharey=True)
    sns.histplot(x="Missing Node Data",hue="Missing Node Data",data=pbDF, discrete=True,ax=allAx[0][0],stat='percent',palette=bar_pal[:3])
    allAx[0][0].get_legend().remove()
    allAx[0][0].set_title("PathBank")
    allAx[0][0].set_xlabel("Amount of Missing Data")
    sns.histplot(x="Unique Localizations",hue="Missing Node Data",data=pbDF[pbDF["Missing Node Data"]==0], discrete=True,ax=allAx[1][0],stat='percent',palette=[bar_pal[0]])
    sns.histplot(x="Unique Localizations",hue="Missing Node Data",data=pbDF[pbDF["Missing Node Data"]==1], discrete=True,ax=allAx[2][0],stat='percent',palette=[bar_pal[1]])
    sns.histplot(x="Unique Localizations",hue="Missing Node Data",data=pbDF[pbDF["Missing Node Data"]==2], discrete=True,ax=allAx[3][0],stat='percent',palette=[bar_pal[2]])

    sns.histplot(x="Missing Node Data",hue="Missing Node Data",data=reactomeDF, discrete=True,ax=allAx[0][1],stat='percent',palette=bar_pal[3:6])
    allAx[0][1].get_legend().remove()
    allAx[0][1].set_title("Reactome")
    allAx[0][1].set_xlabel("Amount of Missing Data")
    sns.histplot(x="Unique Localizations",hue="Missing Node Data",data=reactomeDF[reactomeDF["Missing Node Data"]==0], discrete=True,ax=allAx[1][1],stat='percent',palette=[bar_pal[3]])
    sns.histplot(x="Unique Localizations",hue="Missing Node Data",data=reactomeDF[reactomeDF["Missing Node Data"]==1], discrete=True,ax=allAx[2][1],stat='percent',palette=[bar_pal[4]])
    sns.histplot(x="Unique Localizations",hue="Missing Node Data",data=reactomeDF[reactomeDF["Missing Node Data"]==2], discrete=True,ax=allAx[3][1],stat='percent',palette=[bar_pal[5]])
    plt.show()
    sns.set_palette('crest')
    return

if __name__ == "__main__":

    data = torch.load(argv[1])
    metricDF = data['metrics']
    metricMergedDF = data['mergedMetrics']

    metricDF['Unique Localizations'] = metricDF["Unique Localizations"].astype('category')

    sns.set_theme(style="white", context='paper')
    sns.set(font_scale=1.4)
    sns.set_palette('crest')
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["figure.figsize"] = (12,5)

    plt.rcParams['font.sans-serif'] = "Oswald"
    plt.rcParams['font.family'] = "sans-serif"

    #Rename things
    colRename = {'bal_acc':'Balanced Accuracy','f1':'F1 Score','model':'Model','data':'Data','features':'Features'}
    metricDF = metricDF.rename(mapper=colRename, axis=1)
    metricMergedDF = metricMergedDF.rename(mapper=colRename, axis=1)

    modelRename = {'LinearNN':'FullyConnectedNN','SimpleGCN':'GCN','GATCONV':'GAT','GIN2':'GIN','logit':'Logit','rf':'RF','TrainedPGM':'TrainedPGM','NaivePGM':'NaivePGM'}
    metricDF['Model'] = metricDF['Model'].map(modelRename, na_action='ignore')
    metricMergedDF['Model'] = metricMergedDF['Model'].map(modelRename, na_action='ignore')

    dataRename = {'allReactomePaths':'Reactome','allPathBank':'PathBank'}
    metricDF['Data'] = metricDF['Data'].map(dataRename, na_action='ignore')
    metricMergedDF['Data'] = metricMergedDF['Data'].map(dataRename, na_action='ignore')

    featureRename = {'mergedKeyWords_5':'Uniprot Keywords','comPPINodes':'ComPPI','compartmentsNodes':'Compartments'}
    metricDF['Features'] = metricDF['Features'].map(featureRename, na_action='ignore')
    metricMergedDF['Features'] = metricMergedDF['Features'].map(featureRename, na_action='ignore')

    stratByMissing(metricMergedDF)
    #sns.lmplot(x='missing_features',y='F1 Score',data=metricDF,row='Model',col='Data')
    #f, ax = plt.subplots(1,2,sharey=True,sharex=True)
    #gatDF = metricDF[metricDF['Model']=='GAT']
    #print(gatDF)
    #sns.histplot(data=gatDF[gatDF['Data']=='Reactome'], x='Location', stat='percent',discrete=True,ax=ax[0])
    #ax[0].set_title('Reactome')
    #sns.histplot(data=gatDF[gatDF['Data']=='PathBank'], x='Location', stat='percent',discrete=True,ax=ax[1])
    #ax[1].set_title('PathBank')
    #plt.show()


    ##Features effect
    for dataF in metricDF['Data'].unique():
        sub_metricDF = metricDF[metricDF['Data']==dataF]
        sub_metricMergedDF = metricMergedDF[metricMergedDF['Data']==dataF]

        plotMetric(sub_metricMergedDF, 'Balanced Accuracy', x_val='Model', hue='Features',merged=True)
        plt.suptitle("Merged "+dataF)
        plt.show()

        plotMetric(sub_metricMergedDF, 'F1 Score', x_val='Model', hue='Features',merged=True)
        plt.suptitle("Merged "+dataF)
        plt.show()

        plotMetric(sub_metricDF, 'Balanced Accuracy', x_val='Model', hue='Features')
        plt.suptitle(dataF+" Pathways")
        plt.show()

        plotMetric(sub_metricDF, 'F1 Score', x_val='Model', hue='Features')
        plt.suptitle(dataF+" Pathways")
        plt.show()

    ##Data effect
    #plotMetric(metricMergedDF, 'F1 Score', x_val='Model', hue='Data',title="")
    #plotMetric(metricDF, 'acc', x_val='Model', hue='Data',title="")
    #plotMetric(metricDF, 'F1 Score', x_val='Model', hue='Data',title="")

    ##Feature effect
    #plotMetric(metricMergedDF, 'F1 Score', x_val='Model', hue='Features',title="")
    #plotMetric(metricDF, 'acc', x_val='Model', hue='Features',title="")
    #plotMetric(metricDF, 'F1 Score', x_val='Model', hue='Features',title="")

    cnns = ['FullyConnectedNN','GCN','GAT','GIN']
    cls = ['RF','Logit']
    pgms = ['TrainedPGM','NaivePGM']
    allModels = cnns+cls+pgms

    #sizeMats = dict()
    #for model in metricDF['Model'].unique():
    #    sub_metricDF = metricDF[metricDF["Model"]==model]
    #    sizeMats[model] = confusion_matrix(sub_metricDF['Unique Localizations'],sub_metricDF['Predicted Unique Localizations'])
    #    #True is y axis, predicted is x
    #    tmpDF = pd.DataFrame(sizeMats[model],range(1,7),range(1,7))
    #    sns.heatmap(tmpDF, annot=True, linewidths=0.5, fmt='d',annot_kws={"size": 16}, cmap="YlGnBu")
    #    plt.title(model)
    #    plt.xlabel("Predicted Number of Localizations")
    #    plt.ylabel("True Number of Localizations")
    #    plt.show()

    #f, allAx = plt.subplots(10,1, sharex=True, sharey=True)
    #cnn_df = metricDF[metricDF['Model'].isin(cnns)]
    #cls_df = metricDF[metricDF['Model'].isin(cls)]
    #pgm_df = metricDF[metricDF['Model'].isin(pgms)]
    #sns.histplot(metricDF, x="Unique Localizations", hue="Data", multiple="stack",ax=ax1,discrete=True,stat="percent",palette="ch:s=2,r=0.05,l=0.65,d=0.35_r")
    #sns.histplot(pgm_df, x="Predicted Unique Localizations", hue="Model", multiple="stack",ax=ax4,discrete=True,stat="percent",palette="ch:s=0.5,r=0.05,l=0.65,d=0.35_r")
    #sns.histplot(cls_df, x="Predicted Unique Localizations", hue="Model", multiple="stack",ax=ax3,discrete=True,stat="percent",palette="ch:s=3,r=0.05,l=0.65,d=0.35_r")
    #sns.histplot(cnn_df, x="Predicted Unique Localizations", hue="Model", multiple="stack",ax=ax2,discrete=True,stat="percent",palette="ch:s=2.5,r=0.05,l=0.85,d=0.15_r")
    #plt.show()

    pbDF = metricDF[metricDF['Data']=='PathBank']
    reactomeDF = metricDF[metricDF['Data']=='Reactome']
    f, allAx = plt.subplots(9,2, sharex=True, sharey=True)
    sns.histplot(reactomeDF, hue="Data", x="Unique Localizations", ax=allAx[0][0],discrete=True,stat="percent",palette="ch:s=2,r=0.05,l=0.65,d=0.35_r")
    sns.histplot(pbDF, hue="Data", x="Unique Localizations", ax=allAx[0][1],discrete=True,stat="percent",palette="ch:s=2,r=0.05,l=0.65,d=0.35_r")
    pbDF = metricDF[metricDF['Data']=='PathBank']
    for i in range(len(allModels)):
        cAx = allAx[i+1]
        if allModels[i] in cnns:
            s = '0.5'
        elif allModels[i] in cls:
            s = '3'
        else:
            s = '2.5'
        sns.histplot(reactomeDF[reactomeDF['Model']==allModels[i]], hue="Model", x="Predicted Unique Localizations", ax=cAx[0],discrete=True,stat="percent",palette="ch:s="+s+",r=0.05,l=0.65,d=0.35_r")
        sns.histplot(pbDF[pbDF['Model']==allModels[i]], hue="Model", x="Predicted Unique Localizations", ax=cAx[1],discrete=True,stat="percent",palette="ch:s="+s+",r=0.05,l=0.65,d=0.35_r")
    plt.show()

    #Unique loc effect
    plotMetric(reactomeDF, 'F1 Score', hue='Model', x_val='Unique Localizations')
    plt.show()
    plotMetric(pbDF, 'F1 Score', hue='Model', x_val='Unique Localizations')
    plt.show()
    plotMetric(metricDF, 'F1 Score', hue='Model', x_val='Unique Localizations')
    plt.show()



