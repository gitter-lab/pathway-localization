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

def plotMetric(metricDF, metric_name, x_val = None, hue=None, title="", ax=None):
    if "Merged" in title:
        sns.barplot(x=x_val,y=metric_name, hue=hue, data=metricDF, ax=ax)
    else:
        sns.boxplot(x=x_val,y=metric_name, hue=hue, data=metricDF, ax=ax)
    plt.legend(loc='upper right')
    plt.title(title)
    plt.show()

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

    # Say, "the default sans-serif font is COMIC SANS"
    plt.rcParams['font.sans-serif'] = "Oswald"
    # Then, "ALWAYS use sans-serif fonts"
    plt.rcParams['font.family'] = "sans-serif"

    metricDF["Balanced Accuracy"] = metricDF["bal_acc"]
    ##Features effect
    for dataF in metricDF['data'].unique():
        sub_metricDF = metricDF[metricDF['data']==dataF]
        #sub_metricMergedDF = metricMergedDF[metricMergedDF['data']==dataF]
        #plotMetric(sub_metricMergedDF, 'Balanced Accuracy', x_val='model', hue='features',title="Merged"+dataF)
        #plotMetric(sub_metricDF, 'acc', x_val='model', hue='features',title=dataF)
        plotMetric(sub_metricDF, 'Balanced Accuracy', x_val='model', hue='features',title=dataF)
        #plotMetric(sub_metricDF, 'mcc', x_val='model', hue='features',title=dataF)

    ##Data effect
    #plotMetric(metricMergedDF, 'Balanced Accuracy', x_val='model', hue='data',title="")
    #plotMetric(metricDF, 'acc', x_val='model', hue='data',title="")
    plotMetric(metricDF, 'Balanced Accuracy', x_val='model', hue='data',title="")

    ##Feature effect
    #plotMetric(metricMergedDF, 'Balanced Accuracy', x_val='model', hue='features',title="")
    #plotMetric(metricDF, 'acc', x_val='model', hue='features',title="")
    #plotMetric(metricDF, 'Balanced Accuracy', x_val='model', hue='features',title="")

    cnns = ['LinearNN','SimpleGCN','GATCONV','GIN2']
    cls = ['rf','logit']
    pgms = ['TrainedPGM','NaivePGM']

    #sizeMats = dict()
    #for model in metricDF['model'].unique():
    #    sub_metricDF = metricDF[metricDF["model"]==model]
    #    sizeMats[model] = confusion_matrix(sub_metricDF['Unique Localizations'],sub_metricDF['Predicted Unique Localizations'])
    #    #True is y axis, predicted is x
    #    tmpDF = pd.DataFrame(sizeMats[model],range(1,7),range(1,7))
    #    sns.heatmap(tmpDF, annot=True, linewidths=0.5, fmt='d',annot_kws={"size": 16}, cmap="YlGnBu")
    #    plt.title(model)
    #    plt.xlabel("Predicted Number of Localizations")
    #    plt.ylabel("True Number of Localizations")
    #    plt.show()

    f, (ax1, ax2, ax3, ax4) = plt.subplots(4,1, sharex=True, sharey=True)
    cnn_df = metricDF[metricDF['model'].isin(cnns)]
    cls_df = metricDF[metricDF['model'].isin(cls)]
    pgm_df = metricDF[metricDF['model'].isin(pgms)]
    sns.histplot(metricDF, x="Unique Localizations", hue="data", multiple="stack",ax=ax1,discrete=True,stat="percent",palette="ch:s=2,r=0.05,l=0.65,d=0.35_r")
    sns.histplot(pgm_df, x="Predicted Unique Localizations", hue="model", multiple="stack",ax=ax4,discrete=True,stat="percent",palette="ch:s=0.5,r=0.05,l=0.65,d=0.35_r")
    sns.histplot(cls_df, x="Predicted Unique Localizations", hue="model", multiple="stack",ax=ax3,discrete=True,stat="percent",palette="ch:s=3,r=0.05,l=0.65,d=0.35_r")
    sns.histplot(cnn_df, x="Predicted Unique Localizations", hue="model", multiple="stack",ax=ax2,discrete=True,stat="percent",palette="ch:s=2.5,r=0.05,l=0.85,d=0.15_r")
    plt.show()

    #Unique loc effect
    plotMetric(metricDF, 'Balanced Accuracy', hue='model', x_val='Unique Localizations')



