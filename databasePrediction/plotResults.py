#!/usr/bin/env python
# coding: utf-8
from sys import argv
import matplotlib.pyplot as plt
import seaborn as sns
import torch

def plotMetric(metricDF, metric_name, x_val = None, hue=None, title="", ax=None, merged=False):
    fig=None
    if merged:
        fig = sns.barplot(x=x_val,y=metric_name, hue=hue, data=metricDF, ax=ax)
    else:
        fig = sns.boxplot(x=x_val,y=metric_name, hue=hue, data=metricDF, ax=ax)
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.35), ncol=4)
    plt.title(title)

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

    #Rename things for nice plots
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

    ##Features effect
    for dataF in metricDF['Data'].unique():
        sub_metricDF = metricDF[metricDF['Data']==dataF]
        sub_metricMergedDF = metricMergedDF[metricMergedDF['Data']==dataF]

        plotMetric(sub_metricMergedDF, 'F1 Score', x_val='Model', hue='Features',merged=True)
        plt.suptitle("Merged "+dataF)
        plt.show()

        plotMetric(sub_metricDF, 'F1 Score', x_val='Model', hue='Features')
        plt.suptitle(dataF+" Pathways")
        plt.show()

    cnns = ['FullyConnectedNN','GCN','GAT','GIN']
    cls = ['RF','Logit']
    pgms = ['TrainedPGM','NaivePGM']
    allModels = cnns+cls+pgms

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
