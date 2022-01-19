import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

def getLocData(networksFile, featuresFile):
    #Load in the reactome networks as a dictionary of dataFrames
    sns.set_theme(style="darkgrid")
    #We will not add pathways with fewer nodes than this
    MIN_NETWORK_SIZE_THRESHOLD = 4

    allPathDFs = dict()
    allPathNets = dict()

    locList = ["cytosol","extracellular","membrane","nucleus","secretory-pathway","mitochondrion"]
    locDict = {"cytosol":0,"extracellular":1,"membrane":2,"mitochondrion":5,"nucleus":3,"secretory-pathway":4}

    colNames = ["Interactor1", "Edge Type", "Interactor2", "Location"]
    for line in open(networksFile, "r"):
        pName = line.strip().split("/")[-1]
        pathDF = pd.read_csv(line.strip(), sep="\t", header=None, names=colNames, dtype=str)
        pathDF["loc_feat"] = pathDF['Location'].map(locDict)
        if len(pathDF) >= MIN_NETWORK_SIZE_THRESHOLD:
            allPathDFs[pName] = pathDF
            allPathNets[pName] = nx.from_pandas_edgelist(pathDF, source='Interactor1',
                                                         target='Interactor2', edge_attr= ['Edge Type','Location','loc_feat'])
    print("Loaded in %d pathways" %len(allPathDFs))
    mergedDF = pd.concat(allPathDFs.values())
    dupDF = mergedDF[mergedDF.duplicated(subset=['Interactor1','Interactor2'],keep=False)]
    gdf = dupDF.groupby(['Interactor1','Interactor2'])['Location'].nunique()
    gdfNum = dupDF.groupby(['Interactor1','Interactor2'])['Location'].size()
    print("Number of total edges: %d, duplicated edges: %d, and contradictory edges: %d" %(len(mergedDF),len(dupDF),sum(gdfNum[gdf>1])))
    #print(gdf[gdf>1])

    #Load in all comPPI Data as a dataframe too
    featuresDF = pd.read_csv(featuresFile, sep="\t", index_col="uniprot")
    featuresDF = featuresDF.round(5)
    featuresDict = featuresDF.to_dict('index')

    #Make contradictory edges by features
    allDataDF = mergedDF.copy()
    appliedDF = allDataDF[['Interactor1','Interactor2']].apply(lambda x: getFeatColumns(featuresDict, locList, *x), axis=1, result_type='expand')
    allDataDF = pd.concat([allDataDF, appliedDF], axis='columns')

    fList = []
    for l in locList:
        fList.append(l+"_1")
        fList.append(l+"_2")

    allDataDF = allDataDF.dropna(subset=fList, how='all')
    print("Number of edges with at least 1 hit: %d" %len(allDataDF))

    conflictDF = allDataDF.groupby(fList, dropna=False)['Location'].nunique()
    conflictDFNum = allDataDF.groupby(fList, dropna=False)['Location'].size()
    #print(conflictDF)
    #print(conflictDFNum)
    #print(conflictDF[conflictDF>1])
    print("Number of edges where at least 1 other edge has the same features but a different class: %d" %sum(conflictDFNum[conflictDF>1]))
    meltedDF = allDataDF.melt(id_vars=['Location'], value_vars=fList, var_name='comPPI', value_name='Probability')
    meltedDF['comPPI'] = meltedDF['comPPI'].str[:-2]
    meltedDF = meltedDF.dropna()
    meltedDF['Match'] = meltedDF['Location']==meltedDF['comPPI']
    sns.boxplot(data=meltedDF, x='Location',y='Probability',hue='comPPI',hue_order=locList,order=locList)
    plt.show()

    #Get maxes
    #print(allDataDF[["Location"]])
    allDataDF['Loc_Max'] = allDataDF[fList].idxmax(axis=1).str[:-2]
    sns.catplot(data=allDataDF, kind='count', x="Location", row='Loc_Max',order=locList)
    plt.show()

    #Calculate theoretical max base classification performance
    #print(meltedDF)
    return

    #FeatureHistograms
    #g = sns.PairGrid(featuresDF)
    #g.map_diag(sns.histplot)
    #g.map_offdiag(sns.scatterplot)
    #plt.show()

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

    #Look at one of the networks
    testP = allPathNets['R-HSA-375165.txt']
    print(f'graph has {testP.number_of_nodes()} nodes and {testP.number_of_edges()} undirected edges')
    i=1
    print("\n3 Nodes:")
    for k,v in testP.nodes(data=True):
        print(k,v)
        i+=1
        if i>3: break

    print("\n3 Edges:")
    for k1,k2,v in testP.edges(data=True):
        print(k1,k2,v)
        i+=1
        if i>6: break

    return

def getFeatColumns(fD,locList,i1,i2):
    outDict=dict()
    if i1 in fD and i2 in fD:
        if str(fD[i1]) > str(fD[i2]):
            d1 = makeOutD(locList,fD,i1,"_1")
            d2 = makeOutD(locList,fD,i2,"_2")
            outDict = {**d1, **d2}
        else:
            d2 = makeOutD(locList,fD,i2,"_1")
            d1 = makeOutD(locList,fD,i1,"_2")
            outDict = {**d2, **d1}
    elif i1 in fD:
        d1 = makeOutD(locList,fD,i1,"_1")
        d2 = makeOutD(locList,fD,i2,"_2",True)
        outDict = {**d1, **d2}
    elif i2 in fD:
        d2 = makeOutD(locList,fD,i2,"_1")
        d1 = makeOutD(locList,fD,i1,"_2",True)
        outDict = {**d2, **d1}
    else:
        d1 = makeOutD(locList,fD,i1,"_1",True)
        d2 = makeOutD(locList,fD,i2,"_2",True)
        outDict = {**d1, **d2}

    return outDict

def makeOutD(locList,fD,i1,suff,miss=False):
    outD = dict()
    for l in locList:
        if miss:
            outD[l+suff] = np.NAN
        else:
            outD[l+suff] = fD[i1][l]
    return outD

if __name__ == "__main__":
    #networksFile = 'allDevReactomePathsCom.txt'
    networksFile = 'data/allPathBank.txt'
    featuresFile = 'data/compartmentsNodes.tsv'
    #featuresFile = '../data/uniprotKeywords/mergedKeyWords_5.tsv'
    getLocData(networksFile, featuresFile)
