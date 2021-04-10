import pandas as pd
from sys import argv
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
import numpy as np
import yaml
from sklearn.metrics import balanced_accuracy_score,matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

#sns.set(rc={'axes.facecolor':'#282828', 'figure.facecolor':'#282828'})
#sns.set_theme()

def main():
    localF = argv[1]  # List of reactome pathways with localizations
    comPPIF = argv[2]  # Raw comPPI tsv
    idMapFile = argv[3]  # Name mapping for comPPI to uniprot ID's
    locNameMap = argv[4]  # comPPI localization names to GO terms map
    reactomeNameMap = argv[5]  # GO terms to names map tsv file
    if len(argv) > 6:
        gBasicModelRes = argv[6]  # File with basic graphical model results
        modelsFile = argv[7] # File with other model results

    #Loads GO COMPPI name mapping
    goComNameMap = loadComPPIGOMap(locNameMap)

    # Returns a dict of gene name to uniprot
    nameDict = loadIDMap(idMapFile)

    # Load Reactome Mapping Data
    allPaths = loadNetworksAsTables(localF)

    #Load comPPI data
    comPPIDict = loadComPPI(comPPIF)
    comPPITable = saveComPPIProtTable(comPPIDict, nameDict)

    missList = []
    sizeList = []
    delList = []
    for pathName in allPaths:
       path = allPaths[pathName]
       missList.append(getComPPILabels(path, comPPIDict, nameDict)*100)
       if missList[-1] >= 75.0:
           delList.append(pathName)
       sizeList.append(len(path))
    #for p in delList:
    #    del allPaths[p]
    #sns.scatterplot(x=sizeList,y=missList)
    #plt.xlabel("Pathway Size")
    #plt.ylabel("Percent not in ComPPI")
    #plt.show()

    #Look at Reactome distribution
    compareComPPIReactome(allPaths, goComNameMap)
    goNameMap = dict()
    for line in open(reactomeNameMap):
       if "GO_TERM" in line:
           continue
       lineList = line.strip().split("\t")
       goNameMap[lineList[0]] = lineList[1]
    for path in allPaths:
       namedCol = []
       for index, row in allPaths[path].iterrows():
           goNums = row["LocationGO"].split(",")
           namedLs = []
           for l in goNums:
               n = goNameMap[l]
               namedLs.append(n)
           namedCol.append(",".join(namedLs))
       allPaths[path]["LocationName"] = namedCol

    mapReactomeNames(allPaths, goComNameMap)
    resCols = []
    resCols.append(addResult(allPaths, "trainedModelNoComplex53.txt","Trained No Complexes"))
    for line in open(modelsFile):
        resCols.append(addResult(allPaths, line.strip(), line.strip().split(".")[0]))

    #resCols.append(addResult(allPaths, gBasicModelRes,"Basic Model"))
    #resCols.append(addResult(allPaths, "basicModelDumb10000.txt","Basic Fill Model"))
    #resCols.append(addResult(allPaths, "trainedModel10000.txt","Trained No Groups"))
    #resCols.append(addResult(allPaths, "trainedModelSepInter10000.txt","Trained GMM PottsCS"))
    #resCols.append(addResult(allPaths, "trainedModel53.txt","Trained RF Concat"))
    #resCols.append(addResult(allPaths, "trainedModelNoComplex53.txt","Trained No Complexes"))
    #resCols.append(addResult(allPaths, "kmeans3.txt","H Model 3 Clusters"))

    #pF = open("devBasicReactComPPI.pkl","wb")
    #pkl.dump(allPaths,pF)
    #pF.close()
    #makeRFLabel(allPaths,comPPITable)

    #pF = open("devBasicReactComPPI.pkl", "rb")
    #allPaths = pkl.load(pF)
    #pF.close()
    #for p in allPaths:
    #    print(p)
    #    print(len(allPaths[p][allPaths[p]["Trained No Complexes Loc"] != "Miss"]["Interactor1"].unique()))
        #allPaths[p][allPaths[p]["Trained No Complexes Loc"] != "Miss"].to_csv("../../data/dgmResultTablesNoC/"+p, sep="\t", index=False, header=True, columns=["Interactor1","Edge Type", "Interactor2", "ReactomeLoc","Trained No Complexes Loc","Trained RF Concat Loc","ComPPI_Labels","Basic Fill Model Loc"])
    #return
    analyzeReactomeData(allPaths, resCols)
    return

def addResult(allPaths, fName, colName):
    #Load trained graphical model
    gDict = loadGModelLabels(fName)
    delList = []
    for path in allPaths:
        if path in gDict:
            pathGDict = gDict[path]
            getGModelLabels(colName+" Loc",allPaths[path],pathGDict)
        else:
            delList.append(path)
    for p in delList:
        del allPaths[p]
    compareGModelReactome(allPaths, colName+" Loc", colName+" Results")
    return colName

def analyzeReactomeData(allPaths, rList):
    allEdges = pd.concat(allPaths.values())
    #Remove edges not in model with no complexes
    #allEdges = allEdges[allEdges["Trained No Complexes Loc"] != "Miss"]
    for r in rList:
        allEdges = allEdges[allEdges[r+" Loc"] != "Miss"]
    # Look at reactome data alone
    #print("Mean Location Counts: ",allEdges["LocationCount"].mean())
    #print(allEdges["Edge Type"].value_counts())
    #print(allEdges["Edge Type"].value_counts(normalize=True))
    #print("Number of Locations by Edge Counts:")
    #print(allEdges["LocationCount"].value_counts(normalize=True))

    #print("\n10 Most Common Locations:")
    #print(allEdges["LocationName"].value_counts(normalize=True).nlargest(10))
    #print("\nEdge Types")
    #print(allEdges["Edge Type"].value_counts(normalize=True))
    #print("\nLocation Counts by Edge Type")
    #print(allEdges.groupby(["Edge Type","LocationCount"]).size().reset_index(name="Count").sort_values(by="Count",ascending=False))
    #sns.heatmap(pd.crosstab(allEdges["Edge Type"], allEdges["LocationName"]))
    #plt.show()
    #print(allEdges["LocationGO"].value_counts())
    #print(allEdges["LocationGO"].value_counts(normalize=True).to_string())
    #print((allEdges["Edge Type"][allEdges["LocationCount"]==1]).value_counts(normalize=True))
    #print((allEdges["Edge Type"][allEdges["LocationCount"]==1]).value_counts(normalize=False))
    #cross = pd.crosstab(allEdges["Edge Type"], allEdges["Reactome ComPPI Comparison"],normalize="index")
    #sns.heatmap(cross)
    #plt.show()
    resList = ["Reactome ComPPI Comparison"]
    for r in rList:
        resList.append(r + " Results")
    print("--------------No Model Results----------------")
    print(allEdges["Reactome ComPPI Comparison"].value_counts(normalize=True))

    for r in resList:
        print("--------------"+r+"----------------")
        print(allEdges[r].value_counts(normalize=True))
    #print(pd.crosstab(allEdges["ReactomeLoc"], allEdges["ComPPI_Labels"]))
    #print(pd.crosstab(allEdges["ReactomeLoc"], allEdges["Trained RF Concat Loc"]))
    #print(pd.crosstab(allEdges["ReactomeLoc"], allEdges["Trained No Complexes Loc"]))
    #print(pd.crosstab(allEdges["ReactomeLoc"], allEdges["H Model 3 Clusters"]))
    meltedDF = allEdges.melt(value_vars=resList, var_name="Model", value_name="Prediction")
    #sns.countplot(data=meltedDF,x="Model",hue="Prediction")
    x,y = 'Model', 'Prediction'
    (meltedDF
    .groupby(x)[y]
    .value_counts(normalize=True)
    .mul(100)
    .rename('percent')
    .reset_index()
    .pipe((sns.catplot,'data'), x=x,y='percent',hue=y, hue_order=["Correct","Incorrect","ComPPI miss"],kind='bar'))
    plt.show()

    accList = []
    accList.append(balanced_accuracy_score(allEdges["ReactomeLoc"], allEdges["ComPPI_Labels"], adjusted=True))
    for r in rList:
        adjAcc = balanced_accuracy_score(allEdges["ReactomeLoc"], allEdges[r+" Loc"], adjusted=False)
        accList.append(adjAcc)
    print(accList)
    plt.bar(resList, accList)
    plt.ylabel("Adjusted Accuracy")
    plt.show()

    mccList = []
    mccList.append(matthews_corrcoef(allEdges["ReactomeLoc"], allEdges["ComPPI_Labels"]))
    for r in rList:
        adjAcc = matthews_corrcoef(allEdges["ReactomeLoc"], allEdges[r+" Loc"])
        mccList.append(adjAcc)
    print(mccList)
    plt.bar(resList, mccList)
    plt.ylabel("Multiclass MCC")
    plt.show()
    return


def calcAccuracyByPath(allPaths, colName):
    print("-Results By Pathway-")
    resD = dict()
    for p in allPaths:
        path = allPaths[p]
        counts = path[colName].value_counts(normalize=True)
        for c, val in counts.iteritems():
            if c not in resD:
                resD[c] = []
            resD[c].append(val)
            # if val == 1.0:
            #    print(p + " is all "+c)
    for c in resD:
        print(c, np.mean(resD[c]))
    return

def loadNetworksAsTables(reactomeF):
    allPaths = dict()
    colNames = ["Interactor1", "Edge Type", "Interactor2", "Raw_LocationGO"]
    for line in open(reactomeF, "r"):
        pName = line.strip().split("/")[-1]
        pathDF = pd.read_csv(
            line.strip(), sep="\t", header=None, names=colNames, dtype=str
        )

        # Process locations row into 2 additional columns
        procLocs = []
        numLocs = []
        for i, r in pathDF.iterrows():
            if pd.isna(r["Raw_LocationGO"]):
                procLocs.append("NULL")
                numLocs.append(0)
                continue
            locs = r["Raw_LocationGO"].split(",")
            eLocs = []
            eCount = 0
            for l in locs:
                if l == "NULL" or l in eLocs:
                    continue
                eCount += 1
                eLocs.append(l)
            if eCount == 0:
                procLocs.append("NULL")
                numLocs.append(0)
                continue
            procLocs.append(",".join(eLocs))
            numLocs.append(eCount)
        pathDF["LocationGO"] = procLocs
        pathDF["LocationCount"] = numLocs
        #if len(pathDF) >= 4 and len(pathDF)<1000:
        if len(pathDF) >= 4:
            allPaths[pName] = pathDF
    return allPaths


def getComPPILabels(pathDF, comPPIDict, nameDict):
    # Guess locations w/ max
    locList = []
    numTie = 0
    numMiss = 0
    numTotal = 0
    for index, row in pathDF.iterrows():
        e1 = row["Interactor1"]
        e2 = row["Interactor2"]
        if e1 not in nameDict or e2 not in nameDict:
            locList.append("Miss")
            numMiss += 1
            numTotal += 1
            continue
        p1 = nameDict[e1]
        p2 = nameDict[e2]
        if p1 not in comPPIDict or p2 not in comPPIDict:
            locList.append("Miss")
            numMiss += 1
            numTotal += 1
            continue
        numTotal += 1
        loc, isTie = getMaxLoc(comPPIDict[p1], comPPIDict[p2])
        locList.append(loc)
        if isTie:
            numTie += 1
    #print(
    #    "Percent reactome prots not in comPPI: ", 100 * float(numMiss) / float(numTotal)
    #)
    pathDF["ComPPI_Labels"] = locList
    return float(numMiss)/float(numTotal)

"""
We also add a column with mapped reactome names in this method
"""
def mapReactomeNames(allPaths, locNameMap):
    for path in allPaths:
        rLocsCName = []
        for index, row in allPaths[path].iterrows():
            rLoc = row["LocationGO"].split(",")
            gotMap = False
            for l in rLoc:
                if l in locNameMap:
                    rLocsCName.append(locNameMap[l])
                    gotMap = True
                    break
            if not gotMap:
                print(rLoc)
                rLocsCName.append("cytosol") #TODO: Deal with misses
        allPaths[path]["ReactomeLoc"] = rLocsCName
    return

def compareComPPIReactome(allPaths, locNameMap):
    # Compare comPPI and reactome Data
    for path in allPaths:
        corrCol = []
        for index, row in allPaths[path].iterrows():
            rLoc = row["LocationGO"].split(",")
            cLoc = row["ComPPI_Labels"]
            if cLoc == "Miss":
                corrCol.append("ComPPI miss")
                continue
            if cLoc == "None":
                corrCol.append("Incorrect")
                continue
            gotMap = False
            gotMatch = False
            for l in rLoc:
                if l in locNameMap:
                    gotMap = True
                    if locNameMap[l] == cLoc:
                        gotMatch = True
                        break
            if not gotMap:
                corrCol.append("Haven't mapped GO term")
            elif not gotMatch:
                corrCol.append("Incorrect")
            else:
                corrCol.append("Correct")
        allPaths[path]["Reactome ComPPI Comparison"] = corrCol

"""
We also add a column with mapped reactome names in this method
"""
def compareGModelReactome(allPaths, locColName, resColName):
    for path in allPaths:
        corrCol = []
        for index, row in allPaths[path].iterrows():
            cLoc = row[locColName]
            if cLoc == "Miss":
                corrCol.append("ComPPI miss")
                continue
            rLoc = row["ReactomeLoc"]
            if rLoc=="Miss":
                corrCol.append("GO miss")
            elif rLoc == cLoc:
                corrCol.append("Correct")
            else:
                corrCol.append("Incorrect")
        allPaths[path][resColName] = corrCol
    return

def loadComPPIGOMap(locNameMapF):
    nameDict = dict()
    f = open(locNameMapF,"r")
    yamFile = yaml.load(f, Loader=yaml.FullLoader)
    f.close()
    for loc in yamFile['largelocs']:
        placeList = yamFile['largelocs'][loc]['childrenIncluded']
        for go in placeList:
            goNum = go.split(":")[-1]
            nameDict[goNum] = loc
    return nameDict

def loadGModelLabels(gResF):
    gResDict = dict()
    currPath = ""
    for line in open(gResF):
        lineList = line.strip().split("\t")
        if len(lineList) == 1:
            currPath = line.split("/")[-1].strip().split("_")[-1]
            gResDict[currPath] = dict()
            continue
        e1 = lineList[0].strip()
        e2 = lineList[1].strip()
        loc = lineList[2].strip()

        # Sorting these would probably be a bit more efficient
        gResDict[currPath][e1 + e2] = loc
        gResDict[currPath][e2 + e1] = loc
    return gResDict


def getGModelLabels(colName, pathDF, gResDict):
    locList = []
    numMiss = 0
    numTotal = 0
    for index, row in pathDF.iterrows():
        e1 = row["Interactor1"].strip()
        e2 = row["Interactor2"].strip()
        if (e1 + e2) in gResDict:
            loc = gResDict[e1 + e2]
            locList.append(loc)
        else:
            locList.append("Miss")
            numMiss += 1
        numTotal += 1
    #print("Basic G Model Miss Percent", float(numMiss) / numTotal)
    pathDF[colName] = locList
    return


def loadComPPI(localF):
    # Make Localization db
    localInfoDF = pd.read_csv(localF, sep="\t")
    localInfoDict = dict()
    c1 = "Major Loc With Loc Score"
    c2 = "Protein Name"
    for index, row in localInfoDF.iterrows():
        locList = []
        locs = row[c1]
        locs = locs.split("|")
        for loc in locs:
            locS = loc.split(":")
            locList.append((locS[0], float(locS[1])))
        localInfoDict[row[c2]] = sorted(locList)
    return localInfoDict


def makeRFLabel(allPaths, comPPIDF, colName="randomForestPrediction"):
    allEdges = pd.concat(allPaths.values())
    allEdges = allEdges[allEdges["Trained No Complexes Loc"] != "Miss"]
    featsList = []
    targets = []

    for ind,row in allEdges.iterrows():
        i1 = row["Interactor1"]
        i2 = row["Interactor2"]

        if (i1 not in comPPIDF.index) or (i2 not in comPPIDF.index):
            print("Miss on ",i1, i2)
            continue

        com1 = comPPIDF.loc[i1]
        com2 = comPPIDF.loc[i2]

        com1.index = "N1_" + com1.index
        com2.index = "N2_" + com2.index

        feats = com1.append(com2)
        featsList.append(feats)
        targets.append(row["ReactomeLoc"])
    featureDF = pd.DataFrame(featsList)
    X = featureDF.values
    y = targets

    rf = RandomForestClassifier()
    scores = cross_val_score(rf, X, y, cv=5)

    print("Random Forest Accuracy:", np.mean(scores))
    print(scores)
    return

def saveComPPIProtTable(comPPIDict, nameDict):

    # Invert nameDict
    nameDict = {v: k for k, v in nameDict.items()}

    allLocs = sorted(
        [
            "nucleus",
            "secretory-pathway",
            "extracellular",
            "cytosol",
            "membrane",
            "mitochondrion",
        ]
    )

    dfColumns = dict()
    nameList = []
    for loc in allLocs:
        dfColumns[loc] = []

    for p in comPPIDict:
        if p in nameDict:
            nameList.append(nameDict[p])
        else:
            continue
        locList = comPPIDict[p]
        locInd = 0
        for loc in allLocs:
            if (locInd < len(locList)) and (loc == locList[locInd][0]):
                dfColumns[loc].append(locList[locInd][1])
                locInd += 1
            else:
                dfColumns[loc].append(0.01)  # Add laplace-ish counts
                #dfColumns[loc].append(0.00)  # Add laplace-ish counts

    comPPIDF = pd.DataFrame(dfColumns, index=nameList)
    comPPIDF = comPPIDF.div(comPPIDF.sum(axis=1), axis=0)  # Normalize rows
    #comPPIDF.to_csv("comPPINodesNoNorm.tsv", sep="\t", index=True)
    return comPPIDF


def getMaxLoc(l1, l2):
    percList = []
    i1 = 0
    i2 = 0
    while i1 < len(l1) and i2 < len(l2):
        if l1[i1][0] == l2[i2][0]:  # Location match
            p = l1[i1][1] * l2[i2][1]
            percList.append((p, l1[i1][0]))
            i1 += 1
            i2 += 1
        else:
            if l1[i1][0] < l2[i2][0]:
                i1 += 1
            else:
                i2 += 1
    if len(percList) == 0:
        return "None", False
    else:
        percList = sorted(percList, reverse=True)
        isTie = False
        if len(percList) > 1 and percList[0][0] == percList[1][0]:
            isTie = True
        return percList[0][1], isTie


def loadIDMap(idMapFile):
    nameDict = dict()
    for line in open(idMapFile):
        lineList = line.strip().split()
        nameDict[lineList[1]] = lineList[0]
    return nameDict


main()
