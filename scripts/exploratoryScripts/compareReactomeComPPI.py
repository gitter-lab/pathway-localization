import pandas as pd
from sys import argv
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl

def main():
    localF = argv[1]
    comPPIF = argv[2]
    idMapFile = argv[3]
    locNameMap = argv[4]

    #Returns a dict of gene name to uniprot
    nameDict = loadIDMap(idMapFile)
    #Load Reactome Mapping Data
    #allPaths = loadNetworksAsTables(localF)
    #pF = open("tmpReact.pkl","wb")
    #pkl.dump(allPaths,pF)
    #pF.close()
    pF = open("tmpReactComPPI.pkl","rb")
    allPaths = pkl.load(pF)
    pF.close()
    #Load comPPI data
    #comPPIDict = loadComPPI(comPPIF)
    #for path in allPaths.values():
    #    getComPPILabels(path, comPPIDict, nameDict)
    #Look at Reactome distribution
    #pF = open("tmpReactComPPI.pkl","wb")
    #pkl.dump(allPaths,pF)
    #pF.close()
    analyzeReactomeData(allPaths, locNameMap)
    return

def analyzeReactomeData(allPaths, locNameMapF):
    locNameMap = dict()
    for line in open(locNameMapF):
        lineList = line.strip().split()
        locNameMap[lineList[1].strip()] = lineList[0].strip()
    allEdges = pd.concat(allPaths.values())

    #Look at reactome data alone
    #print("Mean: ",allEdges["LocationCount"].mean())
    #print(allEdges["Edge Type"].value_counts())
    #print(allEdges["Edge Type"].value_counts(normalize=True))

    #print(allEdges["LocationCount"].value_counts())
    #print(allEdges["LocationGO"].value_counts())
    #print(allEdges["LocationGO"].value_counts(normalize=True).to_string())
    #print((allEdges["Edge Type"][allEdges["LocationCount"]==1]).value_counts(normalize=True))
    #print((allEdges["Edge Type"][allEdges["LocationCount"]==1]).value_counts(normalize=False))

    #Compare comPPI and reactome Data
    numCorrect = 0
    numWrong = 0
    numMissMap = 0
    numMissComPPI = 0
    numNoneComPPI = 0
    for index, row in allEdges.iterrows():
        rLoc = row["LocationGO"].split(",")
        cLoc = row["ComPPI_Labels"]
        if cLoc == "Miss":
            numMissComPPI += 1
            continue
        if cLoc == "None":
            numNoneComPPI += 1
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
            numMissMap += 1
        elif not gotMatch:
            numWrong += 1
        else:
            numCorrect += 1
    print("# Correct: ",numCorrect)
    print("# Wrong: ",numWrong)
    print("# No ComPPI Shared Locs: ",numNoneComPPI)
    print("# No ComPPI Prot: ",numMissComPPI)
    print("# No GO Map: ",numMissMap)
    return

def loadNetworksAsTables(reactomeF):
    allPaths = dict()
    colNames = ["Interactor1", "Edge Type", "Interactor2", "Raw_LocationGO"]
    for line in open(reactomeF,"r"):
        pName = line.strip().split("/")[-1]
        pathDF = pd.read_csv(line.strip(), sep="\t", header=None, names=colNames, dtype=str)

        #Process locations row into 2 additional columns
        procLocs = []
        numLocs = []
        for i,r in pathDF.iterrows():
            if pd.isna(r["Raw_LocationGO"]):
                procLocs.append("NULL")
                numLocs.append(0)
                continue
            locs = r["Raw_LocationGO"].split(",")
            eLocs = []
            eCount = 0
            for l in locs:
                if l=="NULL" or l in eLocs:
                    continue
                eCount += 1
                eLocs.append(l)
            if eCount==0:
                procLocs.append("NULL")
                numLocs.append(0)
                continue
            procLocs.append(",".join(eLocs))
            numLocs.append(eCount)
        pathDF["LocationGO"] = procLocs
        pathDF["LocationCount"] = numLocs
        if len(pathDF) >= 4:
            allPaths[pName] = pathDF
    return allPaths


def getComPPILabels(pathDF, comPPIDict, nameDict):
    #Guess locations w/ max
    locList = []
    numTie = 0
    numMiss = 0
    numTotal = 0
    for index, row in pathDF.iterrows():
        e1 = row["Interactor1"]
        e2 = row["Interactor2"]
        if e1 not in nameDict or e2 not in nameDict:
            locList.append("Miss")
            numMiss+=1
            numTotal+=1
            continue
        p1 = nameDict[e1]
        p2 = nameDict[e2]
        if p1 not in comPPIDict or p2 not in comPPIDict:
            locList.append("Miss")
            numMiss+=1
            numTotal+=1
            continue
        numTotal+=1
        loc,isTie = getMaxLoc(comPPIDict[p1],comPPIDict[p2])
        locList.append(loc)
        if isTie:
            numTie+=1
    print("Percent Misses: ",float(numMiss)/float(numTotal))
    pathDF["ComPPI_Labels"] = locList
    return

def loadComPPI(localF):
    #Make Localization db
    localInfoDF = pd.read_csv(localF,sep="\t")
    localInfoDict = dict()
    c1 = "Major Loc With Loc Score"
    c2 = "Protein Name"
    for index, row in localInfoDF.iterrows():
        locList = []
        locs = row[c1]
        locs = locs.split("|")
        for loc in locs:
            locS = loc.split(":")
            locList.append((locS[0],float(locS[1])))
        localInfoDict[row[c2]]=sorted(locList)
    return localInfoDict

def getMaxLoc(l1,l2):
    percList = []
    i1 = 0
    i2 = 0
    while i1<len(l1) and i2<len(l2):
        if l1[i1][0] == l2[i2][0]: #Locaction match
            p = l1[i1][1]*l2[i2][1]
            percList.append((p,l1[i1][0]))
            i1+=1
            i2+=1
        else:
            if l1[i1][0]<l2[i2][0]:
                i1+=1
            else:
                i2+=1
    if len(percList)==0:
        return "None",False
    else:
        percList = sorted(percList,reverse=True)
        isTie = False
        if len(percList)>1 and percList[0][0]==percList[1][0]:
            isTie=True
        return percList[0][1],isTie

def loadIDMap(idMapFile):
    nameDict = dict()
    for line in open(idMapFile):
        lineList = line.strip().split()
        nameDict[lineList[1]] = lineList[0]
    return nameDict
main()
