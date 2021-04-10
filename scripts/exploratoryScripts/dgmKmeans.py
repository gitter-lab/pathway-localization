import sys
from sklearn.cluster import KMeans
import numpy as np


def main():
    fList = sys.argv[1]
    numClust = int(sys.argv[2])
    outPref = sys.argv[3]
    potList = []
    potMat = []
    for potFile in open(fList):
        potFile = potFile.strip()
        pot = []
        for line in open(potFile):
            lineList = line.strip().split()
            if len(lineList) > 1:
                break
            pot.append(float(lineList[0]))
        if np.any(np.isnan(pot)):
            continue
        potMat.append(pot)
        potList.append(potFile)
    res = KMeans(n_clusters=numClust).fit(potMat)
    fOutList = []
    for i in range(numClust):
        fOutList.append([])
    for i in range(len(potList)):
        pathName = "../../data/labeledReactomeNoComplexes/"+potList[i].split("_")[-1]
        fOutList[res.labels_[i]].append(pathName+"\n")
    for i in range(numClust):
        outF = outPref + "cluster"+str(i)+"of"+str(numClust)+".txt"
        outFS = open(outF, "w")
        outFS.writelines(fOutList[i])
        outFS.close()
    return
main()
