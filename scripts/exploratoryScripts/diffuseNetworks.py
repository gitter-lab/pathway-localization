import networkx as nx
from sys import argv
import numpy as np
import pandas as pd


def main():
    networksFile = argv[1]
    locDataFile = argv[2]
    outPref = argv[3]

    #Read loc data file
    locDF = pd.read_csv(locDataFile,sep="\t")
    locDict = locDF.set_index("uniprot").to_dict(orient="index")
    for netF in open(networksFile):
        netF = netF.strip()
        diffLocData = dict()
        eList = []
        for line in open(netF):
            lineList = line.strip().split()
            i1 = lineList[0].strip()
            i2 = lineList[2].strip()
            if i1 not in diffLocData:
                if i1 not in locDict:
                    diffLocData[i1] = dict()
                else:
                    diffLocData[i1] = locDict[i1].copy()
            if i2 not in diffLocData:
                if i2 not in locDict:
                    diffLocData[i2] = dict()
                else:
                    diffLocData[i2] = locDict[i2].copy()

            e = i1+"\t"+i2
            eList.append(e)
        if len(eList) < 3:
            print(netF, " is too small")
            continue
        n = nx.parse_edgelist(eList)
        if not nx.algorithms.components.is_connected(n):
            print(netF, " is not connected")
            continue

        #Calculate new feature sums
        print(netF)
        for node1 in n.nodes():
            diffDict = nx.algorithms.centrality.current_flow_betweenness_centrality_subset(n, sources=[node1],targets=n.nodes(),solver='full')
            for node2 in n.nodes():
                if node1==node2:
                    continue
                flowAmt = np.abs(diffDict[node2])
                if node1 in locDict and node2 not in locDict:
                    for loc in locDict[node1]:
                        if loc in diffLocData[node2]:
                            diffLocData[node2][loc] += flowAmt*locDict[node1][loc]
                        else:
                            diffLocData[node2][loc] = flowAmt*locDict[node1][loc]
        #Normalize values
        for node in diffLocData:
            runSum = 0.0
            for loc in diffLocData[node]:
                runSum += diffLocData[node][loc]
            for loc in diffLocData[node]:
                diffLocData[node][loc] = diffLocData[node][loc]/runSum
        #Save as table
        endTbl = pd.DataFrame.from_dict(diffLocData,orient="index")
        endTbl = endTbl.reset_index()
        endTbl = endTbl.rename(columns = {"index":"uniprot"})
        f = outPref+netF.split("/")[-1]
        endTbl.to_csv(outPref+netF.split("/")[-1],sep="\t",index=False)
    return
main()
