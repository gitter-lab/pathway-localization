"""
12/07/2020

This script takes in a reactome network and condenses all complexes in that network into a single node.
"""
import networkx as nx
import pandas as pd
from sys import argv
import os

def main():
    #Load in network to networkx
    eList = []
    comEList = []
    inLines = []
    delList = []
    comDict = dict()
    netF = argv[1]
    outDir = argv[2]
    for line in open(argv[1]):
        lineList = line.strip().split()
        inLines.append("\t".join(lineList[:-1])+"\n")
        eList.append("\t".join([lineList[0], lineList[2], lineList[3]]))
        if lineList[1] == "in-complex-with":
            comEList.append("\t".join([lineList[0], lineList[2], lineList[3]]))
            cList = lineList[4].split(";")
            for c in cList:
                if "complex" in c or "Complex" in c:
                    if c not in comDict:
                        comDict[c] = set()
                    comDict[c].add(lineList[0])
                    comDict[c].add(lineList[2])


    net = nx.parse_edgelist(eList, data=(("Location", str),))

    #Make copy of network which only has complex edges, and one with none
    comNet = nx.parse_edgelist(comEList, data=(("Location", str),))
    noComNet = net.copy()

    #If the network is only in-complex-with edges, do nothing
    #if len(net.edges())==len(comNet.edges()):
    #    print("All Complexes")
    #    rmLinesAndSave(inLines, delList)
    #    return

    #Loop through all cliques of graph
    allComs = list(comDict.values())
    print(len(allComs), end="", flush=True)
    i=0
    allCSets = []
    for com in allComs:
        noComNet.remove_edges_from(net.subgraph(list(com)).edges()) #Remove all within-complex edges
        allCSets.append(set())
        if len(com) < 3:
            continue
        i += 1
        #if i%10 == 0:
        #    print(i, end="", flush=True)
        #elif i%1 == 0:
        #    print(".", end="", flush=True)
        initSet = False
        for n1 in com:
            nei1 = set(net.neighbors(n1))
            if not initSet:
                allCSets[-1] = set(nei1)
                initSet = True
            else:
                allCSets[-1] = allCSets[-1].intersection(nei1)
            if len(allCSets[-1]) == 0:
                break
        #print("Complex edges in common:", len(allCSets[-1]))

    nNodes = 0
    allDNodes = dict()
    for i in range(len(allComs)):
        com = allComs[i]
        comSet = allCSets[i]
        comDList = []
        for n in com:
            neis = list(noComNet.neighbors(n))
            if set(neis) == comSet:
                if n in allDNodes:
                    continue
                allDNodes[n] = 1
                nNodes += 1
                comDList.append(n)
        delList.append(comDList)
    print("\nTotal Nodes:",len(net), "Complex Redundant Nodes:", nNodes)
    print("Total Edges Before:",len(net.edges()), end="")
    #Remove all-but-one nodes which contain all and only those edges
    linesDelList = []
    for delSet in delList:
        if len(delSet)<2:
            continue
        for i in range(len(delSet)-1):
            net.remove_node(delSet[i])
            for j in range(len(inLines)):
                if delSet[i] in inLines[j] and j not in linesDelList:
                    linesDelList.append(j)

    print(" Total Edges After:",len(net.edges()))

    outNetF = os.path.join(outDir, os.path.basename(netF))
    outNet = open(outNetF, "w")
    for i in range(len(inLines)):
        if i in linesDelList:
            continue
        outNet.write(inLines[i])
    outNet.close()
    #nx.write_edgelist(net, outNetF, delimiter="\t", data=["Location"])
    return


def rmLinesAndSave(lines, delList):
    return
main()
