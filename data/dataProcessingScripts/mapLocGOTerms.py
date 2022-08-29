from sys import argv
import yaml
import os

"""
This script maps the GO terms in the network files to comPPI location names.
"""

def main():
    nameFile = argv[1]
    inNet = argv[2]
    outDir = argv[3]

    nameDict = loadComPPIGOMap(nameFile)
    newLines = []
    for line in open(inNet):
        lineList = line.strip().split()
        goLocs = lineList[3].split(",")
        rLoc = ""
        gotMap = False
        for l in goLocs:
            if l in nameDict:
                rLoc = nameDict[l]
                gotMap = True
                break
        if not gotMap:
            print(goLocs)
            continue
            #rLoc = "cytosol" #TODO: Deal with misses
        lineList[3] = rLoc
        lineList.append("\n")
        newLines.append("\t".join(lineList))
    outFName = os.path.join(outDir, os.path.basename(inNet))
    outF = open(outFName, "w")
    outF.writelines(newLines)
    outF.close()
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

main()
