import yaml
from sys import argv


def main():
    mapNameF = argv[1] #yaml file
    compartmentF = argv[2]
    #protNameMapF = argv[3]
    nameMap = loadComPPIGOMap(mapNameF)

    #protNameMap = dict()
    #for line in open(protNameMapF):
    #    lineList = line.strip().split()
    #    protNameMap[lineList[1]] = lineList[-1].split("_")[0]

    protMap = dict()

    for line in open(compartmentF):
        lineList = line.strip().split('\t')
        prot = lineList[1]

        #if prot not in protNameMap:
        #    continue
        #prot = protNameMap[prot]

        locGO = lineList[2]
        conf = lineList[-1]
        locGO = locGO.split(":")[-1]

        if locGO not in nameMap:
            continue
        loc = nameMap[locGO]
        conf = float(conf)

        if prot not in protMap:
            protMap[prot] = dict()
        if loc not in protMap[prot]:
            protMap[prot][loc] = conf
        elif protMap[prot][loc] < conf:
            protMap[prot][loc] = conf

    locList = ["cytosol", "extracellular", "membrane", "mitochondrion", "nucleus", "secretory-pathway"]
    outLines = []
    outLines.append('uniprot\t' + '\t'.join(locList) + "\n")
    for prot in protMap:
        outStr = prot+'\t'
        for loc in locList:
            if loc in protMap[prot]:
                outStr += str(protMap[prot][loc])+'\t'
            else:
                outStr += "0.0\t"
                #print("Miss on "+prot+" for "+loc)
        outLines.append(outStr+"\n")
    outFN = 'compartmentsNodes_2.tsv'
    outF = open(outFN,'w')
    outF.writelines(outLines)
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
