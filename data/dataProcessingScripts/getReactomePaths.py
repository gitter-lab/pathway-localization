"""
Author: Chris Magnano
05/16/18
Modified to get Cellular Locations: 06/12/20

This is a one-off script for getting and parsing all of the reactome pathways sifs with labeled cellular locations.

Cellular locations are based on cellular component GO terms.

It will also output a table

It expects the file of heirarchical pathway structures from https://reactome.org/download/current/ReactomePathwaysRelation.txt and the table of "top level" pathways in reactome found from https://reactome.org/content/schema/objects/TopLevelPathway?speciesTaxId=9606 (edited into a tsv table).

Usage: python getReactomePaths.py topLevelFile.tsv pathwaysRelation.txt outputDirectory
"""
import sys
import requests
import time
import json
import csv
import pickle as pkl
import os

#GLOBAL HSA DICT
reactionDict = dict()

def main():
    reactomeNamesF = sys.argv[1]
    outDir = sys.argv[2]

    locNameMap = dict()
    #allPaths, delList =getPathwayFiles(reactomeNamesF)
    #allData={"allPaths":allPaths,"delList":delList}
    #with open("reactomeDump.pkl","wb") as outF:
    #    pkl.dump(allData, outF)
    infile = open("reactomeDump.pkl","rb")
    allData = pkl.load(infile)
    allPaths = allData["allPaths"]
    delList = allData["delList"]
    allPathsWithLocs = dict()
    print("Got all pathways, getting locations")
    i=0
    for path in allPaths:
        if path in delList or os.path.isfile(outDir+path+".txt"):
            print("Skipping ",path," due to no pathway or already have it")
            continue
        print("Getting locations for "+path+": ", end='', flush=True)
        allPathsWithLocs[path],locMap = getPathwayLocs(path, allPaths[path])
        for l in locMap:
            if l not in locNameMap:
                locNameMap[l] = locMap[l]
        print("Done")
        i+=1
        if i%10==0:
            print("Reached "+str(i)+" pathways, saving output.")
            saveOutputAndMap(allPathsWithLocs, outDir, delList, locNameMap)
            allPathsWithLocs=dict()
    saveOutputAndMap(allPathsWithLocs, outDir, delList, locNameMap)
    return


def saveOutputAndMap(allPathsWithLocs, outDir, delList, locNameMap):
    for path in allPathsWithLocs:
        if path in delList:
            continue
        pathF = open(outDir+path+".txt","w")
        pathF.write(allPathsWithLocs[path])
        pathF.close()
    mapF = open(outDir + "goNameMap.tsv", "w")
    w = csv.writer(mapF, delimiter="\t")
    w.writerow(["GO_TERM","Name"])
    for k,v in locNameMap.items():
        w.writerow([k,v])
    mapF.close()
    global reactionDict
    rD = open("reactionDict","wb")
    pkl.dump(reactionDict, rD)
    rD.close()
    return

def getPathwayFiles(reactomeNamesF):
    allPaths=dict()

    #Load all reactome pathway names
    for line in open(reactomeNamesF):
        allPaths[line.strip().split()[0]] = ""

    #Get pathways from pathwaycommons via web API
    reqTxt = "http://www.pathwaycommons.org/pc2/get"
    reqParams= {"uri":"http://identifiers.org/reactome/","format":"TXT"}
    idURI="http://identifiers.org/reactome/"
    delList = [] #List of errors to skip

    for path in allPaths:
        print(path)
        reqParams["uri"] = idURI+path

        #Pathway commons reccomends using post, download can be slow so wait up to 10 minutes
        try:
            r = requests.post(reqTxt,params=reqParams,timeout=600.00)
        except requests.exceptions.RequestException as e:
            print("Got exception ",e)
            print("\n Removing pathway "+path+" from analysis and continuing")
            delList.append(path)
            time.sleep(0.4)
            continue

        allPaths[path] = r.text
        #Pathway commons warns not to do multiple per second.
        #Annoying, but needed to prevent a IP address ban.
        time.sleep(0.4)
    return allPaths, delList

"""
Takes in a pathway ID and text file.
Returns a pathway commons-style SIF file with locations as a 4th column.
A single reaction can have multiple location, seperated by commas
"""
def getPathwayLocs(pathHSA,pathwayTXT):
    newPathLines = []
    locNameMap = dict()
    for line in pathwayTXT.split("\n")[1:]:
        if len(line)==0:
            break
        lineList = line.strip().split("\t")
        r1 = lineList[0]
        eType = lineList[1]
        r2 = lineList[2]
        mediatorIDs = lineList[-1].split(";")

        loc=""
        checkComplex = True
        if eType!="in-complex-with":
            checkComplex = False
            print("r", end="",flush=True)
            eHSAs = []
            for m in mediatorIDs:
                pcID = m.split("/")[-1]
                if pcID.startswith("R-HSA"):
                    #if len(eHSAs)>0:
                    #    print("pathway has multiple identifiers: ",pathHSA,line)
                    eHSAs.append(pcID)
            if len(eHSAs)==0:
                for m in mediatorIDs:
                    if "complex" in m or "Complex" in m:
                        print(".",end="",flush=True)
                        checkComplex=True
                if not checkComplex:
                    print("non-complex interaction had no identifers in: ",pathHSA,lineList[0],lineList[1],lineList[2])
                continue
            hasLoc = True
            locs = []
            for eHSA in eHSAs:
                locTuple = getLocFromHSA(eHSA)
                if locTuple is None:
                    continue
                if locTuple[1] not in locNameMap:
                    locNameMap[locTuple[1]] = locTuple[0]
                locs.append(locTuple[1])
            loc = ",".join(locs)
        if checkComplex:
            print("c", end="",flush=True)
            locTuples = getComplexLoc(mediatorIDs)
            locs = []
            for l in locTuples:
                if l[1] not in locNameMap: #It'd be nice to prefer the reactome names over pathwayCommons
                    locNameMap[l[1]] = l[0]
                locs.append(l[1])
            loc = ",".join(locs)

        newLine = "\t".join([r1,eType,r2,loc])
        newPathLines.append(newLine)

    pathWithLocs = "\n".join(newPathLines)
    return pathWithLocs, locNameMap

def getLocFromHSA(hsa):
    global reactionDict
    if hsa in reactionDict:
        return reactionDict[hsa]
    reqTxt = "https://reactome.org/ContentService/data/query/"+hsa
    try:
        r = requests.get(reqTxt,timeout=600.00)
    except requests.exceptions.RequestException as e:
        print("Got exception in HSA: ",e)
        time.sleep(0.1)
        return
    jsonOutput = json.loads(r.text)
    time.sleep(0.1)

    #Find compartment field
    if "compartment" not in jsonOutput:
        print("could not find compartment in "+hsa)
        goNum = "NULL"
        name = "NULL"
    else:
        comp = jsonOutput["compartment"][0]
        goNum = comp["accession"]
        name = comp["name"]
        assert comp["databaseName"] == "GO"

    loc = (name, goNum)
    reactionDict[hsa] = loc
    return loc

"""
Gets a complex's reactome identifier ("R-HSA-...") from its mediatorIDs.
Currently not used as we get the location from pathwayCommons for complexes
but leaving it in as that may well and likely change.
"""
def getComplexLoc(mediatorIDs):
    locs = []
    global reactionDict
    for m in mediatorIDs:
        if m in reactionDict:
            locs.append(reactionDict[m])
            continue
        #Get JSON file
        reqTxt = "http://www.pathwaycommons.org/pc2/get"
        reqParams= {"uri":m,"format":"JSONLD"}
        try:
            r = requests.post(reqTxt,params=reqParams,timeout=600.00)
        except requests.exceptions.RequestException as e:
            print("Got exception ",e)
            time.sleep(0.4)
            continue
        jsonOutput = json.loads(r.text)
        time.sleep(0.4)

        #Find line with same ID as mediatorID
        graphPart = jsonOutput["@graph"]
        locDict = dict()
        locID = ""
        for part in graphPart:
            if part["@type"]=="bp:CellularLocationVocabulary":
                if isinstance(part["term"],str):
                    partName = part["term"]
                else:
                    partName = part["term"][1]
                locDict[part["@id"]]=(partName,part["xref"].split("_")[-1])
            elif part["@id"]==m:
                #TODO: Here's where we'd also grab the reactome hsa ID if we need it
                locID = part["cellularLocation"]

        assert len(locID)>0, "Could not find complexID or location within JSON"
        assert locID in locDict, "Location ID was not in JSON"

        locs.append(locDict[locID])
        reactionDict[m] = locDict[locID]
    return locs
main()
