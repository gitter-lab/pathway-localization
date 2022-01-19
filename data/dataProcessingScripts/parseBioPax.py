import pybiopax
from sys import argv
import pandas as pd
import numpy as np
import networkx as nx
import os, sys

from tqdm import tqdm
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


def main():
    outPref='../labeledPathBankComID/'
    locNameMap = dict()
    for line in open('../pathBank/locMap.csv'):
        lineList = line.split(',')
        pbName = lineList[0].strip()
        outName = lineList[1].strip()
        locNameMap[pbName] = outName

    #Get all entitys from PC
    pcDF = pd.read_csv('../pathBank/PathwayCommons12.entityRef.txt', index_col='UNIFICATION_XREF',sep='\t')
    pcDF.index = pcDF.index.str.split(':').str[-1]

    #Get all pathways from file
    #for line in open('../pathBank/PathwayCommons12.pathbank.hgnc.txt',sep='\t'):
    pathsDF = pd.read_csv('../pathBank/PathwayCommons12.pathbank.hgnc.txt',sep='\t')
    pathsDict = dict()
    for index,row in pathsDF.iterrows():
        pathNames = row['PATHWAY_NAMES']
        if pd.isna(pathNames):
            continue
        pathNames = pathNames.split(';')
        i1 = row['PARTICIPANT_A']
        i2 = row['PARTICIPANT_B']
        iType = row['INTERACTION_TYPE']
        medIDs = row['MEDIATOR_IDS']
        for pathN in pathNames:
            if not pathN in pathsDict:
                pathsDict[pathN] = []
            pathsDict[pathN].append([i1,iType,i2,'',medIDs])
    print('Found this many pathways: ',len(pathsDict))


    #Load pathbank name mapping file
    prot_DF = pd.read_csv('../pathBank/pathbank_all_proteins.csv',index_col='Protein Name') #Uniprot ID
    met_DF = pd.read_csv('../pathBank/pathbank_all_metabolites.csv',index_col='Metabolite Name') #ChEBI ID
    prot_DF = prot_DF[~prot_DF.index.duplicated(keep='first')]
    met_DF = met_DF[~met_DF.index.duplicated(keep='first')]

    fOutDict = dict()

    for line in open("goodPathBanks.txt"):
        modelF = line.strip()
        getBioPaxInfo(modelF, pathsDict, prot_DF, met_DF, pcDF, fOutDict, locNameMap, outPref)
    for outFN in fOutDict:
        eList = fOutDict[outFN]
        outF = open(outFN, 'w')
        for newE in eList:
            outL = "\t".join(newE)+"\n"
            #print(outL)
            outF.write(outL)
        outF.close()
    return


def getBioPaxInfo(modelF, pathsDict, prot_DF, met_DF, pcDF, fOutDict, locNameMap, outPref):
    pathsLocDict = dict()
    model = None

    #Load pw while supressing prints
    model = pybiopax.model_from_owl_file(modelF)
    pathwayName = ""

    #Check to make sure this is a pathway that we want
    for sp in model.get_objects_by_type(pybiopax.biopax.BioSource):
        if not sp.display_name=='Homo sapiens':
            print('Wrong Species')
            return
    for pa in model.get_objects_by_type(pybiopax.biopax.Pathway):
        pList = str(pa.name).strip('][').split(',')
        if len(pList)==1:
            pathwayName=pList[0].strip('\'')
    if len(pathwayName)==0:
        print('No Pathway Name')
        return
    if pathwayName not in pathsDict:
        print('No match')
        return
    pathList = pathsDict[pathwayName]
    if len(pathList)<4:
        return

    #We found a good pathway
    locDict= dict()
    interDict = dict()
    complexDict = dict()
    for pe in model.get_objects_by_type(pybiopax.biopax.PhysicalEntity):
        peName = str(pe)
        isProt = False
        if "Protein(" in peName:
            isProt = True
        peName = peName.split("(",maxsplit=1)[-1][:-1]
        peLoc = str(pe.cellular_location).split("(",maxsplit=1)[-1][:-1].strip("\"")
        if (peLoc=='Non'):
            continue
        peLoc = locNameMap[peLoc.strip()]

        #Get information on complexes and interactions
        peInterSet = pe.participant_of
        peInterList = []
        if isinstance(peInterSet, set):
            for inter in peInterSet:
                peInterList.append(inter.uid)
        else:
            peInterList = [peInterSet.uid]

        #peComplexSet = pe.component_of
        #peComplexList = []
        #if isinstance(peComplexSet, set):
        #    for inter in peComplexSet:
        #        peComplexList.append(inter.uid)
        #else:
        #    peComplexList = [peComplexSet.uid]
        #if peName not in complexDict:
        #    complexDict[peName] = dict()
        #for compl in peComplexList:
        #    complexDict[peName][compl] = peLoc

        #Do all the name mapping
        uniprotName = ''
        try:
            if isProt:
                uniprotName = prot_DF.loc[peName]['Uniprot ID']
            else:
                uniprotName = str(met_DF.loc[peName]['ChEBI ID'])
                uniprotName = uniprotName[:-2]
        except KeyError:
            #print('Miss: ',pe)
            continue
        if pd.isna(uniprotName):
            continue
        try:
            mappedName = pcDF.loc[uniprotName]['PARTICIPANT']
        except KeyError:
            #print('UMiss:',peName)
            continue
        if isinstance(mappedName, pd.Series):
            for mN in list(mappedName):
                pathsLocDict[mN]=peLoc
            continue
        if mappedName not in interDict:
            interDict[mappedName] = dict()
        for peInter in peInterList:
            interDict[mappedName][peInter] = peLoc
        if mappedName in pathsLocDict:
            if(pathsLocDict[mappedName]=='membrane') or (peLoc=='membrane'):
                pathsLocDict[mappedName] = 'membrane'
        else:
            pathsLocDict[mappedName]=peLoc
    newEList = []
    for e in pathList:
        eLoc = get_edge_loc(e, pathsLocDict, interDict, complexDict)
        if eLoc=='':
            continue
        newE = list(e)
        newE[3] = eLoc
        newEList.append(newE)

    pathwayName = pathwayName.replace('/','-')
    outFN = outPref+"_".join(pathwayName.split())+'.txt'
    if not outFN in fOutDict:
        fOutDict[outFN] = []
    else:
        pass
        #print("Duplicate Pathways:",len(newEList),len(fOutDict[outFN]))
    if len(fOutDict[outFN])==0 or len(fOutDict[outFN])>len(newEList):
        fOutDict[outFN] = list(newEList)
    return

def get_edge_loc(edge, pathsLocDict, interDict, complexDict):
    i1 = edge[0]
    eT = edge[1]
    i2 = edge[2]
    loc1 = ''
    loc2 = ''
    eLoc = ''
    #First see if they got the same interaction somewhere
    foundLoc = False
    if i1 in interDict and i2 in interDict:
        inter1 = interDict[i1].keys()
        inter2 = interDict[i2].keys()
        for inter in inter1:
            if inter in inter2:
                eLoc = interDict[i1][inter]
                foundLoc = True
    ##Then see if they are in the same complex anywhere
    #if i1 in complexDict and i2 in complexDict:
    #    complex1 = complexDict[i1].keys()
    #    complex2 = complexDict[i2].keys()
    #    for compl in complex1:
    #        if compl in complex2:
    #            eLoc = complexDict[i1][compl]
    #            foundLoc = True
    #If not see how other localizations agree
    if not foundLoc:
        if i1 in pathsLocDict:
            loc1 = pathsLocDict[i1]
        if i2 in pathsLocDict:
            loc2 = pathsLocDict[i2]
        if loc1==loc2:
            eLoc = loc1
        #We care less about molecules than proteins
        elif 'CHEBI' in i1 and 'CHEBI' in i1:
            if loc1=='':
                eLoc = loc2
            elif loc2=='':
                eLoc = loc1
            else:
                #Let's prioritize non-cytoplasm
                if loc1 == 'cytosol':
                    eLoc=loc2
                elif loc2 == 'cytosol':
                    eLoc=loc1
                else:
                    #print('Ambiguity!',loc1,loc2)
                    eLoc = loc1
        elif loc1=='':
            eLoc = loc2
        elif loc2=='':
            eLoc = loc1
        elif 'CHEBI' in i1:
            eLoc = loc2
        elif 'CHEBI' in i2:
            eLoc = loc1
        else:
            #print('BIG Ambiguity!',loc1,loc2)
            if loc1 == 'cytosol':
                eLoc=loc2
            elif loc2 == 'cytosol':
                eLoc=loc1
            else:
                eLoc = loc1
    return eLoc
main()
