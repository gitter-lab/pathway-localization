import pybiopax
from sys import argv
import pandas as pd
import numpy as np
import networkx as nx


def main():

    locNameMap = dict()
    for line in open('../pathBank/locMap.csv'):
        lineList = line.split(',')
        pbName = lineList[0].strip()
        outName = lineList[1].strip()
        locNameMap[pbName] = outName

    #Get all entitys from PC
    pcDF = pd.read_csv('../pathBank/PathwayCommons12.entityRef.txt', index_col='UNIFICATION_XREF',sep='\t')
    pcDF.index = pcDF.index.str.split(':').str[-1]
    print(pcDF)

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
        for pathN in pathNames:
            if not pathN in pathsDict:
                pathsDict[pathN] = []
            pathsDict[pathN].append([i1,iType,i2])
    print('Found this many pathways: ',len(pathsDict))


    #Load pathbank name mapping file
    prot_DF = pd.read_csv('../pathBank/pathbank_all_proteins.csv',index_col='Protein Name') #Uniprot ID
    met_DF = pd.read_csv('../pathBank/pathbank_all_metabolites.csv',index_col='Metabolite Name') #ChEBI ID
    prot_DF = prot_DF[~prot_DF.index.duplicated(keep='first')]
    met_DF = met_DF[~met_DF.index.duplicated(keep='first')]

    for line in open("../pathBank/allPathBankBioPax.txt"):
        pathsLocDict = dict()
        modelF = line.strip()
        print(modelF)
        model = pybiopax.model_from_owl_file(modelF)
        pName = ""
        for pa in model.get_objects_by_type(pybiopax.biopax.Pathway):
            pList = str(pa.name).strip('][').split(',')
            if len(pList)==1:
                pName=pList[0].strip('\'')
        if len(pName)==0:
            print('No Pathway Name!!')
            continue
        if pName not in pathsDict:
            print('No match')
            continue
        locDict= dict()
        for pe in model.get_objects_by_type(pybiopax.biopax.PhysicalEntity):
            peName = str(pe)
            isProt = False
            if "Protein(" in peName:
                isProt = True
            peName = peName.split("(",maxsplit=1)[-1][:-1]
            peLoc = str(pe.cellular_location).split("(",maxsplit=1)[-1][:-1]
            peLoc = locNameMap[peLoc]
            if (peLoc=='Non'):
                continue
            peInter = str(pe.participant_of)
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
            #print(mappedName,"Uni: ",uniprotName,peName,"DONE")
            if mappedName in pathsLocDict:
                if(pathsLocDict[mappedName]=='membrane') or (peLoc=='membrane'):
                    pathsLocDict[mappedName] = 'membrane'
                #print('DUPLICATE AHH',peLoc, pathsLocDict[mappedName],mappedName)
                continue
            pathsLocDict[mappedName]=peLoc
        for e in pathList:
            i1 = e[0]
            eT = e[1]
            i2 = e[2]
            loc1 = ''
            loc2 = ''
            eLoc = ''
            if i1 in pathsLocDict:
                loc1 = pathsLocDict[i1]
            if i2 in pathsLocDict:
                loc2 = pathsLocDict[i2]
            if loc1==loc2:
                eLoc = loc1
            elif 'CHEBI' in i1 and 'CHEBI' in i1:
                print('Ambiguity!',loc1,loc2)
                eLoc = loc1
            elif 'CHEBI' in i1:
                eLoc = loc2
            elif 'CHEBI' in i2:
                eLoc = loc1
            else:
                print('BIG Ambiguity!',loc1,loc2)
    #1. parse txt file into pathways
    #2. parse owl files into locations
    #3. merge
    return

main()
