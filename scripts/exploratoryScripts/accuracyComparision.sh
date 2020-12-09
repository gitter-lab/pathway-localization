
#Looks at baseline accuracy of all reactome edges
#python compareReactomeComPPI.py ../../data/reactomeLocFileNames.txt ../../data/comPPI/comppi--proteins_locs--tax_hsapiens_loc_all.txt ../../data/comppi_uniprot_map.txt ../../data/comppi_GO_nameMap.txt ../../data/reactomeLocalizations/goNameMap.tsv

#Looks at basic graphical model accuracy of all developmental edges
python compareReactomeComPPI.py allDevPaths.txt ../../data/comPPI/comppi--proteins_locs--tax_hsapiens_loc_all.txt ../../data/comppi_uniprot_map.txt ../../data/comPPIGOMap.yml ../../data/reactomeLocalizations/goNameMap.tsv basicModelOutput.txt

