dgm/netDiffuse/bin/train_nets allDevPaths.txt allDevPaths.txt ../../data/missDiffCompPPI 5 3 $2;
ls output/pots_R-HSA-* >> ${2}allPotsFiles.txt;
python dgmKmeans.py ${2}allPotsFiles.txt $1 $2;

for i in `ls ${2}cluster*of${1}.txt`;
    do dgm/netDiffuse/bin/train_nets $i $i ../../data/missDiffCompPPI 5 3 $2;
done;

echo "Done second train";

for i in `ls ${2}trainedModel*`; do echo $i >> diffusedKmeans${1}.txt; cat $i >> diffusedKmeans${1}.txt; done;

ls diffusedKmeans${1}* >> modelFiles.txt;

#python compareReactomeComPPI.py allDevPaths.txt ../../data/comPPI/comppi--proteins_locs--tax_hsapiens_loc_all.txt ../../data/comppi_uniprot_map.txt ../../data/comPPIGOMap.yml ../../data/reactomeLocalizations/goNameMap.tsv basicModelOutput.txt modelFiles.txt;
