#Args are networks features model output oDir
python prepPGMData.py $1 $2 $5

for i in {0..4};
do
    pgm/train_nets ${5}/trainFold${i}.txt ${5}/testFold${i}.txt $2 5 3 $5/;
done;

rm ${4};
for i in `ls ${5}/trainedModel*`; do echo $i >> ${4}; cat $i >> ${4}; done;

#grep -qxF ${1}.txt modelFiles.txt || echo ${1}.txt >> modelFiles.txt;

#python compareReactomeComPPI.py allPathsComplexes.txt ../../data/comPPI/comppi--proteins_locs--tax_hsapiens_loc_all.txt ../../data/comppi_uniprot_map.txt ../../data/comPPIGOMap.yml ../../data/reactomeLocalizations/goNameMap.tsv basicModelOutput.txt modelFiles.txt;
#python compareReactomeComPPI.py allDevPaths.txt ../../data/comPPI/comppi--proteins_locs--tax_hsapiens_loc_all.txt ../../data/comppi_uniprot_map.txt ../../data/comPPIGOMap.yml ../../data/reactomeLocalizations/goNameMap.tsv basicModelOutput.txt modelFiles.txt;
