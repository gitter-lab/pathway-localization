#Args: networks features model output oDir

mkdir ${5}

mOut="${4}_raw";
if [ "${3}" = "TrainedPGM" ];
then
    python prepPGMData.py $1 $2 $5;

    for i in {0..4};
    do
        pgm/train_nets ${5}/trainFold${i}.txt ${5}/testFold${i}.txt $2 5 3 ${5}/;
    done;

    rm ${mOut};
    for i in `ls ${5}/trainedModel*`; do echo $i >> ${mOut}; cat $i >> ${mOut}; done;

else
    #There is no training on this model so we can just go
    for i in `cat ${1}`;
    do
        pgm/no_training_model ${i} ${2} ${5}/;
    done;

    rm ${mOut};
    for i in `ls ${5}/pgmDirectFeatures*`; do echo $i >> ${mOut}; cat $i >> ${mOut}; done;
fi

python parsePGMOutput.py ${mOut} ${4} ${3} ${1}

