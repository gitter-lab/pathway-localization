# This script unzips the included data files into ../data/ if the first argument does not exist as a file
FILE=$1

if [ ! -f "${FILE}" ]; then
    # All pathbank pathways with localization annotations and merged complexes
    unzip -u ../data/labeledPathBank.zip -d ../data/

    # All reactome pathways with localization annotations and merged complexes
    unzip -u ../data/labeledReactomeNoComplexes.zip -d ../data/

    # ~100 Reactome pathways belonging to the 'developmental' top category,
    # used for hyperparameter tuning. Was also used for model development and
    # selection (which is why it has its own folder).
    unzip -u ../data/labeledReactomeDevNoComplexes.zip -d ../data/

    # Add a local txt file which indicates to snakemake that we're done
    # unpacking data.
    echo "TRUE" >> ${FILE}
fi
