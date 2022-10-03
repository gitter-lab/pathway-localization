[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7140733.svg)](https://doi.org/10.5281/zenodo.7140733)

Scripts, data, and supplementary information for "Graph algorithms for predicting subcellular localization at the pathway level".

Supplementary text, figures, and tables can be found in `supplement.pdf`

## Setting up environment

Instructions for setting up a conda environment are coming soon.

## Recreating plots on archived results

The files in the `databasePrediction` directory can be used to recreate plots from the manuscript.
First, extract the file in `allRes.tar` into the results directory:

`tar -xvzf allRes.tar`

Then run:

`python plotResults.py results/allRes.p`

## Datasets
Reactome data were from version 72 retrieved on June 26, 2020. These files were licensed with the [CC0 license](https://reactome.org/license).

[Pathway Commons](http://www.pathwaycommons.org/) data were from version 12 retrieved on June 26, 2020. These files have the same licenses as the original source databases.

ComPPI data were retrieved on November 9, 2022. These files were licensed with the [CC BY-SA 4.0 license](https://comppi.linkgroup.hu/help/terms_of_use).

Compartments data were retrieved on September 29, 2021. These files were licensed with the [CC BY 4.0 license](https://compartments.jensenlab.org/Downloads).

UniProt data were licensed with the [CC BY 4.0 license](https://www.uniprot.org/help/license).

## License
All software (files matching the patterns `Snakefile`,`*.py`, and `*.sh`) is available under a MIT licenese ([`LICENSE-MIT.txt`](LICENSE-MIT.txt)).

All other files (including data, figures, and the supplementary information) are availble under a CC BY 4.0 license ([`LICENSE-CC-BY.txt`](LICENSE-CC-BY.txt)) except for those derived from ComPPI.
Files derived from ComPPI (`databasePrediction/data/comPPINodes.tsv`) are available under a [CC BY-SA 4.0 license](https://creativecommons.org/licenses/by-sa/4.0/) in accordance with the ComPPI [terms of use](https://comppi.linkgroup.hu/help/terms_of_use).
