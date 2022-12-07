[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7140733.svg)](https://doi.org/10.5281/zenodo.7140733)

Scripts, data, and supplementary information for the publication:

[Graph algorithms for predicting subcellular localization at the pathway level](https://doi.org/10.1142/9789811270611_0014).
Chris S Magnano, Anthony Gitter.
*Pacific Symposium on Biocomputing*, 2023.

Supplementary text, figures, and tables can be found in `supplement.pdf`

## Setting up environment

1. Install python/anaconda if needed.

   The easiest way to install Python and the required packages is with [Anaconda](https://www.anaconda.com/download/).
   The Carpentries [Anaconda installation instructions](https://carpentries.github.io/workshop-template/#python) provide guides and videos on how to install Anaconda for your operating system.
   After installing Anaconda, you can run the following commands from the root directory of the `spras` repository

2. Create the conda environment.

   From the pathway-localization directory, run

   ```
   conda env create -f environment.yml
   conda activate pw_loc
   ```
   to create a conda environment with the required packages and activate that environment.
   
3. Install [Ax](https://ax.dev/).
   
   The `ax-platform` package, used for Bayesian optimization within PyTorch, needs to be installed separately using `pip` after activating the environment:
   ```
   pip3 install ax-platform --no-cache-dir
   ```
   
## Recreating plots on archived results

The files in the `databasePrediction` directory can be used to recreate plots from the manuscript.
First, extract the file in `allRes.tar` into the results directory:

```
tar -xvzf allRes.tar
```

Then run:

```
python plotResults.py results/allRes.p
```

## Running localization prediction

After following the steps in `Setting up the environment`, navigating to the directory `databasePrediction` and using the command:

```
snakemake --cores [n] all
```

will run the full localization prediction experiment, where `[n]` is the number of desired cores.

## Datasets
- Reactome data were from version 72 retrieved on June 26, 2020. These files were licensed with the [CC0 license](https://reactome.org/license).
- [Pathway Commons](http://www.pathwaycommons.org/) data were from version 12 retrieved on June 26, 2020. These files have the same licenses as the original source databases. The PathBank files were licensed with the [ODbL v1.0 license](https://www.pathbank.org/about).
- ComPPI data were retrieved on November 9, 2022. These files were licensed with the [CC BY-SA 4.0 license](https://comppi.linkgroup.hu/help/terms_of_use).
- Compartments data were retrieved on September 29, 2021. These files were licensed with the [CC BY 4.0 license](https://compartments.jensenlab.org/Downloads).
- UniProt data were retreieved in October 2021. These files were licensed with the [CC BY 4.0 license](https://www.uniprot.org/help/license).

Any publications using these datasets should cite the original sources of the data.

## Licenses
All software (files matching the patterns `Snakefile`,`*.py`, and `*.sh`) is available under a MIT licenese ([`LICENSE-MIT.txt`](LICENSE-MIT.txt)).

All other files (including data, figures, and the supplementary information) are availble under a CC BY 4.0 license ([`LICENSE-CC-BY.txt`](LICENSE-CC-BY.txt)) except for those derived from PathBank and ComPPI.
Files derived from PathBank (`data/labeledPathBank.zip`) are available under a [ODbL v1.0 license](https://opendatacommons.org/licenses/odbl/1-0/) in accordance with the PathBank [terms of use](https://www.pathbank.org/about).
Files derived from ComPPI (`databasePrediction/data/comPPINodes.tsv`) are available under a [CC BY-SA 4.0 license](https://creativecommons.org/licenses/by-sa/4.0/) in accordance with the ComPPI [terms of use](https://comppi.linkgroup.hu/help/terms_of_use).
