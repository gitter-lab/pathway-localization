[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7140733.svg)](https://doi.org/10.5281/zenodo.7140733)

Scripts, data, and supplementary information for the publication:

[Graph algorithms for predicting subcellular localization at the pathway level](https://doi.org/10.1142/9789811270611_0014).
Chris S Magnano, Anthony Gitter.
*Pacific Symposium on Biocomputing*, 2023.

Supplementary text, figures, and tables can be found in `supplement.pdf`

## Repository overview

- `caseStudy` - Code and data used for the case study predicting subcellular localizations in primary fibroblasts during human cytomeglovirus infection. Instructions for running this analysis are in the subdirectory. 
- `data` - Pathway data used for all experiments and scripts originally used to process that data. 
- `databasePrediction` - Code and data used for predicting pathway database subcellular localizations from protein-level subcellular localization data. Instructions for running this analysis via Snakemake are below. 
- `exploratoryAnalyses` - Preliminary figures from developing localization models and exploring possible pipelines using approximately 100 Reactome pathways in the 'Developmental' category. 

## Setting up the environment

1. Install Anaconda if needed.

   The easiest way to install Python and the required packages is with [Anaconda](https://www.anaconda.com/download/).
   The Carpentries [Anaconda installation instructions](https://carpentries.github.io/workshop-template/#python) provide guides and videos on how to install Anaconda for your operating system.

2. Create the conda environment.

   From the `pathway-localization` directory, run:

   ```
   conda env create -f environment.yml
   conda activate pw_loc
   ```
   to create a conda environment with the required packages and activate that environment.
   
   
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
This script will interactively display all plots from the database prediction experiment. 

## Running localization prediction

After following the steps in `Setting up the environment`, navigating to the directory `databasePrediction` and using the command:

```
snakemake --cores [n] all
```

will run the full localization prediction experiment, where `[n]` is the number of desired cores.

This workflow ends by running `plotResults.py`, which recreates the database prediction figures from the paper interactively. 
It does not save any figures automatically. 

The file `results/allRes.p` contains 2 Pandas DataFrames saved using PyTorch, which store results metrics for each pathway and model configuration. 
`metrics` contains the per-pathway metrics for each run and pathway, while `mergedMetrics` contains metrics calculated by considering all edges at once to perform a single metric calculation per model. 
They can be loaded into Python as:

```
import torch
data = torch.load('results/allRes.p')
metricDF = data['metrics']
metricMergedDF = data['mergedMetrics']
```

## Datasets
- Reactome data were from version 72 retrieved on June 26, 2020. These files were licensed with the [CC0 license](https://reactome.org/license).
- [Pathway Commons](http://www.pathwaycommons.org/) data were from version 12 retrieved on June 26, 2020. These files have the same licenses as the original source databases. The PathBank files were licensed with the [ODbL v1.0 license](https://www.pathbank.org/about).
- ComPPI data were retrieved on November 9, 2022. These files were licensed with the [CC BY-SA 4.0 license](https://comppi.linkgroup.hu/help/terms_of_use).
- Compartments data were retrieved on September 29, 2021. These files were licensed with the [CC BY 4.0 license](https://compartments.jensenlab.org/Downloads).
- UniProt data were retreieved in October 2021. These files were licensed with the [CC BY 4.0 license](https://www.uniprot.org/help/license).

Any publications using these datasets should cite the original sources of the data.

## Licenses
All software (e.g. files matching the patterns `Snakefile`, `CMake*`, `*.py`, `*.h`, `*.cpp`, and `*.sh`) is available under a MIT license ([`LICENSE-MIT.txt`](LICENSE-MIT.txt)).

All other files (including data, figures, and the supplementary information) are available under a CC BY 4.0 license ([`LICENSE-CC-BY.txt`](LICENSE-CC-BY.txt)) except for those derived from PathBank and ComPPI.
Files derived from PathBank (`data/labeledPathBank.zip`) are available under a [ODbL v1.0 license](https://opendatacommons.org/licenses/odbl/1-0/) in accordance with the PathBank [terms of use](https://www.pathbank.org/about).
Files derived from ComPPI (`databasePrediction/data/comPPINodes.tsv`) are available under a [CC BY-SA 4.0 license](https://creativecommons.org/licenses/by-sa/4.0/) in accordance with the ComPPI [terms of use](https://comppi.linkgroup.hu/help/terms_of_use).
