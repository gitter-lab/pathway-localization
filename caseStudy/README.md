# Case Study

This directory contains all data and code used for the case study on data from primary fibroblasts during human cytomeglovirus infection. 

Three different experimental scenarios were explored.
For all scenarios, the model and parameters used were from one of the best performing models in the database prediction experiment. This was done by copying the complete model's `.json` file into the `axRuns` directory and loading that models for all scenarios. 

1. The best performing pre-trained model from the database prediction expedriment was taken with no further modifications and used it to predict localizations at the 120hpi timepoint.
  This scenario is referred to as `120` in code and other files.
  The pretrained model was retrieved from the database prediction experiment by copying the checkpoint file from the completed database prediction experiment, so from the `runs` directory the command
  ```
  cp ../../databasePrediction/runs/GATCONV-allPathBank-compartmentsNodes.p_checkpoint 120-GATCONV-topPathways-compartmentsNodes.p_checkpoint
  ```
  copies the trained model over to the case study. 
  This allows the trained model to be loaded for prediction and evaluation. 

2. We trained a model using a separate dataset that measured protein localization using a similar method on a different cell type and under a different biological condition, HeLa cells undergoing EGF stimulation. 
  This scenario is refered to as `egf120` in code and other files. 

3. We trained a model on the same HCMV infection experiment at the 24hpi timepoint.
  This scenario is refered to as `same` in code and other files.

More information on these scenarios can be found in the paper and supplement. 


