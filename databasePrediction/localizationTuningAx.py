#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import networkx as nx
import numpy as np
import torch
from torch_geometric.utils.convert import from_networkx
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
from torch_geometric.nn import GCNConv, GINConv, GATv2Conv
import random
import math
from sklearn.model_selection import KFold
from sys import argv
import pickle as pkl
import os.path
from scipy.stats import sem
from ax.service.ax_client import AxClient
from models import *
from localizationPyTorchGeo import *

seed = 24
torch.manual_seed(seed)
np.random.seed(seed)

def runTrial(client):
    parameters, trial_index = ax_client.get_next_trial()
    ax_client.complete_trial(trial_index=trial_index, raw_data=testCNNs(parameters))
    return

if __name__ == "__main__":
    mName = argv[1]
    dataFile = argv[2]
    outF = argv[3]

    ax_client = AxClient()

    #This is where the NN parameters are all defined
    parameters=[
        {"name": "lRate", "type": "range", "bounds": [1e-5, 0.01], "log_scale": True},
        {"name": "l_depth", "type": "range", "value_type": "int", "bounds": [1, 5]},
        {"name": "dropout", "type": "choice", "value_type": "float", "values": [0.0, 0.5]},
        {"name": "epochs", "type": "fixed", "value_type": "int", "value": 1000},
        {"name": "dataFile", "type": "fixed", "value_type": "str", "value": dataFile},
        {"name": "mName", "type": "fixed", "value_type": "str", "value": mName},
        {"name": "validationRun", "type": "fixed", "value_type": "bool", "value": True}]
    if mName == 'LinearNN':
        parameters.append({"name": "dim", "type": "range", "value_type": "int", "bounds": [24, 128]})
        parameters.append({"name": "activation", "type": "choice", "value_type": "str", "values": ['relu','tanh']})
    elif mName == 'SimpleGCN':
        parameters.append({"name": "dim", "type": "range", "value_type": "int", "bounds": [24, 128]})
        parameters.append({"name": "c_depth", "type": "range", "value_type": "int", "bounds": [1, 10]})
    elif mName == 'GATCONV':
        parameters.append({"name": "dim", "type": "range", "value_type": "int", "bounds": [12, 48]})
        parameters.append({"name": "c_depth", "type": "range", "value_type": "int", "bounds": [1, 10]})
        parameters.append({"name": "num_heads", "type": "range", "value_type": "int", "bounds": [1, 5]})
    elif mName == 'PANCONV':
        parameters.append({"name": "dim", "type": "range", "value_type": "int", "bounds": [24, 128]})
        parameters.append({"name": "c_depth", "type": "range", "value_type": "int", "bounds": [1, 10]})
        parameters.append({"name": "filters", "type": "range", "value_type": "int", "bounds": [1, 10]})
    elif mName == 'GIN2':
        parameters.append({"name": "dim", "type": "range", "value_type": "int", "bounds": [24, 128]})
        parameters.append({"name": "c_depth", "type": "range", "value_type": "int", "bounds": [1, 10]})

    ax_client.create_experiment(
        name=mName,
        parameters=parameters,
        objective_name="accuracy",
        minimize=False)

    startI = 0
    pastParams1 = {}
    for i in range(29,-1,-1):
        curPath = outF+"_iter"+str(i)
        if os.path.exists(curPath):
            startI = i+1
            ax_client = AxClient.load_from_json_file(curPath)
            break
    for i in range(startI,30):
        parameters, trial_index = ax_client.get_next_trial()
        #We stalled out, no need to continue
        if pastParams1 == parameters:
            print("Stoppong Ax Early Because Trials Started Repeating")
            break
        # Local evaluation here can be replaced with deployment to external system.
        ax_client.complete_trial(trial_index=trial_index, raw_data=testCNNs(parameters))
        pastParams1 = parameters
        ax_client.save_to_json_file(filepath=outF+"_iter"+str(i))
    best_parameters, values = ax_client.get_best_parameters()
    ax_client.save_to_json_file(filepath=outF)
    print(best_parameters)



