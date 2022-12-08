#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
from models import *
from sys import argv
import os.path
from scipy.stats import sem
from ax.service.ax_client import AxClient
from sklearn.metrics import balanced_accuracy_score
import warnings


seed = 24
torch.manual_seed(seed)
np.random.seed(seed)

def testCNNs(parameterization):
    dataFile = parameterization["dataFile"]

    mName = parameterization["mName"]
    epochs = parameterization["epochs"]
    learningRate = parameterization["lRate"]
    #This parameter denotes that this is validation/model selection and not final training/testing
    validationRun = parameterization['validationRun']

    dataState = torch.load(dataFile)
    train_loaders = dataState['train_loaders']
    test_loaders = dataState['test_loaders']
    dataList = dataState['dataList']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'

    if mName == 'LinearNN':
        models = []
        for i in range(len(train_loaders)):
            #There was some debate online about if deepcopy works, so let's just make new ones
            #dataList[0] just shows the model the type of data shape it's looking at
            models.append(LinearNN(dataList[0],parameterization).to(device))

    elif mName == 'SimpleGCN':
        models = []
        for i in range(len(train_loaders)):
            models.append(SimpleGCN(dataList[0],parameterization).to(device))

    elif mName == 'GATCONV':
        models = []
        for i in range(len(train_loaders)):
            models.append(GATCONV(dataList[0],parameterization).to(device))

    elif mName == 'PANCONV':
        models = []
        for i in range(len(train_loaders)):
            models.append(PANCONV(dataList[0],parameterization).to(device))

    elif mName == 'GIN2':
        models = []
        for i in range(len(train_loaders)):
            models.append(GIN2(dataList[0],parameterization).to(device))

    else:
        print('Invalid Model!')
        return

    performance = evalModelCV(models, train_loaders, test_loaders,device, mName,
                           parameterization, epochs, learningRate, validationRun)
    return performance

def train(loader, model, optimizer, criterion,device):
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    for data in loader:  # Iterate in batches over the training dataset.
         #if isinstance(data,list): #Black magic
         #   data = data[0]
         data.to(device)
         out,e = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
         loss = criterion(out[data.is_pred], data.y[data.is_pred])  # Compute the loss only where we have data
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.

def testTraining(loader, model,device, validationRun, test=False):
     model.eval()
     correct = 0
     total = 0
     allPred = None
     allY = None
     testInd = None
     for data in loader:  # Iterate in batches over the training/test dataset.
         #if isinstance(data,list): #Black magic
         #   data = data[0]
         data.to(device)
         out,e = model(data.x, data.edge_index, data.batch)
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         if allPred is None:
             allPred = pred.cpu().numpy()
             allY = data.y.cpu().numpy()
             testInd = data.is_pred.cpu().numpy()
         else:
            allPred = np.concatenate((allPred,pred.cpu().numpy()))
            allY = np.concatenate((allY, data.y.cpu().numpy()))
            testInd = np.concatenate((testInd,data.is_pred.cpu().numpy()))
     #We supress warnings otherwise we get yelled at for the first few epochs every time
     bal_acc = 0
     with warnings.catch_warnings():
         warnings.simplefilter('ignore', category=UserWarning)
         #Only check in places where there is data
         bal_acc = balanced_accuracy_score(allY[testInd], allPred[testInd])
     return bal_acc  # Derive ratio of correct predictions.

def getEmbedding(loader, model):
    model.eval()
    embeddingAll = None
    yAll = None
    for data in loader:  # Iterate in batches over the training/test dataset.
         #if isinstance(data,list): #Black magic
         #   data = data[0]
         out,e = model(data.x, data.edge_index, data.batch)
         if (embeddingAll == None):
            embeddingAll = e
            yAll = data.y
         else:
            embeddingAll = torch.cat((embeddingAll,e), dim=0)
            yAll = torch.cat((yAll,data.y), dim=0)
    return embeddingAll, yAll

def evalModelCV(models, train_loaders, test_loaders,device, mName, parameters,
                epochs=1000, lr=0.001, validationRun=False):
    optimizers = []
    losses = []

    #We could make these parameters to pass in, but it doesn't seem worth it
    checkpoint_interval = 50
    print_interval = 10
    if not validationRun:
        print_interval = 10

    #Right now we only do early stopping in validation runs
    #We would need to bring the validation set to the final run
    patience = 50
    bestAcc = 0.0
    bestEpoch = 0

    start_epoch = 1
    for i in range(len(train_loaders)):
        optimizers.append(torch.optim.Adam(models[i].parameters(), lr=lr))
        losses.append(torch.nn.CrossEntropyLoss())

    train_acc_list = []
    test_acc_list = []
    perfDict = dict()
    perfDict["Epoch"] = []
    perfDict["Accuracy"] = []
    perfDict["Data"] = []
    perfDict["Fold"] = []
    perfDict["Model"] = []

    #Load already trained model
    #checkpoint_filename_parts = parameters['outputFile'].split('-')
    #checkpoint_filename_parts[1] = parameters['trainedNets']
    #checkpoint_filename = "-".join(checkpoint_filename_parts)+"_checkpoint"
    checkpoint_filename = parameters['outputFile']+"_checkpoint"
    if not validationRun and os.path.exists(checkpoint_filename):
        checkpoint = torch.load(checkpoint_filename)
        model_states = checkpoint['model_states']
        optimizer_states = checkpoint['optimizer_states']
        losses = checkpoint['losses']
        for i in range(len(train_loaders)):
            models[i].load_state_dict(model_states[i])
            optimizers[i].load_state_dict(optimizer_states[i])
        start_epoch = 1 #checkpoint['epoch']

    #Start training
    test_acc_batch_list = []
    patienceLeft = patience
    for epoch in range(start_epoch, epochs+1):
        train_acc_total = 0.0
        test_acc_total = 0.0
        test_acc_batch_list = []
        for i in range(len(train_loaders)):
            train(train_loaders[i], models[i], optimizers[i], losses[i], device)
            train_acc = testTraining(train_loaders[i], models[i], device, validationRun, test=False)
            test_acc = testTraining(test_loaders[i], models[i], device, validationRun, test=True)
            train_acc_total += train_acc
            test_acc_total += test_acc
            test_acc_batch_list.append(test_acc)
            if not validationRun:
                perfDict["Epoch"].append(epoch)
                perfDict["Accuracy"].append(train_acc)
                perfDict["Fold"].append(i)
                perfDict["Data"].append("Training Set")
                perfDict["Model"].append(mName)

                perfDict["Epoch"].append(epoch)
                perfDict["Accuracy"].append(test_acc)
                perfDict["Fold"].append(i)
                perfDict["Data"].append("Testing Set")
                perfDict["Model"].append(mName)
        train_acc_list.append(train_acc_total/len(train_loaders))
        test_acc_list.append(test_acc_total/len(train_loaders))

        if validationRun:
            if bestAcc < test_acc_list[-1] or epoch<100:
                patienceLeft = patience
                bestAcc = test_acc_list[-1]
                bestEpoch = epoch
            else:
                patienceLeft-=1
            if patienceLeft==0 and validationRun:
                print('Stopping early due to lack of validation improvements')
                break

        if (epoch%print_interval)==0:
            print(parameters['outputFile'],f' Epoch: {epoch:03d}, Train Acc: {train_acc_list[-1]:.4f}, Test Acc: {test_acc_list[-1]:.4f}')

        #Tuning runs aren't typically worth the space to checkpoint
        if not validationRun and epoch%checkpoint_interval==0:
            model_states=[]
            optimizer_states=[]
            for i in range(len(train_loaders)):
                models[i].eval()
                model_states.append(models[i].state_dict())
                optimizer_states.append(optimizers[i].state_dict())
            torch.save({
                 'parameters':parameters,
                 'perfDict':perfDict,
                 'model_states': model_states,
                 'optimizer_states': optimizer_states,
                 'losses': losses,
                 'epoch': epoch
                 },checkpoint_filename)
    parameters['best_epoch'] = bestEpoch

    #Finished Training
    if not validationRun:
        perfDF = pd.DataFrame.from_dict(perfDict)
        model_states=[]
        optimizer_states=[]
        for i in range(len(train_loaders)):
            models[i].eval()
            model_states.append(models[i].state_dict())
            optimizer_states.append(optimizers[i].state_dict())
        torch.save({
             'parameters':parameters,
             'perfDF':perfDF,
             'model_states': model_states,
             'optimizer_states': optimizer_states,
             'losses': losses,
             },parameters['outputFile'])
    return {'accuracy': (np.mean(test_acc_batch_list),sem(test_acc_batch_list))}

if __name__ == "__main__":
    inRun = argv[1]
    inData = argv[2]
    outF = argv[3]

    ax_client = AxClient.load_from_json_file(inRun)
    best_parameters, values = ax_client.get_best_parameters()
    best_parameters['dataFile'] = inData
    best_parameters['outputFile'] = outF
    best_parameters['epochs'] = 300
    best_parameters['validationRun'] = False
    print(best_parameters)
    testCNNs(best_parameters)




