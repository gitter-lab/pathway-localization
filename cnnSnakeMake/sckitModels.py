from sys import argv
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV,KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import get_scorer
from prepPytorchData import loc_data_to_tables
from skopt import BayesSearchCV
from torch import save as t_save

seed = 24
np.random.seed(seed)

def load_scikit_data(networks_file, features_file):
    allPathNets, allPathDFs, featuresDF, locList, locDict, pOrder = loc_data_to_tables(networks_file, features_file)
    mergedDF = allPathDFs[pOrder[0]]
    for i in range(1,len(pOrder)):
        curPath = pOrder[i]
        mergedDF = mergedDF.append(allPathDFs[curPath], ignore_index=True, sort=False)

    #pd.concat(allPathDFs.values(), ignore_index=True)

    featuresDict = featuresDF.to_dict('index')
    featuresList = list(featuresDF.columns)

    appliedDF = mergedDF[['Interactor1','Interactor2']].apply(lambda x: getFeatColumns(featuresDict, featuresList, *x), axis=1, result_type='expand')
    mergedDF = pd.concat([mergedDF, appliedDF], axis='columns')
    fList = []
    for l in featuresList:
        fList.append(l+"_1")
        fList.append(l+"_2")

    y = mergedDF['Location']
    x = mergedDF[fList]

    #Lets us grab X and y by network
    network_index = dict()
    cur_ind = 0
    for p in pOrder:
        net = allPathDFs[p]
        network_index[p] = np.zeros(len(y), dtype=bool)
        size = len(allPathDFs[p])
        network_index[p][cur_ind:cur_ind+size] = 1
        cur_ind += size

    return x, y, pOrder, network_index

def fit_grid_search():
    return best_params

def getFeatColumns(fD,locList,i1,i2):
    outDict=dict()
    if i1 in fD and i2 in fD:
        if str(fD[i1]) > str(fD[i2]):
            d1 = makeOutD(locList,fD,i1,"_1")
            d2 = makeOutD(locList,fD,i2,"_2")
            outDict = {**d1, **d2}
        else:
            d2 = makeOutD(locList,fD,i2,"_1")
            d1 = makeOutD(locList,fD,i1,"_2")
            outDict = {**d2, **d1}
    elif i1 in fD:
        d1 = makeOutD(locList,fD,i1,"_1")
        d2 = makeOutD(locList,fD,i2,"_2",True)
        outDict = {**d1, **d2}
    elif i2 in fD:
        d2 = makeOutD(locList,fD,i2,"_1")
        d1 = makeOutD(locList,fD,i1,"_2",True)
        outDict = {**d2, **d1}
    else:
        d1 = makeOutD(locList,fD,i1,"_1",True)
        d2 = makeOutD(locList,fD,i2,"_2",True)
        outDict = {**d1, **d2}
    return outDict

def makeOutD(locList,fD,i1,suff,miss=False):
    outD = dict()
    for l in locList:
        if miss:
            #Do the same as in the deep networks here
            outD[l+suff] = 1.0/len(locList)
        else:
            outD[l+suff] = fD[i1][l]
    return outD

def eval_sklearn_model(networksFile, networksFile_val, featuresFile, model, outFile):
    X, y, path_order, network_index = load_scikit_data(networksFile, featuresFile)
    X_val, y_val, path_order_val, network_index_val = load_scikit_data(networksFile_val, featuresFile)
    metric = 'balanced_accuracy'
    path_order = np.asarray(path_order)

    nFolds = 5
    kf = KFold(n_splits = nFolds)
    clf = None
    grid = None

    #Model tuning
    if model=='logit':
        #Logistic Regression
        lr_param_grid = {'tol':(1e-6,1e-1,'log-uniform'),
                      'penalty':['l2','none'],
                      'C': (0.001, 100.0, 'log-uniform'),
                      'class_weight':['balanced',None]}
        grid = BayesSearchCV(LogisticRegression(), lr_param_grid,
                            scoring=metric, refit=True, verbose=1, n_iter=30)
        grid.fit(X_val, y_val)
        print("Best Params for :",model, grid.best_params_)
        clf = LogisticRegression(**grid.best_params_)

    elif model=='rf':
        #Random Forest
        rf_param_grid = {'max_depth': (1,10),
                'min_samples_split': (2,10),
                      'n_estimators': (1,100),
                      'class_weight':['balanced',None]}
        grid = BayesSearchCV(RandomForestClassifier(), rf_param_grid,
                            scoring=metric, refit=True, verbose=1, n_iter=30)
        grid.fit(X_val, y_val)
        print("Best Params for :",model, grid.best_params_)
        clf = RandomForestClassifier(**grid.best_params_)

    #Peform final prediction
    perfs = []
    preds = []
    y_all = []
    for train_ind, test_ind in kf.split(path_order):
        edge_index_train = np.zeros(len(y), dtype=bool)
        edge_index_test = np.zeros(len(y), dtype=bool)
        for path in path_order[train_ind]:
            edge_index_train[network_index[path]]=1
        for path in path_order[test_ind]:
            edge_index_test[network_index[path]]=1
        X_train, X_test = X[edge_index_train], X[edge_index_test]
        y_train, y_test = y[edge_index_train], y[edge_index_test]
        clf.fit(X_train, y_train)
        out = clf.predict(X_test)
        preds = np.concatenate((preds, out))
        y_all = np.concatenate((y_all, y_test))
        scorer = get_scorer(metric)
        score = scorer._score_func(y_test, out)
        perfs.append(score)
        print(score)
    t_save({'metrics': perfs,
            'predictions': preds,
            'best_params': grid.best_params_,
            'y_all': y_all,
            'model': model,
            'network_index': network_index}
            ,outFile)



if __name__ == "__main__":
    networksFile = argv[1]
    networksFile_val = argv[2]
    featuresFile = argv[3]
    model = argv[4]
    outFile = argv[5]

    eval_sklearn_model(networksFile, networksFile_val, featuresFile, model, outFile)
