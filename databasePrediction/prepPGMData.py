from sys import argv
import numpy as np
from sklearn.model_selection import KFold
from prepPytorchData import loc_data_to_tables
import os

seed = 24
np.random.seed(seed)

def load_pgm_data(networks_file, features_file, out_dir):
    #Right now we only use this for pOrder, would save seconds and be a bit cleaner if we made getting
    #pOrder a separate method
    allPathNets, allPathDFs, featuresDF, locList, locDict, pOrder = loc_data_to_tables(networks_file, features_file)

    numFeatures = len(featuresDF.columns)
    #Get directory of networks
    net_dir_dict = dict()
    for line in open(networks_file):
        f = line.strip()
        net_dir_dict[os.path.basename(f)] = f

    i=0
    kf = KFold(n_splits = 5)
    pOrder = np.array(pOrder)
    for train_ind, test_ind in kf.split(pOrder):
        net_train, net_test = pOrder[train_ind], pOrder[test_ind]
        #Save to file
        o_file_train = os.path.join(out_dir, "trainFold"+str(i)+'.txt')
        o_file_test = os.path.join(out_dir, "testFold"+str(i)+'.txt')
        with open(o_file_train, 'w') as oFile:
            for net in net_train:
                oFile.write(net_dir_dict[net.strip()]+'\n')
        with open(o_file_test, 'w') as oFile:
            for net in net_test:
                oFile.write(net_dir_dict[net.strip()]+'\n')
        i+=1
    return


if __name__ == "__main__":
    networksFile = argv[1]
    #networksFile_val = argv[2]
    featuresFile = argv[2]
    out_dir = argv[3]

    #We need a lot of files for each run, so making it a directory makes sense
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass
    load_pgm_data(networksFile, featuresFile, out_dir)
