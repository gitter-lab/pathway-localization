from sys import argv
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import PCA

"""
This script takes in a tsv of all uniprot keywords, and constructs features based on the top X keywords. We might try clustering or PCA in the future.

NOW WE DOIN PCA

Ya know, is there a form of feature clustering to maximize variance AND minimize missing data?
"""

def main():

    #percent of uniprot keyword has to be in to be included
    thresh = 25

    #dictionary of name-to-indexes of keywords
    keyDict = dict()

    #dictionary of proteins to keywords
    protDict = dict()

    #argv[1] is uniprot file
    df = pd.read_csv(argv[1],sep="\t",usecols=["Entry name","Keywords"])
    df["Keywords"] = df["Keywords"].str.split(";")
    df["Entry name"] = df["Entry name"].str.replace("_HUMAN","")
    df = df.set_index("Entry name")
    #print(df)
    #This converts the list column into binary columns
    s = df['Keywords']

    mlb = MultiLabelBinarizer()
    binDF = pd.DataFrame(mlb.fit_transform(s),columns=mlb.classes_, index=df.index)
    binDF = binDF.drop(columns=["Reference proteome","Alternative splicing", "3D-structure","Direct protein sequencing"])
    s = binDF.sum().sort_values(ascending=False, inplace=False)
    binDF = binDF[binDF.columns[binDF.sum()>(len(binDF)*(thresh*0.01))]]
    #print(s[:20])

    pca = PCA(n_components=6, svd_solver="full")
    pca.fit(binDF)

    print(pca.explained_variance_ratio_)
    print(sum(pca.explained_variance_ratio_))
    #print(abs(pca.components_[0]))
    for i in range(6):
        print(s.index[np.argmax(abs(pca.components_[i]))])

    newFeatures = pca.transform(binDF)
    minVal = np.amin(newFeatures)
    newFeatures = newFeatures-minVal
    newDF = pd.DataFrame(newFeatures, columns=["pca" + str(s) for s in range(6)],index=binDF.index)
    newDF.index.names = ['uniprot']
    newDF.to_csv('uniprotKeywordsPCA_'+str(thresh)+'.tsv',sep="\t")
    return

main()
