from sys import argv
import torch

def main():
    inF = argv[1]
    outF = argv[2]
    mName = argv[3]
    netF = argv[4]
    torch.save({'inF':inF,'mName':mName,'netF':netF}, outF)
main()
