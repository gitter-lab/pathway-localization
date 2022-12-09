from sys import argv
import torch
"""
parsePGMOutput.py
Author: Chris Magnano

This small script saves the output of a probabilistic graphical
model run from the DGM library into the same torch format as 
all other models. 
"""

def main():
    inF = argv[1]
    outF = argv[2]
    mName = argv[3]
    netF = argv[4]
    torch.save({'inF':inF,'mName':mName,'netF':netF}, outF)
main()
