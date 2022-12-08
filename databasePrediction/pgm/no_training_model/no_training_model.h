#pragma once

#include "DGM.h"

using namespace DirectGraphicalModels;

    void makeAndInferGraph(void);
	void Main(void);
    void fillGraph(CGraphPairwise &graph);
    void printMarginals(const CGraphPairwise &graph, const std::string &string);
    void makeGraphFromFile(std::string fName, CGraphPairwise &graph);

