/// @cond DemoCode
#pragma once

#include "DGM.h"

using namespace DirectGraphicalModels;

    void makeAndInferGraph(std::string gFil, std::string comPPIFile, std::shared_ptr<CTrainNode> &nodeTrainerProt, std::shared_ptr<CTrainNode> &nodeTrainerInter, std::shared_ptr<CTrainEdge> &edgeTrainer, std::string outPref);
	void Main(void);
    void fillGraph(CGraphPairwise &graph);
    void printMarginals(const CGraphPairwise &graph, const std::string &string);
    void trainPots(std::string trainFile, std::shared_ptr<CTrainNode> &nodeTrainerProt, std::shared_ptr<CTrainNode> &nodeTrainerInter, std::shared_ptr<CTrainEdge> &edgeTrainer, std::string comPPIFile);
    void trainAndTestGraphs(std::string trainFile, std::string testFile, std::string comPPIFile, std::string outPref);
    void makeGraphFromFile(std::string fName, CGraphPairwise &graph, std::shared_ptr<CTrainNode> *nodeTrainerProt, std::shared_ptr<CTrainNode> *nodeTrainerInter, std::shared_ptr<CTrainEdge> *edgeTrainer, std::string outPref);
