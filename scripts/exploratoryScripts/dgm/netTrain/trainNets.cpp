#include "trainNets.h"
#include <iostream>
#include <fstream>
#include <map>
#include <boost/algorithm/string.hpp>

const byte   nStates = 6;						//comPPI locations
const byte   nFeatures = 6;                     //also comPPI locations

void print_help(char *argv0)
{
	printf("Usage: %s <app>\n", argv0);
	printf("where <app> is on of the following:\n");
	printf("  exact - for exact inference and decoding\n");
	printf("  chain - for chain-structured graphical model\n");
	printf("  tree  - for tree-structured graphical model\n");
}

void printMarginals(const CGraphPairwise &graph, const std::string &str)
{
	Mat		 nodePot;
    int nNodes = graph.getNumNodes();
	printf("%s:\t", str.c_str());
	for (int i = 0; i < nNodes; i++) {
		graph.getNode(i, nodePot);
		printf((i < nNodes - 1) ? " %.2f\t%.2f\t|" : " %.2f\t%.2f\n", nodePot.at<float>(0, 0), nodePot.at<float>(1, 0));
	}
}

/*
 *This function takes in a network from Reactome and trains it using ComPPI data as features
 */
void trainPots(std::string trainFile, std::shared_ptr<CTrainNode> &nodeTrainerProt, std::shared_ptr<CTrainNode> &nodeTrainerInter, std::shared_ptr<CTrainEdge> &edgeTrainer, std::string comPPIFile)
{
    std::string n1, n2, eType, eLoc, line, uniprot;
    std::map<std::string, std::array<int, 6>> comPPIMap;
    int nodeID1, nodeID2, edgeID, i;
    float comPPIConf;
    Mat nodeFVec1(nFeatures, 1, CV_8UC1);       // Node features
    Mat nodeFVec2(nFeatures, 1, CV_8UC1);       // Node features
    Mat nodeFVec3(nFeatures, 1, CV_8UC1);       // Node features
    byte gt;                                    // Ground truth for each node   

    std::array<std::string, 6> allLocs = {"cytosol","extracellular","membrane","mitochondrion","nucleus","secretory-pathway"};
    std::map<std::string, int> locNames;

    for (int i=0; i<6; i++){
        locNames[allLocs[i]] = i;
    }
    

    //Read in comPPI Data
    std::ifstream inComFile(comPPIFile);
    while(std::getline(inComFile,line))
    {
        std::vector<std::string> result;
        split(result, line, boost::is_any_of("\t"), boost::token_compress_on);
        uniprot = result[0];
        
        if (uniprot == "uniprot"){
            continue;
        }

        std::array<int, 6> protList;
        for (int i=1; i<7; i++){
            comPPIConf  = std::stof(result[i]);
            protList[i-1] = static_cast<int>(comPPIConf*100);
        }
        comPPIMap[uniprot] = protList;
    }

    //Read in training files
    std::ifstream trainFileStream(trainFile);
    while(std::getline(trainFileStream,line))
    {
        std::string fName = line;
        std::ifstream inFile(fName);
        while(std::getline(inFile,line))
        {
            std::vector<std::string> result;
            split(result, line, boost::is_any_of("\t"), boost::token_compress_on);
            n1 = result[0];
            eType = result[1];
            n2 = result[2];
            eLoc = result[3]; //We probably need to map this to ComPPI data. This is our ground truth.
            gt = locNames[eLoc]; //Assuming there's no misses

            //Check that we have comPPI data
            if (!comPPIMap.count(n1)){
                for (i=0; i<nStates; i++){
                    nodeFVec1.at<int>(i,0) = 1.0f/6.0f;
                }
            } else {
                for (i=0; i<nStates; i++){
                    nodeFVec1.at<int>(i,0) = comPPIMap[n1][i];
                }
            }
            if (!comPPIMap.count(n2)){
                for (i=0; i<nStates; i++){
                    nodeFVec2.at<int>(i,0) = 1.0f/6.0f;
                }
            } else {
                for (i=0; i<nStates; i++){
                    nodeFVec2.at<int>(i,0) = comPPIMap[n2][i];
                }
            }

            //Edge 'node'
            for (i=0; i<nStates; i++){
                nodeFVec3.at<int>(i,0) = nodeFVec1.at<int>(i,0) * nodeFVec2.at<int>(i,0);
            }

            nodeTrainerProt->addFeatureVec(nodeFVec1, gt);
            nodeTrainerProt->addFeatureVec(nodeFVec2, gt);
            nodeTrainerInter->addFeatureVec(nodeFVec3, gt);
            edgeTrainer->addFeatureVecs(nodeFVec1, gt, nodeFVec3, gt);
            edgeTrainer->addFeatureVecs(nodeFVec2, gt, nodeFVec3, gt);
        }
    }

    printf("Training\n");
    nodeTrainerProt->train();
    nodeTrainerInter->train();
    edgeTrainer->train();
    printf("Done Training\n");
    return;
}

std::map<int, std::string> makeGraphFromFile(std::string fName, CGraphPairwise &graph, std::string comPPIFile, std::shared_ptr<CTrainNode> &nodeTrainerProt, std::shared_ptr<CTrainNode> &nodeTrainerInter, std::shared_ptr<CTrainEdge> &edgeTrainer, int edgeModel, std::string outPref)
{
    std::string n1, n2, eType, eLoc, line, uniprot;
    std::map<std::string, int> nameMap;
    std::map<int, std::string> revNameMap;
    std::map<std::string, std::array<int, 6>> comPPIMap;
    float comPPIConf, nPotCount, ePotCount;
    int nodeID1, nodeID2, edgeID, i;
    vec_float_t vParams = {100, 0.01f};                // What's used in the demo. Should look into.
    if (edgeModel <= 1 || edgeModel == 4) vParams.pop_back();   // Potts and Concat models need ony 1 parameter
    if (edgeModel == 0) vParams[0] = 1;                         // Emulate "No edges"
    nPotCount = 0;
    ePotCount = 0;

    Mat nodePot1(nStates, 1, CV_32FC1);			// node Potential (column-vector)
	Mat nodePot2(nStates, 1, CV_32FC1);			// node Potential (column-vector)
	Mat nodePot3(nStates, 1, CV_32FC1);			// node Potential (column-vector)
	Mat edgePot(nStates, nStates, CV_32FC1);	// edge Potential (matrix)
	
    Mat meanNodePot(nStates, 1, CV_32FC1);			// mean node potentials for graph
	Mat meanEdgePot(nStates, nStates, CV_32FC1);	// mean edge potentials for graph
    meanNodePot = 0;
    meanEdgePot = 0;
    
    Mat nodeFVec1(nFeatures, 1, CV_8UC1);       // Node features
    Mat nodeFVec2(nFeatures, 1, CV_8UC1);       // Node features
    Mat nodeFVec3(nFeatures, 1, CV_8UC1);       // Node features

    std::array<std::string, 6> allLocs = {"cytosol","extracellular","membrane","mitochondrion","nucleus","secretory-pathway"};
    

    //Read in comPPI Data
    std::ifstream inComFile(comPPIFile);
    while(std::getline(inComFile,line))
    {
        std::vector<std::string> result;
        split(result, line, boost::is_any_of("\t"), boost::token_compress_on);
        uniprot = result[0];
        
        if (uniprot == "uniprot"){
            continue;
        }

        std::array<int, 6> protList;
        for (int i=1; i<7; i++){
            comPPIConf  = std::stof(result[i]);
            protList[i-1] = static_cast<int>(comPPIConf*100);
        }
        comPPIMap[uniprot] = protList;
    }

    //Read in sif file with states
    std::ifstream inFile(fName);
    while(std::getline(inFile,line))
    {
        std::vector<std::string> result;
        split(result, line, boost::is_any_of("\t"), boost::token_compress_on);
        n1 = result[0];
        eType = result[1];
        n2 = result[2];
        eLoc = result[3];
        
        if (!comPPIMap.count(n1)){
            for (i=0; i<nStates; i++){
                nodeFVec1.at<int>(i,0) = 1.0f/6.0f;
            }
        } else {
            for (i=0; i<nStates; i++){
                nodeFVec1.at<int>(i,0) = comPPIMap[n1][i];
            }
        }
        if (!comPPIMap.count(n2)){
            for (i=0; i<nStates; i++){
                nodeFVec2.at<int>(i,0) = 1.0f/6.0f;
            }
        } else {
            for (i=0; i<nStates; i++){
                nodeFVec2.at<int>(i,0) = comPPIMap[n2][i];
            }
        }
        //Edge 'node'
        for (i=0; i<nStates; i++){
            nodeFVec3.at<int>(i,0) = nodeFVec1.at<int>(i,0) * nodeFVec2.at<int>(i,0);
        }
        
        //Set edge potentials
        for (i=0; i<nStates; i++){
            for (int j=0; j<nStates; j++){
                edgePot.at<float>(i,j) = nodePot1.at<float>(i,0) * nodePot2.at<float>(j,0);
            }
        }
        
        if (!nameMap.count(n1)) {
            nodePot1 = nodeTrainerProt->getNodePotentials(nodeFVec1,1.0);
            meanNodePot = meanNodePot + nodePot1;
            nPotCount += 1;
            nodeID1 = graph.addNode(nodePot1);
            nameMap.insert(std::pair<std::string, int>(n1, nodeID1));
            //revNameMap.insert(std::pair<int, std::string>(nodeID1, n1));

        } else {    
            nodeID1 = nameMap[n1];
        }
        if (!nameMap.count(n2)) {
            nodePot2 = nodeTrainerProt->getNodePotentials(nodeFVec2,1.0);
            meanNodePot = meanNodePot + nodePot2;
            nPotCount += 1;
            nodeID2 = graph.addNode(nodePot2);
            nameMap.insert(std::pair<std::string, int>(n2, nodeID2));
            //revNameMap.insert(std::pair<int, std::string>(nodeID1, n1));

        } else {    
            nodeID2 = nameMap[n2];
        }
        
        nodePot3 = nodeTrainerInter->getNodePotentials(nodeFVec3,1.0);
        meanNodePot = meanNodePot + nodePot3;
        nPotCount += 1;
        edgeID = graph.addNode(nodePot3);
        revNameMap.insert(std::pair<int, std::string>(edgeID, n1+"\t"+n2));

        edgePot = edgeTrainer->getEdgePotentials(nodeFVec1, nodeFVec3, vParams);
        meanEdgePot = meanEdgePot + edgePot;
        ePotCount += 1;
        graph.addArc(nodeID1, edgeID, edgePot);
        
        edgePot = edgeTrainer->getEdgePotentials(nodeFVec2, nodeFVec3, vParams);
        meanEdgePot = meanEdgePot + edgePot;
        ePotCount += 1;
        graph.addArc(nodeID2, edgeID, edgePot);
    }
    
    meanNodePot = meanNodePot/nPotCount;
    meanEdgePot = meanEdgePot/ePotCount;
    
    //Should probably make this a function
    std::string name, ext, oFileName;
    size_t sep = fName.find_last_of("/");
    if (sep != std::string::npos)
        fName = fName.substr(sep + 1, fName.size() - sep - 1);

    size_t dot = fName.find_last_of(".");
    if (dot != std::string::npos)
    {
        name = fName.substr(0, dot);
        ext  = fName.substr(dot, fName.size() - dot);
    }
    else
    {
        name = fName;
        ext  = "";
    }
    oFileName = outPref + "pots_"+name+".txt";
    std::ofstream oFile(oFileName);
    if (oFile.is_open()){
        for(int i=0; i<meanNodePot.rows; i++)
        {
            for(int j=0; j<meanNodePot.cols; j++)
            {
            oFile<<meanNodePot.at<float>(i,j)<<"\t";
            }
        oFile<<std::endl;
        }
        //for(int i=0; i<meanEdgePot.rows; i++)
        //{
        //    for(int j=0; j<meanEdgePot.cols; j++)
        //    {
        //    oFile<<meanEdgePot.at<float>(i,j)<<"\t";
        //    }
        //oFile<<std::endl;
        //}
        oFile.close(); 
    } else {
        printf("FILE AINT OPEN MOOM");
    }
    
    return revNameMap;
}

void makeAndInferGraph(std::string gFile, std::string comPPIFile, std::shared_ptr<CTrainNode> &nodeTrainerProt, std::shared_ptr<CTrainNode> &nodeTrainerInter, std::shared_ptr<CTrainEdge> &edgeTrainer, int edgeModel, std::string outPref)
{
	size_t			i,j,nNodes;
	CGraphPairwise	graph(nStates);
    std::map<int, std::string> revNameMap;
    std::string resFileName = "basicModelGuess";
    std::string name, ext;
	
    CDecodeExact	decoderExcact(graph);
	CInferExact		infererExact(graph);
	CInferChain		infererChain(graph);
	CInferTree		infererTree(graph);
	CInferViterbi	infererViterbi(graph);
    
    std::array<std::string, 6> allLocs = {"cytosol","extracellular","membrane","mitochondrion","nucleus","secretory-pathway"};
    
    revNameMap = makeGraphFromFile(gFile, graph, comPPIFile, nodeTrainerProt, nodeTrainerInter, edgeTrainer, edgeModel, outPref);
    nNodes = graph.getNumNodes();
    
    printf("%s\n", gFile.c_str());
    
    size_t sep = gFile.find_last_of("/");
    if (sep != std::string::npos)
        gFile = gFile.substr(sep + 1, gFile.size() - sep - 1);

    size_t dot = gFile.find_last_of(".");
    if (dot != std::string::npos)
    {
        name = gFile.substr(0, dot);
        ext  = gFile.substr(dot, gFile.size() - dot);
    }
    else
    {
        name = gFile;
        ext  = "";
    }
    

    for (i=4; i < 5; i++){
        //std::string resFileName = "trainedModel_"+name+"_"+std::to_string(i*10000)+".txt";
        std::string resFileName = outPref+"trainedModel_"+name+".txt";
	    CInferLBP		infererLBP(graph);
        infererLBP.infer(10000*(i+1));											// Loopy Belief Probagation inference
        vec_byte_t decoding_infererLBP = infererLBP.decode();			// Loopy Belief Probagation decoding

        //printf("\n\t\t\t\tDECODING\n\t\t");
        //printf("LBP inferer:\n");	
        //for (i = 0; i < nNodes; i++) {
        //    printf("%s\t%s", revNameMap[i].c_str(), allLocs[decoding_infererLBP[i]].c_str());	
        //    printf("\n");
        //}
        printf("%s\n",resFileName.c_str());
        std::ofstream oFile(resFileName);
        if (oFile.is_open()){
            for (std::pair<int, std::string> el : revNameMap) {
                j = el.first;
                std::string name = el.second;
                oFile << name << "\t" << allLocs[decoding_infererLBP[j]] << "\n";
                //printf("%s\t%s", name.c_str(), allLocs[decoding_infererLBP[i]].c_str());	
                //printf("\n");
            }
            oFile.close();
        } else {
            printf("FILE AINT OPEN MOOM");
        }
    }
}

void trainAndTestGraphs(std::string trainFile, std::string testFile, std::string comPPIFile, int nodeModel, int edgeModel, std::string outPref){
    auto nodeTrainerProt = CTrainNode::create(nodeModel, nStates, nFeatures); 
    auto nodeTrainerInter = CTrainNode::create(nodeModel, nStates, nFeatures); 
    auto edgeTrainer = CTrainEdge::create(edgeModel, nStates, nFeatures);
    trainPots(trainFile, nodeTrainerProt, nodeTrainerInter, edgeTrainer, comPPIFile);
    std::string line;
    
    //Read in testing files
    std::ifstream testFileStream(testFile);
    while(std::getline(testFileStream,line))
    {
        std::string fName = line;
        makeAndInferGraph(fName, comPPIFile, nodeTrainerProt, nodeTrainerInter, edgeTrainer, edgeModel, outPref);	
    }
    return;
}


int main(int argc, char *argv[])
{
	if (argc != 7) {
		print_help(argv[0]);
		return 0;
	}
	std::string trainFile = argv[1];
	std::string testFile = argv[2];
	std::string comPPIFile = argv[3];
    int nodeModel   = atoi(argv[4]);
    int edgeModel   = atoi(argv[5]); 
    std::string outPref   = argv[6]; 
    trainAndTestGraphs(trainFile, testFile, comPPIFile, nodeModel, edgeModel, outPref);
    
    return 0;
}
