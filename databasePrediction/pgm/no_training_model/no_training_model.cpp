#include "no_training_model.h"
#include <iostream>
#include <fstream>
#include <map>
#include <boost/algorithm/string.hpp>

const byte   nStates = 6;						//comPPI locations
const byte   nFeatures = 6;                     //also comPPI locations

void print_help(char *argv0)
{
	printf("Usage: %s <graphsFile> <featuresFile>\n", argv0);
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

std::map<int, std::string> makeGraphFromFile(std::string fName, CGraphPairwise &graph, std::string comPPIFile)
{
    std::string n1, n2, eType, eLoc, line, uniprot;
    std::map<std::string, int> nameMap;
    std::map<int, std::string> revNameMap;
    std::map<std::string, std::array<float, 6>> comPPIMap;
    int nodeID1, nodeID2, edgeID, i;
	Mat goodNodePot(nStates, 1, CV_32FC1);			// node Potential (column-vector)
	Mat nodePot1(nStates, 1, CV_32FC1);			// node Potential (column-vector)
	Mat nodePot2(nStates, 1, CV_32FC1);			// node Potential (column-vector)
	Mat nodePot3(nStates, 1, CV_32FC1);			// node Potential (column-vector)
	Mat edgePot(nStates, nStates, CV_32FC1);	// edge Potential (matrix)
    bool haveGoodNodePot = false;

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

        std::array<float, 6> protList;
        for (int i=1; i<7; i++){
            protList[i-1] = std::stof(result[i]);
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
        
        goodNodePot = false;
        //Check that we have comPPI data
        if (!comPPIMap.count(n1)){
            if (haveGoodNodePot){
                nodePot1 = goodNodePot;
            } else {
                for (i=0; i<nStates; i++){
                    nodePot1.at<float>(i,0) = 1.0f/6.0f;
                }
                printf("Node %s is not in comPPI", n1.c_str());
            }
        } else {
            for (i=0; i<nStates; i++){
                nodePot1.at<float>(i,0) = comPPIMap[n1][i];
            }
            goodNodePot = nodePot1;
        }
        if (!comPPIMap.count(n2)){
            if (haveGoodNodePot){
                nodePot2 = goodNodePot;
            } else {
                for (i=0; i<nStates; i++){
                    nodePot2.at<float>(i,0) = 1.0f/6.0f;
                }
                printf("Node %s is not in comPPI", n2.c_str());
            }
        } else {
            for (i=0; i<nStates; i++){
                nodePot2.at<float>(i,0) = comPPIMap[n2][i];
            }
            goodNodePot = nodePot2;
        }

        //Set edge potentials
        for (i=0; i<nStates; i++){
            nodePot3.at<float>(i,0) = 1.0f/6.0f;
            for (int j=0; j<nStates; j++){
                edgePot.at<float>(i,j) = nodePot1.at<float>(i,0) * nodePot2.at<float>(j,0);
            }
        }
        
        if (!nameMap.count(n1)) {
            nodeID1 = graph.addNode(nodePot1);
            nameMap.insert(std::pair<std::string, int>(n1, nodeID1));

        } else {    
            nodeID1 = nameMap[n1];
        }
        if (!nameMap.count(n2)) {
            nodeID2 = graph.addNode(nodePot2);
            nameMap.insert(std::pair<std::string, int>(n2, nodeID2));

        } else {    
            nodeID2 = nameMap[n2];
        }
        edgeID = graph.addNode(nodePot3);
        revNameMap.insert(std::pair<int, std::string>(edgeID, n1+"\t"+n2));

        graph.addArc(nodeID1, edgeID, edgePot);
        graph.addArc(nodeID2, edgeID, edgePot);
    }
    return revNameMap;
}

void makeAndInferGraph(std::string gFile, std::string comPPIFile, std::string outPref)
{
	size_t			i,j,nNodes;
	CGraphPairwise	graph(nStates);
    std::map<int, std::string> revNameMap;
    std::string name, ext;
	
    std::array<std::string, 6> allLocs = {"cytosol","extracellular","membrane","mitochondrion","nucleus","secretory-pathway"};
    
    revNameMap = makeGraphFromFile(gFile, graph, comPPIFile);
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
    

    std::string resFileName = outPref+"pgmDirectFeatures_"+name+".txt";
    CInferTRW inferer(graph);
    inferer.infer(10000);											// Loopy Belief Probagation inference
    vec_byte_t decoding_inferer = inferer.decode();			// Loopy Belief Probagation decoding

    printf("%s\n",resFileName.c_str());
    std::ofstream oFile(resFileName);
    if (oFile.is_open()){
        for (std::pair<int, std::string> el : revNameMap) {
            j = el.first;
            std::string name = el.second;
            oFile << name << "\t" << allLocs[decoding_inferer[j]] << "\n";
        }
        oFile.close();
    } else {
        printf("Error opening file");
    }
}

int main(int argc, char *argv[])
{
	if (argc != 4) {
		print_help(argv[0]);
		return 0;
	}
	std::string gFile = argv[1];
	std::string comPPIFile = argv[2];
	std::string outPref = argv[3];
    makeAndInferGraph(gFile, comPPIFile, outPref);	
    
    return 0;
}
