"""
Models.py
Author: Chris Magnano

Various Pytorch and Pytorch Geometric neural network models used 
for localization prediction. Most of these models were based on 
Google Colab examples found at: 
https://pytorch-geometric.readthedocs.io/en/latest/notes/colabs.html
"""
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
from torch_geometric.nn import GCNConv, GINConv, GATv2Conv
import math

seed = 2
torch.manual_seed(seed)

class LinearNN(torch.nn.Module):
    """
    A fully-connected neural network with [dim] neurons in each of its layers. 
    """
    def __init__(self, dataset, mParams):
        super(LinearNN, self).__init__()
        torch.manual_seed(seed)
        dim = mParams['dim']
        dim2 = max(1,math.ceil(dim/2.0))
        dim4 = max(1,math.ceil(dim/4.0))
        self.dropout_chance = mParams['dropout']
        l_depth = mParams['l_depth']

        #Programatically set activation function
        activation_param = mParams['activation']
        self.activation_func = None
        if activation_param=='relu':
            self.activation_func = torch.nn.ReLU()
        elif activation_param=='tanh':
            self.activation_func = torch.nn.Tanh()

        self.lin_list = torch.nn.ModuleList()
        self.l1 = Linear(dataset.num_features*2, dim)
        for i in range(l_depth-1):
            self.lin_list.append(Linear(dim,dim))
        self.classifier = Linear(dim, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        #Take the node embeddings and concat nodes for each edge
        e = torch.cat((x[edge_index[0]], x[edge_index[1]]), dim=1)
        h = self.l1(e)
        h = self.activation_func(h)
        for layer in self.lin_list:
            h = layer(h)
            h = self.activation_func(h)
        h = F.dropout(h, p=self.dropout_chance, training=self.training)
        out = self.classifier(h)

        return out, h

class SimpleGCN(torch.nn.Module):
    """
    A simply graph convolutional network. This network has [c_depth] 
    convolutional layers followed by [l_depth] fully connected layers. 
    """
    def __init__(self, dataset, mParams):
        super(SimpleGCN, self).__init__()
        torch.manual_seed(seed)
        dim = mParams['dim']
        dim2 = max(1,math.ceil(dim/2.0))
        dim4 = max(1,math.ceil(dim/4.0))
        self.dropout_chance = mParams['dropout']
        c_depth = mParams['c_depth']
        l_depth = mParams['l_depth']
        self.lin_list = torch.nn.ModuleList()
        self.conv_list = torch.nn.ModuleList()
        self.conv1 = GCNConv(dataset.num_features,dim)
        for i in range(c_depth-1):
            self.conv_list.append(GCNConv(dim,dim))
        for i in range(l_depth):
            self.lin_list.append(Linear(dim,dim))
        self.classifier = Linear(dim*2, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        h = self.conv1(x, edge_index).relu()
        for i in range(len(self.conv_list)):
            layer = self.conv_list[i]
            h = layer(h, edge_index).relu()
        for layer in self.lin_list:
            h = layer(h).relu()
        h = F.dropout(h, p=self.dropout_chance, training=self.training)

        #Take the node embeddings and concat nodes for each edge
        e = torch.cat((h[edge_index[0]], h[edge_index[1]]), dim=1)

        # Apply a final (linear) classifier.
        out = self.classifier(e)

        return out, e


class GATCONV(torch.nn.Module):
    """
    A graph attention network with [c_depth] graph layers and [l_depth] linear layers.
    Uses v2 of graph attention that provides dynamic instead of static attention.
    The graph layer dimension and number of attention heads are specified by [dim] 
    and [num_heads].
    """
    def __init__(self, dataset, mParams):
        super(GATCONV, self).__init__()
        torch.manual_seed(seed)
        dim = mParams['dim']
        dim2 = max(1,math.ceil(dim/2.0))
        dim4 = max(1,math.ceil(dim/4.0))
        self.dropout_chance = mParams['dropout']
        c_depth = mParams['c_depth']
        l_depth = mParams['l_depth']
        num_heads = mParams['num_heads']
        self.conv_list = torch.nn.ModuleList()
        self.lin_list = torch.nn.ModuleList()

        # The graph attention layer provides a re-weighted version of the output node representation for every
        # attention heads in the default setting, which is why we multiply by num_heads
        self.conv1 = GATv2Conv(in_channels=dataset.num_features, out_channels=dim, heads=num_heads)
        for i in range(c_depth-1):
                self.conv_list.append(GATv2Conv(in_channels=dim * num_heads, out_channels=dim, heads=num_heads))
        self.l1 = Linear(dim*num_heads,dim)
        for i in range(l_depth-1):
            self.lin_list.append(Linear(dim,dim))
        self.classifier = Linear(dim*2, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        h = self.conv1(x, edge_index).relu()
        for layer in self.conv_list:
            h = layer(h, edge_index).relu()
        h = self.l1(h).relu()
        for layer in self.lin_list:
            h = layer(h).relu()
        h = F.dropout(h, p=self.dropout_chance, training=self.training)

        #Take the node embeddings and concat nodes for each edge
        e = torch.cat((h[edge_index[0]], h[edge_index[1]]), dim=1)
        out = self.classifier(e)

        return out, e

class GIN2(torch.nn.Module):
    """
    A graph isomorphism network with [c_depth] convolutional layers and [l_depth] fully
    connected layers.  
    """
    def __init__(self, dataset,mParams):
        super(GIN2, self).__init__()
        torch.manual_seed(seed)
        dim = mParams['dim']
        dim2 = max(1,math.ceil(dim/2.0))
        dim4 = max(1,math.ceil(dim/4.0))
        self.dropout_chance = mParams['dropout']
        c_depth = mParams['c_depth']
        l_depth = mParams['l_depth']
        self.conv1 = GINConv(Sequential(Linear(dataset.num_features, dim), BatchNorm1d(dim), ReLU(),
                                        Linear(dim, dim), ReLU()))
        self.lin_list = torch.nn.ModuleList()
        self.conv_list = torch.nn.ModuleList()
        for i in range(c_depth-1):
            self.conv_list.append(GINConv(Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                                            Linear(dim, dim), ReLU())))
        for i in range(l_depth):
            self.lin_list.append(Linear(dim,dim))
        self.classifier = Linear(dim*2, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        h = self.conv1(x, edge_index).relu()
        for layer in self.conv_list:
            h = layer(h, edge_index).relu()
        for layer in self.lin_list:
            h = layer(h).relu()
        h = F.dropout(h, p=self.dropout_chance, training=self.training)

        #Take the node embeddings and concat nodes for each edge
        e = torch.cat((h[edge_index[0]], h[edge_index[1]]), dim=1)
        out = self.classifier(e)

        return out, e
