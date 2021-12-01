import torch
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
from torch_geometric.nn import GCNConv, GINConv, GATv2Conv
import random
import math
from sklearn.model_selection import KFold

seed = 2
torch.manual_seed(seed)

class LinearNN(torch.nn.Module):
    def __init__(self, dataset):
        super(LinearNN, self).__init__()
        torch.manual_seed(seed)
        self.l1 = Linear(dataset.num_features*2, 24)
        self.l2 = Linear(24, 18)
        self.l3 = Linear(18, 12)
        self.classifier = Linear(12, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        #Take the node embeddings and concat nodes for each edge
        e = torch.cat((x[edge_index[0]], x[edge_index[1]]), dim=1)

        h = self.l1(e)
        h = h.tanh()
        h = self.l2(h)
        h = h.tanh()
        h = self.l3(h)
        h = h.tanh()
        out = self.classifier(h)

        return out, h

# GCN class from tutorial notebook 1
# Modified the seed, the constructor, from Tony
class SimpleGCN(torch.nn.Module):
    def __init__(self, dataset):
        super(SimpleGCN, self).__init__()
        torch.manual_seed(seed)
        self.conv1 = GCNConv(dataset.num_features, 18)
        self.conv2 = GCNConv(18, 24)
        self.conv3 = GCNConv(24, dataset.num_features*2)
        self.lin1 = Linear(dataset.num_features*2, 24)
        self.lin2 = Linear(24, 18)
        self.lin3 = Linear(18, 12)
        self.classifier = Linear(12*2, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()
        h = self.lin1(h)
        h = h.tanh()
        h = self.lin2(h)
        h = h.tanh()
        h = self.lin3(h)
        h = h.tanh()# Final GNN embedding space.

        #Take the node embeddings and concat nodes for each edge
        e = torch.cat((h[edge_index[0]], h[edge_index[1]]), dim=1)

        # Apply a final (linear) classifier.
        out = self.classifier(e)

        return out, e


class GATCONV(torch.nn.Module):
    """
    A graph attention network with 4 graph layers and 2 linear layers.
    Uses v2 of graph attention that provides dynamic instead of static attention.
    The graph layer dimension and number of attention heads can be specified.
    """
    def __init__(self, dataset, dim=8, num_heads=4):
        super(GATCONV, self).__init__()
        torch.manual_seed(seed)
        self.conv1 = GATv2Conv(in_channels=dataset.num_features, out_channels=dim, heads=num_heads)
        # The graph attention layer provides a re-weighted version of the output node representation for every
        # attention heads in the default setting, which is why we multiply by num_heads
        self.conv2 = GATv2Conv(in_channels=dim * num_heads, out_channels=dim, heads=num_heads)
        self.conv3 = GATv2Conv(in_channels=dim * num_heads, out_channels=dim, heads=num_heads)
        self.conv4 = GATv2Conv(in_channels=dim * num_heads, out_channels=dim, heads=num_heads)
        self.lin1 = Linear(dim * num_heads, dim)
        self.lin2 = Linear(dim*2, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        h = self.conv1(x, edge_index).relu()
        h = self.conv2(h, edge_index).relu()
        h = self.conv3(h, edge_index).relu()
        h = self.conv4(h, edge_index).relu()
        h = self.lin1(h).relu()
        h = F.dropout(h, p=0.5, training=self.training)

        #Take the node embeddings and concat nodes for each edge
        e = torch.cat((h[edge_index[0]], h[edge_index[1]]), dim=1)

        out = self.lin2(e)

        return out, e


class DeepGCN(torch.nn.Module):
    """
    A graph convolutional network with 5 graph layers and 1 linear layer.
    The dimension of the hidden layers is customizable and decreases
    at the higher layers.
    """
    def __init__(self, dataset, hidden_channels=30):
        super(DeepGCN, self).__init__()
        torch.manual_seed(seed)
        # Automatically determine how much to shrink the dimensionality of the hidden layers
        hidden_channels_2 = max(1, math.floor(hidden_channels / 2))
        hidden_channels_3 = max(1, math.floor(hidden_channels_2 / 2))

        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels_2)
        self.conv4 = GCNConv(hidden_channels_2, hidden_channels_2)
        self.conv5 = GCNConv(hidden_channels_2, dataset.num_features*2)
        self.lin1 = Linear(dataset.num_features*2, 24)
        self.lin2 = Linear(24, 18)
        self.lin3 = Linear(18, 12)
        self.classifier = Linear(12*2, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()
        h = self.conv4(h, edge_index)
        h = h.tanh()
        h = self.conv5(h, edge_index)
        h = h.tanh()
        h = self.lin1(h)
        h = h.tanh()
        h = self.lin2(h)
        h = h.tanh()
        h = self.lin3(h)
        h = h.tanh()# Final GNN embedding space.# Final GNN embedding space.

        #Take the node embeddings and concat nodes for each edge
        e = torch.cat((h[edge_index[0]], h[edge_index[1]]), dim=1)

        # Apply a final (linear) classifier.
        out = self.classifier(e)

        return out, e


# See https://github.com/pyg-team/pytorch_geometric/blob/master/examples/mutag_gin.py
##### Use this version with cross entropy loss instead of the GIN class above
class GIN2(torch.nn.Module):
    def __init__(self, dataset, dim=16):
        super(GIN2, self).__init__()
        torch.manual_seed(seed)
        self.conv1 = GINConv(Sequential(Linear(dataset.num_features, dim), BatchNorm1d(dim), ReLU(),
                                        Linear(dim, dim), ReLU()))
        self.conv2 = GINConv(Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                                        Linear(dim, dim), ReLU()))
        self.conv3 = GINConv(Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                                        Linear(dim, dim), ReLU()))
        self.conv4 = GINConv(Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                                        Linear(dim, dim), ReLU()))
        self.lin1 = Linear(dim, dim)
        self.lin2 = Linear(dim*2, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        h = self.conv1(x, edge_index)
        h = self.conv2(h, edge_index)
        h = self.conv3(h, edge_index)
        h = self.conv4(h, edge_index)
        h = self.lin1(h).relu()
        h = F.dropout(h, p=0.5, training=self.training)

        #Take the node embeddings and concat nodes for each edge
        e = torch.cat((h[edge_index[0]], h[edge_index[1]]), dim=1)

        out = self.lin2(e)

        return out, e

