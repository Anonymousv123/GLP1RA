# -*- coding: utf-8 -*-

import pandas as pd
import os
import matplotlib.pyplot as plt

PTH = '/.../data'

"""# PYG"""

!pip install torch-geometric

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import networkx as nx
import numpy as np

weighted_edgelist = pd.read_csv(os.path.join(PTH,'g_edgelist.csv'))
weighted_edgelist.columns = ['node_from', 'node_to','weight']
weighted_edgelist

# create node list from edgelist

nodes = pd.concat(
              [
                  edges[['node_from','weight']].rename(columns={'node_from': 'node'}),
                  edges[['node_to','weight']].rename(columns={'node_to': 'node'})
              ],
              axis = 0
).reset_index(drop=True)

nodes = nodes.groupby('node')['weight'].sum().reset_index()

# calc the frequency of each side effect

nodes['freq'] = nodes['weight']/nodes['weight'].max()

nodes.loc[ (nodes['freq'] < 0.0001),'group'] = 'very rare'
nodes.loc[ (nodes['freq'] >= 0.0001) & (nodes['freq'] < 0.001) ,'group'] = 'rare'
nodes.loc[ (nodes['freq'] >= 0.001) & (nodes['freq'] < 0.01) ,'group'] = 'infrequent'
nodes.loc[ (nodes['freq'] >= 0.01) & (nodes['freq'] < 0.1) ,'group'] = 'frequent'
nodes.loc[ (nodes['freq'] >= 0.1),'group'] = 'very frequent'

nodes['group'].value_counts()

# compbine very frequent and frequent groups

nodes['class'] = 0

nodes.loc[nodes['group'].isin( ['frequent', 'very frequent']), 'class'] = 1

nodes['class'].value_counts()

nodes

"""# 10-fold CV"""

import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.data import DataLoader

# Create a dictionary to map node names to node indices
node_to_index = {node: i for i, node in enumerate(nodes['node'])}

# Convert node names to node indices in the edge list
edges = weighted_edgelist[['node_from', 'node_to']].apply(
    lambda row: (node_to_index[row['node_from']], node_to_index[row['node_to']]), axis=1
)
edges = torch.tensor(list(edges), dtype=torch.long).T

# Compute node degrees
# Here, we use NetworkX to compute the degrees
import networkx as nx
G = nx.Graph()
G.add_edges_from(edges.numpy().T)
degrees = torch.tensor([d for _, d in G.degree()], dtype=torch.float).view(-1, 1)

# Create a PyG graph data object with node degrees as features
num_nodes = len(nodes)
node_features = degrees
graph_data = Data(x=node_features, edge_index=edges)

# Split the nodes into class 0 and class 1
class_0_nodes = nodes[nodes['class'] == 0]['node'].values
class_1_nodes = nodes[nodes['class'] == 1]['node'].values

# Assign class labels to nodes in the graph
graph_data.y = torch.zeros(num_nodes, dtype=torch.long)  # Initialize with zeros
class_0_indices = [node_to_index[node] for node in class_0_nodes]
class_1_indices = [node_to_index[node] for node in class_1_nodes]
graph_data.y[class_0_indices] = 0  # Class 0
graph_data.y[class_1_indices] = 1  # Class 1

# Define the GCN model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

# Define the 10-fold cross-validation
num_epochs = 50
num_nodes = len(nodes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

f1_scores = []
roc_auc_scores = []

# Initialize variables to store false positive rate and true positive rate
all_fpr = np.linspace(0, 1, 100)
mean_tpr = 0

for train_idx, test_idx in skf.split(range(num_nodes), graph_data.y.numpy()):
    model = GCN(in_channels=1, out_channels=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.NLLLoss()

    train_loader = DataLoader([graph_data], batch_size=32, shuffle=False)
    test_loader = DataLoader([graph_data], batch_size=32, shuffle=False)

    for epoch in range(num_epochs):
        model.train()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output[train_idx], data.y[train_idx])
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        output = model(graph_data).cpu().numpy()


    fpr, tpr, _ = roc_curve(graph_data.y[test_idx].numpy(), output[test_idx][:, 1])
    mean_tpr += np.interp(all_fpr, fpr, tpr)

    f1 = f1_score(graph_data.y[test_idx].numpy(), output[test_idx].argmax(axis=1), average='binary')
    roc_auc = roc_auc_score(graph_data.y[test_idx].numpy(), output[test_idx][:, 1])

    f1_scores.append(f1)
    roc_auc_scores.append(roc_auc)

mean_tpr /= skf.get_n_splits()

# Calculate the mean ROC-AUC score
mean_auc = auc(all_fpr, mean_tpr)


print("Average F1-score:", sum(f1_scores) / len(f1_scores))
print("Average ROC-AUC:", sum(roc_auc_scores) / len(roc_auc_scores))


# Plot the ROC-AUC curve
plt.figure(figsize=(8, 8))
plt.plot(all_fpr, mean_tpr, color='b', label=f'Mean ROC-AUC = {mean_auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2, label='Random')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
