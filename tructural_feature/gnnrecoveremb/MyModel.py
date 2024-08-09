import os
import numpy as np
# import torch_sparse
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from scipy.sparse import csr_matrix
import networkx as nx
from scipy.linalg import orthogonal_procrustes
import torch
import torch.optim as optim
from torch_geometric.data import Data
from sklearn.manifold import TSNE
import pandas as pd
from util import Net, GIN, GAT, moon, stationary, reconstruct, dG



np.random.seed(0)
torch.manual_seed(0)

n = 5000
m = 1534
x, n = moon(n)
n_train = int(n * 0.7)
train_ind = torch.randperm(n)[:n_train]
test_ind = torch.LongTensor(list(set(np.arange(n)) - set(train_ind.tolist())))
K = int(np.sqrt(n) * np.log2(n) / 10)
D = pairwise_distances(x)
fr = np.arange(n).repeat(K).reshape(-1)
to = np.argsort(D, axis=1)[:, 1:K + 1].reshape(-1)
A = csr_matrix((np.ones(n * K) / K, (fr, to)))

edge_index = np.vstack([fr, to])
edge_index = torch.tensor(edge_index, dtype=torch.long)
X = torch.tensor([[K, n] for i in range(n)], dtype=torch.float)



net = GIN(m)
optimizer = optim.Adam(net.parameters(), lr=0.001)
net.train()
for epoch in range(100):
    ind = torch.eye(n)[:, torch.randperm(n)[:m]]
    X_extended = torch.hstack([X, ind])
    data = Data(x=X_extended, edge_index=edge_index)
    rec = net(data)
    loss = dG(torch.FloatTensor(x)[train_ind], rec[train_ind])
    print(float(loss))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



# 读入数据
feas = pd.read_excel('disease_drug_vector.xlsx')
edges  = pd.read_excel('drug_dis_id1.xlsx')
X = torch.tensor(feas.iloc[:,1:3].values, dtype=torch.float)
feas = torch.tensor(feas.iloc[:,3:].values, dtype=torch.float)
X_extended = torch.hstack([X, feas])
edges = edges[['dis', 'drug']]
edges = edges.T
edges =edges.values
edges = torch.tensor(edges, dtype=torch.long)
data = Data(x=X_extended, edge_index=edges)

output_features = net(data)

output_features = pd.DataFrame(data=output_features.detach().numpy())
output_features.to_csv('output_features.csv', index=False)
