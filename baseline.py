# -*- coding: utf-8 -*-
# xinye chen
# this code is the first version, has some mistake but can run thoroughly; better do not run it, use gcn.py and train.py instead.


import torch
import numpy as np
import matplotlib.pyplot as plt

class GCN(torch.nn.Module):
    def __init__(self, A, feat_dim, hidden_dim, num_class):
        super(GCN, self).__init__()
        self.W_N = torch.nn.Parameter(data=torch.rand(feat_dim, hidden_dim), requires_grad=True)
        self.W_W = torch.nn.Parameter(data=torch.rand(hidden_dim, num_class), requires_grad=True)
        
        D = torch.Tensor(np.diag(np.array(np.power(np.sum(A.data.numpy(), axis=0),-1))))
        #torch.Tensor(torch.diagonal(pow(torch.sum(self.A, axis=0),-0.5)))
        self.D = torch.Tensor(D).cuda()
        self.A = torch.Tensor(A).cuda()
        
    def gcn_layer(self, A, D, X, W):
        return torch.mm(torch.mm(torch.mm(D, A),X),W)
    
    def forward(self, X):
        hidden = torch.nn.ReLU()(self.gcn_layer(self.A, self.D, X, self.W_N))
        hidden = torch.nn.Dropout(p=0.5, inplace=False)(hidden)
        output = self.gcn_layer(self.A, self.D, hidden, self.W_W)
        output = torch.nn.LogSoftmax(dim=1)(output)
        return output
    
def adjacencyBuild(n, neg, num):
    """n: current node, neg: neighbour node, num: number of the nodes"""
    
    if len(n) != len(neg):
        print("error")
        return 
    N = len(n)
    A = np.zeros((num, num))
    for i in range(N):
        A[n[i], neg[i]] = 1
        A[neg[i],n[i]] = 1
    return A.T

feat_Matrix = torch.Tensor(np.load("data/features.npy"))
label_list = torch.Tensor(np.load("data/label_list.npy"))
cites = np.load("data/edge_index.npy")

node_num = len(feat_Matrix)
feat_dim = feat_Matrix.shape[1]
num_class = 7


A = adjacencyBuild(cites[0,:],cites[1,:], node_num)
A = A + np.eye(A.shape[0])
A = torch.Tensor(A)
feat_Matrix = torch.Tensor(feat_Matrix)


from torch_geometric.datasets import Planetoid
import random

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_data(folder="cora", data_name="cora"):
    dataset = Planetoid(root=folder, name=data_name)
    return dataset.shuffle()
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GCN(A, feat_dim, 16, num_class)

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

fix_seed(2020)
cora_dataset = get_data()
data = cora_dataset[0].to(device)
train_mask = data.train_mask
test_mask = data.test_mask

model.train()
losses = []
for epoch in range(200):
    optimizer.zero_grad()#model.zero_grad()
    output = model(feat_Matrix.cuda())

    loss = torch.nn.NLLLoss()(output[train_mask].cuda(), label_list[train_mask].long().cuda())
    loss.backward(retain_graph=True)
    optimizer.step()
    print("epoch", epoch + 1, "loss", loss.item())
    losses.append(loss.data.cpu().numpy())
    
    
model.eval()
_, prediction = model(feat_Matrix.cuda()).max(dim=1)
prediction = prediction.cpu()
correct = prediction[test_mask].eq(label_list[test_mask]).sum().item()
total = test_mask.sum().item()

print("accuracy of test samples: ", correct / total)


plt.plot(losses, label='losses', color='g')
plt.legend()
plt.show()