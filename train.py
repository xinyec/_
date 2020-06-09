import torch
import random
import numpy as np
from gcn import GCN
import matplotlib.pyplot as plt

def adjacencyBuild(n, neg, num):
    """n: current node, neg: neighbour node, num: number of the nodes"""
    if len(n) != len(neg):
        print("error")
        return 
    N = len(n)
    A = np.zeros((num, num))
    for i in range(N):
        A[n[i]-1, neg[i]-1] = 1
    return A.T


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

fix_seed(2020)
feat_Matrix = torch.Tensor(np.load("data/feat_Matrix.npy"))
label_list = torch.Tensor(np.load("data/label_list.npy"))
cites = np.load("data/cites.npy")

node_num = len(feat_Matrix)
feat_dim = feat_Matrix.shape[1]
num_class = 7

A = adjacencyBuild(cites[0],cites[1], node_num)
A = A + np.eye(A.shape[0])
A = torch.Tensor(A)
feat_Matrix = torch.Tensor(feat_Matrix)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_mask = torch.Tensor(np.load('data/train_mask.npy')).bool()
test_mask = torch.Tensor(np.load('data/test_mask.npy')).bool()

model = GCN(A, feat_dim, 16, num_class)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

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
test_correct = prediction[test_mask].eq(label_list[test_mask]).sum().item()
test_number = test_mask.sum().item()

print("Accuracy of Test Samples: ", test_correct / test_number)

plt.plot(losses, label='losses', color='g')
plt.legend()
plt.show()