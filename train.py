import torch
import random
import numpy as np
from gcn import GCN
#import matplotlib.pyplot as plt

def adjacencyBuild(n, neg, num):
    """n: current node, neg: neighbour node, num: number of the nodes"""
    if len(n) != len(neg):
        print("error")
        return
    N = len(n)
    A = np.zeros((num, num))
    for i in range(N):
        A[n[i], neg[i]] = 1
        A[neg[i], n[i]] = 1
    return A

def acc_calc(true, pred):
    correct = pred.eq(true).sum().item()
    total = test_mask.sum().item()
    return (correct / total)

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

fix_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_mask = torch.Tensor(np.load("data/train_mask.npy")).bool()
test_mask = torch.Tensor(np.load("data/test_mask.npy")).bool()
features = torch.Tensor(np.load("data/features.npy"))
label_list = torch.Tensor(np.load("data/label_list.npy"))
edge_index = np.load("data/edge_index.npy")

node_num = features.shape[0]
A = adjacencyBuild(edge_index[0,:],edge_index[1,:], node_num)
A = A + np.eye(A.shape[0])
A = torch.Tensor(A)

feat_dim = features.shape[1]
num_class = int(label_list.max().data.numpy() + 1) #7

model = GCN(A.to(device), feat_dim, 16, num_class, droprate=0.5)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()
losses = []

for epoch in range(200):
    optimizer.zero_grad()
    output = model(features.to(device))

    loss = torch.nn.NLLLoss()(output[train_mask].to(device), label_list[train_mask].long().to(device))
    loss.backward(retain_graph=True)
    optimizer.step()
    loss_val = torch.nn.NLLLoss()(output[test_mask].to(device), label_list[test_mask].long().to(device))

    print('epoch: {:03d}'.format(epoch+1),
          'loss_train: {:.3f}'.format(loss.item()),
          'loss_val: {:.3f}'.format(loss_val.item()))

    losses.append(loss.data.cpu().numpy())

model.eval()
_, prediction = model(features.cuda()).max(dim=1)
acc = acc_calc(label_list[test_mask].to(device),prediction[test_mask].to(device))
print("accuracy of in test set: ", acc)

#plt.plot(losses, label='losses', color='g')
#plt.legend()
#plt.show()