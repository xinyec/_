import torch
import numpy as np


content_path = "./cora/cora.content"
cite_path = "./cora/cora.cites"

with open(content_path, "r") as fp:
    contents = fp.readlines()
with open(cite_path, "r") as fp:
    cites = fp.readlines()

contents = np.array([np.array(l.strip().split("\t")) for l in contents])
paper_list, feat_list, label_list = np.split(contents, [1,-1], axis=1)
paper_list, label_list = np.squeeze(paper_list), np.squeeze(label_list)

# paper index
paper_dict = dict([(key, val) for val, key in enumerate(paper_list)])

# label index 
labels = list(set(label_list))
label_dict = dict([(key, val) for val, key in enumerate(labels)])

# edge index
cites = [i.strip().split("\t") for i in cites]
cites = np.array([[paper_dict[i[0]], paper_dict[i[1]]] for i in cites], np.int64).T   # (2, edge)
cites = np.concatenate((cites, cites[::-1, :]), axis=1)  # (2, 2*edge) or (2, E)

# input
node_num = len(paper_list)
feat_dim = feat_list.shape[1]
num_class = len(labels)

feat_Matrix = feat_list.astype(np.float32)
label_list = np.array([label_dict[i] for i in label_list])
label_list = torch.from_numpy(label_list)


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
        A[n[i]-1, neg[i]-1] = 1
    return A.T

A = adjacencyBuild(cites[0],cites[1], node_num)
A = A + np.eye(A.shape[0])
display(A.shape)
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
    
    

#train_mask = torch.zeros(node_num, dtype=torch.bool)
#train_mask[:node_num - 1000] = 1                  # 1700 training
#test_mask = torch.zeros(node_num, dtype=torch.bool)
#test_mask[node_num - 1000] = 1                    # 1000 test


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

    loss = torch.nn.NLLLoss()(output[test_mask].cuda(), label_list[test_mask].long().cuda())
    loss.backward(retain_graph=True)
    optimizer.step()
    print("epoch", epoch + 1, "loss", loss.item())
    losses.append(loss.data.cpu().numpy())
    
    
 model.eval()
_, prediction = model(feat_Matrix.cuda()).max(dim=1)
prediction = prediction.cpu()
test_correct = prediction[train_mask].eq(label_list[train_mask]).sum().item()
test_number = train_mask.sum().item()

print("Accuracy of Test Samples: ", test_correct / test_number)

import matplotlib.pyplot as plt
%matplotlib inline
plt.plot(losses, label='losses', color='g')
plt.legend()
plt.show()