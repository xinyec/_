import torch
import numpy as np

class GCN(torch.nn.Module):
    def __init__(self, A, feat_dim, hidden_dim, num_class, droprate=0.1):
        super(GCN, self).__init__()
        self.W_N = torch.nn.Parameter(data=torch.rand(feat_dim, hidden_dim), requires_grad=True)
        self.W_W = torch.nn.Parameter(data=torch.rand(hidden_dim, num_class), requires_grad=True)
        
        D = torch.Tensor(np.diag(np.array(np.power(np.sum(A.data.numpy(), axis=0),-1))))
        #torch.Tensor(torch.diagonal(pow(torch.sum(self.A, axis=0),-0.5)))
        self.D = torch.Tensor(D).cuda()
        self.A = torch.Tensor(A).cuda()
        self.droprate=droprate

    def gcn_layer(self, A, D, X, W):
        return torch.mm(torch.mm(torch.mm(D, A),X),W)
    
    def forward(self, X):
        hidden = torch.nn.ReLU()(self.gcn_layer(self.A, self.D, X, self.W_N))
        hidden = torch.nn.Dropout(p=self.droprate, inplace=False)(hidden)
        output = self.gcn_layer(self.A, self.D, hidden, self.W_W)
        output = torch.nn.LogSoftmax(dim=1)(output)
        return output
    
