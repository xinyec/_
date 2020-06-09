# -*- coding: utf-8 -*-

import torch

class GCNlayer(torch.nn.Module):
    def __init__(self, feat_dim, out_dim, bias=False):
        super(GCNlayer, self).__init__()
        self.W = torch.nn.Parameter(data=torch.rand(feat_dim, out_dim), requires_grad=True)
        if bias:
            self.bias = torch.nn.Parameter(
                data=torch.empty(1, out_dim).uniform_(0, 0.05), requires_grad=True)
        else:
            self.bias = None

    def forward(self, A, X):
        D = torch.diag(pow(torch.sum(A, axis=0), -1 / 2))
        if self.bias != None:
            output = torch.mm(torch.mm(torch.mm(torch.mm(D, A), D), X), self.W) + self.bias
        else:
            output = torch.mm(torch.mm(torch.mm(torch.mm(D, A), D), X), self.W)
        return (output)


class GCN(torch.nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_class, droprate=0.1, bias=True):
        super(GCN, self).__init__()
        self.GConv1 = GCNlayer(feat_dim, hidden_dim, bias)
        self.GConv2 = GCNlayer(hidden_dim, num_class, bias)
        self.dropout = torch.nn.Dropout(p=droprate, inplace=False)
        self.relu = torch.nn.ReLU()

    def forward(self, A, X):  # input -> hidden -> dropout -> logsoftmax, forward propagation
        #D = torch.diag(pow(torch.sum(A, axis=0), -1 / 2))
        hidden = self.relu(self.GConv1(A, X))
        hidden = self.dropout(hidden)
        output = self.GConv2(A, hidden)
        output = torch.nn.LogSoftmax(dim=1)(output)
        return output
