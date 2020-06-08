# -*- coding: utf-8 -*-

import os
import torch
import random
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Planetoid
import torch_geometric.nn as pyg_nn
import numpy as np


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_data(folder="node_classify/cora", data_name="cora"):
    dataset = Planetoid(root=folder, name=data_name)
    return dataset


class GraphCNN(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super(GraphCNN, self).__init__()
        self.conv1 = pyg_nn.GCNConv(in_channels=in_c, out_channels=hid_c)
        self.conv2 = pyg_nn.GCNConv(in_channels=hid_c, out_channels=out_c)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        hid = self.conv1(x=x, edge_index=edge_index)
        hid = F.relu(hid)

        out = self.conv2(x=hid, edge_index=edge_index)
        out = F.log_softmax(out, dim=1)

        return out


class Model1(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super(Model1, self).__init__()
        hid_c2 = 30
        self.conv1 = pyg_nn.GCNConv(in_channels=in_c, out_channels=hid_c)
        self.conv2 = pyg_nn.GCNConv(in_channels=hid_c, out_channels=hid_c)
        self.conv3 = pyg_nn.GCNConv(in_channels=hid_c, out_channels=hid_c2)
        self.fn = nn.Linear(in_features=hid_c2, out_features=out_c, bias=True)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        hid = self.conv1(x=x, edge_index=edge_index)
        hid = nn.Dropout(p=0.2, inplace=False)(hid)
        hid = self.conv2(x=hid, edge_index=edge_index)
        hid = nn.Dropout(p=0.2, inplace=False)(hid)
        hid = self.conv3(x=hid, edge_index=edge_index)
        hid = self.fn(hid)
        out = nn.LogSoftmax(dim=1)(hid)
        return out

class Model2(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super(Model2, self).__init__()
        self.conv1 = pyg_nn.GATConv(in_channels=in_c, out_channels=hid_c, dropout=0.5)
        self.conv2 = pyg_nn.GATConv(in_channels=hid_c, out_channels=out_c, dropout=0.5)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        hid = self.conv1(x=x, edge_index=edge_index)
        hid = F.relu(hid)
        hid = self.conv2(x=hid, edge_index=edge_index)
        hid = F.relu(hid)
        out = nn.LogSoftmax(dim=1)(hid)
        return out


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    cora_dataset = get_data()
    print(cora_dataset[0])
    # todo list

    fix_seed(42)
    net = Model2(in_c = 1433, hid_c = 50, out_c = 8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = net.to(device)
    data = cora_dataset[0].to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    net.train()
    for epoch in range(240):
        optimizer.zero_grad()
        output = net(data)
        loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        print("Epoch", epoch + 1, "Loss", loss.item())

    net.eval()
    _, prediction = net(data).max(dim=1)
    target = data.y
    test_correct = prediction[data.test_mask].eq(target[data.test_mask]).sum().item()
    test_number = data.test_mask.sum().item()

    print("Accuracy of Test Samples: ", test_correct / test_number)

