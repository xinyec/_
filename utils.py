# -*- coding: utf-8 -*-

#import torch
import numpy as np
from torch_geometric.datasets import Planetoid

def get_data(folder="/Cora", data_name="cora"):
    dataset = Planetoid(root=folder, name=data_name)
    return dataset

def load_standata():
    """get data from Planetoid"""
    cora_dataset = get_data()
    data = cora_dataset[0]
    train_mask = data.train_mask.numpy()
    test_mask = data.test_mask.numpy()
    edge_index = data.edge_index.numpy()
    features = data.x.numpy()
    np.save("data/train_mask.npy", train_mask)
    np.save("data/test_mask.npy", test_mask)
    np.save("data/edge_index.npy", edge_index)
    np.save("data/features.npy", features)
    np.save("data/label_list.npy", data.y.numpy())

def load_data():
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
    cites = np.array([[paper_dict[i[0]], paper_dict[i[1]]] for i in cites], np.int64).T
    cites = np.concatenate((cites, cites[::-1, :]), axis=1)

    # input
    #node_num = len(paper_list)
    #feat_dim = feat_list.shape[1]
    #num_class = len(labels)

    feat_Matrix = feat_list.astype(np.float32)
    label_list = np.array([label_dict[i] for i in label_list])
    #label_list = torch.from_numpy(label_list)

    np.save("data/feat_Matrix.npy",feat_Matrix)
    np.save("data/label_list.npy",label_list)
    np.save("data/cites.npy",cites)


if __name__ == "__main__":
    load_standata()