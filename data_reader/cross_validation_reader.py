
import torch
from random import seed
import numpy as np
import random
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader

from utils.utils import get_graph_diameter


rnd_state = np.random.RandomState(seed(1))


def collate_batch(batch):
    '''
    Creates a batch of same size graphs by zero-padding node features and adjacency matrices up to
    the maximum number of nodes in the CURRENT batch rather than in the entire dataset.
    Graphs in the batches are usually much smaller than the largest graph in the dataset, so this method is fast.
    :param batch: batch in the PyTorch Geometric format or [node_features*batch_size, A*batch_size, label*batch_size]
    :return: [node_features, A, graph_support, N_nodes, label]
    '''
    B = len(batch)
    N_nodes = [len(batch[b].x) for b in range(B)]
    C = batch[0].x.shape[1]

    N_nodes_max = int(np.max(N_nodes))

    graph_support = torch.zeros(B, N_nodes_max)
    A = torch.zeros(B, N_nodes_max, N_nodes_max)
    x = torch.zeros(B, N_nodes_max, C)
    for b in range(B):
        x[b, :N_nodes[b]] = batch[b].x
        A[b].index_put_((batch[b].edge_index[0], batch[b].edge_index[1]), torch.Tensor([1]))
        graph_support[b][:N_nodes[b]] = 1  # mask with values of 0 for dummy (zero padded) nodes, otherwise 1

    N_nodes = torch.from_numpy(np.array(N_nodes)).long()
    labels = torch.from_numpy(np.array([batch[b].y  for b in range(B)])).long()
    return [x, A, graph_support, N_nodes, labels]

def split_ids(ids, folds=10):
    n = len(ids)
    stride = int(np.ceil(n / float(folds)))
    test_ids = [ids[i: i + stride] for i in range(0, n, stride)]

    assert np.all(
        np.unique(np.concatenate(test_ids)) == sorted(ids)), 'some graphs are missing in the test sets'
    assert len(test_ids) == folds, 'invalid test sets'
    valid_ids = []
    train_ids = []

    for fold in range(folds):
        valid_fold = []
        while len(valid_fold) < stride:
            id = random.choice(ids)
            if id not in test_ids[fold] and id not in valid_fold:
               valid_fold.append(id)

        valid_ids.append(np.asarray(valid_fold))
        train_ids.append(np.array([e for e in ids if e not in test_ids[fold] and e not in valid_ids[fold]]))
        assert len(train_ids[fold]) + len(test_ids[fold]) + len(valid_ids[fold]) == len(np.unique(list(train_ids[fold]) + list(test_ids[fold]) + list(valid_ids[fold]))) == n, 'invalid splits'


    return train_ids, test_ids, valid_ids


def getcross_validation_split(dataset_path='~/Dataset/MUTAG', dataset_name='MUTAG', n_folds=2, batch_size=1, use_node_attr=False):

    dataset = TUDataset(root=dataset_path, name=dataset_name, pre_transform=get_graph_diameter, use_node_attr=use_node_attr)
    train_ids, test_ids, valid_ids = split_ids(rnd_state.permutation(len(dataset)), folds=n_folds)
    splits=[]

    for fold_id in range(n_folds):
        loaders = []
        for split in [train_ids, test_ids, valid_ids]:
            # print(torch.from_numpy((train_ids if split.find('train') >= 0 else test_ids)[fold_id]))
            # print("---")
            gdata = dataset[torch.from_numpy(split[fold_id])]

            loader = DataLoader(gdata,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=4)#,
                                #collate_fn=collate_batch)
            loaders.append(loader)
        splits.append(loaders)
        # print("---")

    return splits #0-train, 1-test, 2-valid

if __name__ == '__main__':
    cv_splits=getcross_validation_split(n_folds=10)

    split_1=cv_splits[2]
    train=split_1[0]
    test=split_1[1]
    valid=split_1[2]
    for batch in train:
        print(batch.diameter)
    print(len(train))
    print(len(test))
    print(len(valid))