import numpy as np
import scipy.sparse as sp
import torch


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c  in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32) # Python 3.x 返回迭代器。
    return labels_onehot


def load_data(path="./data/cora/", dataset='cora'):
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # 由于文件中节点并非是按顺序排列的，因此建立一个编号为0-(node_size-1)的哈希表idx_map，
    # 哈希表中每一项为id: number，即节点id对应的编号为number
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unorderd = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unorderd.flatten())), dtype=np.int32).reshape((edges_unorderd.shape))
    # 注意这里的adj是有向图的，因为.coo_matrix是根据(x,y)赋值1的，所以我们需要将有向图转为无向图
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, -1])),
                               shape=(labels.shape[0], labels.shape[0]),
                               dtype=np.float32)
    # 无向图——>有向图
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # 定义特征，调用归一化函数
    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    # 这里将onthot label转回index, 返回的是元组（）
    labels = torch.LongTensor(np.where(labels)[1])
    # 邻接矩阵转为tensor处理
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)

    return mx


def accuracy(output, labels):
    '''

    :param output:
    :param labels:
    :return:
    tensor类型数.max(dim) 返回
        values = tensor([4, 76]),
        indices = tensor([3, 3]))
    '''
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

# 把一个sparse matrix转为torch稀疏张量
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    '''

    :param sparse_mx:
    :return:

    numpy中的ndarray转化成pytorch中的tensor : torch.from_numpy()
    pytorch中的tensor转化成numpy中的ndarray : numpy()
    '''
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))

    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape) # 设置torch类型数据size大小的函数

    return torch.sparse.FloatTensor(indices, values, shape)


