import torch
import numpy as np
import scipy.io as sio
import scipy.sparse as sp

def load_mat(dataset, datadir='./datas', device='cuda', preprocess=True, self_loop=False):
    data = sio.loadmat(f'{datadir}/{dataset}.mat')

    adj = data['Network']
    adj = process_adj(adj, self_loop=self_loop).to(device)

    features = data['Attributes']
    features = sp.lil_matrix(features)
    if preprocess:
        features = preprocess_features(features)
        features = l2_normalize_features(features)
    features = torch.FloatTensor(features.A).to(device)

    labels = data['Label'].flatten()
    labels = torch.LongTensor(labels).to(device)

    return adj, features, labels

def process_adj(adj, self_loop=False):
    if self_loop:
        adj = adj + sp.eye(adj.shape[0])

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = normalize_adj(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj

def normalize_adj(adj):
    rowsum = np.array(adj.sum(1))

    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.

    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)

    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)

    return torch.sparse_coo_tensor(indices, values, shape)

def preprocess_features(features):
    rowsum = np.array(features.sum(1))

    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.

    r_mat_inv = sp.diags(r_inv)

    features = r_mat_inv.dot(features)

    return features

def l2_normalize_features(features):
    feat_csr = features.tocsr()

    epsilon = 1e-10
    norm_values = np.sqrt(feat_csr.multiply(feat_csr).sum(axis=1))    
    norm_values[norm_values == 0] = epsilon

    feat_norm = feat_csr / norm_values

    return feat_norm
