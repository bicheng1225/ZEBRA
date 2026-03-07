import torch
import random
import warnings
import argparse

import numpy as np

from sklearn.metrics import roc_auc_score, average_precision_score

from model import ZEBRA
from dataloader import load_mat

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def main(args):
    auroc_history = []
    auprc_history = []

    for seed in [513]:
        print(f"{'='*22} Experiment seed: {seed} {'='*22}")

        set_seed(seed)

        adj, features, labels = load_mat(args.dataset, args.datadir, args.device, self_loop=False)

        num_nodes = features.shape[0]

        adj_in = adj.T
        adj_out = adj

        model = ZEBRA(args, labels, num_nodes)

        scores = model(adj_in, adj_out, features, args.alpha, args.beta, args.num_hops, args.num_topk)

        scores = scores.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        labeled_mask = (labels != -1)

        scores = scores[labeled_mask]
        labels = labels[labeled_mask]

        auroc_history.append(roc_auc_score(labels, scores))
        auprc_history.append(average_precision_score(labels, scores))

    # calculate the mean and std of AUROC and AUPRC
    auroc_mean = np.array(auroc_history).mean()
    auprc_mean = np.array(auprc_history).mean()
    auroc_std = np.array(auroc_history).std()
    auprc_std = np.array(auprc_history).std()

    print(f"{'='*22}     Metrices      {'='*22}")
    print('Dataset: {}, AUROC: {:.4f}+-{:.3f}, AUPRC: {:.4f}+-{:.3f}'.format(args.dataset, auroc_mean, auroc_std, auprc_mean, auprc_std))

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', default="./datas", help='data path')
    parser.add_argument('--dataset', default="Reddit", help='data name')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha')
    parser.add_argument('--beta', type=float, default=0.9, help='beta')
    parser.add_argument('--num-topk', type=int, default=30, help='the num of anchor node')
    parser.add_argument('--num-hops', type=int, default=4, help='the num of propagation iteration')
    parser.add_argument('--device', default='cuda', type=str, help='cuda/cpu')

    args = parser.parse_args()

    main(args)
