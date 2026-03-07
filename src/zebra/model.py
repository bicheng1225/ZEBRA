import torch

import numpy as np
import torch.nn.functional as F

class ZEBRA():
    def __init__(self, args, labels, num_nodes):
        self.device = args.device

        self.labels = labels
        self.num_nodes = num_nodes

    def __call__(self, adj_in, adj_out, features, alpha, beta, num_hops, num_topk) -> torch.Tensor:
        propagated_features_in = self.propagated(adj_in, features, num_hops)
        weighted_features_in, local_scores_in = self.Affinity_Gated_Residual_Encoder(propagated_features_in[1:])

        propagated_features_out = self.propagated(adj_out, features, num_hops)
        weighted_features_out, local_scores_out = self.Affinity_Gated_Residual_Encoder(propagated_features_out[1:])

        weighted_features = weighted_features_in + weighted_features_out
        original_features = propagated_features_in[0]

        propagated_similarities = torch.nn.functional.cosine_similarity(weighted_features, original_features)

        max_mask_matrix, min_mask_matrix = self.Anchor_node_selection(propagated_similarities, num_topk)

        final_scores = self.Anchor_Guided_Anomaly_Scoring(weighted_features, alpha, beta, max_mask_matrix, min_mask_matrix)

        return final_scores

    def propagated(self, adj, features, num_hops):
        propagated_features = [features]

        tf = adj * torch.pow(torch.sparse.sum(adj, dim=0).to_dense().squeeze(), -1)

        itf = np.log(adj.shape[0] * torch.pow(torch.sparse.sum(self.binarize_sparse_tensor(adj), dim=0).to_dense().squeeze(), -1))

        wTW_sparse = (tf * itf).to(self.device)

        for hop in range(num_hops):
            last_features = propagated_features[-1]

            aggregated_features = torch.spmm(wTW_sparse, last_features)
            propagated_features.append(last_features - aggregated_features)

        return propagated_features

    def Affinity_Gated_Residual_Encoder(self, propagated_features):
        attention_scores = []
        for layer in range(len(propagated_features)):
            noramlized_features = torch.norm(propagated_features[layer], 2, 1, keepdim=True).add(1e-10) # L2 norm (N, 1)

            attention_scores.append(noramlized_features)

        attention_scores = torch.cat(attention_scores, dim=1) # (N, L)
        attention_weights = F.softmax(attention_scores, dim=1) # (N, L)

        expanded_weights = attention_weights.unsqueeze(1) # (N, 1, L)
        permuted_features = torch.stack(propagated_features, dim=1) # (N, L, D)

        weighted_features = torch.bmm(expanded_weights, permuted_features) # (N, 1, D)
        weighted_features = weighted_features.squeeze(1) # (N, D)

        local_scores = torch.norm(weighted_features, 2, 1).add(1e-10) # (N,)

        return weighted_features, local_scores

    def Anchor_node_selection(self, similarities, num_topk):
        max_value, max_index = torch.topk(similarities, num_topk, largest=True)
        min_value, min_index = torch.topk(similarities, num_topk, largest=False)

        print('top-{} max similarity values: min={:.4f}, max={:.4f}, mean={:.4f}'.format(num_topk, max_value.min(), max_value.max(), max_value.mean()))
        print('top-{} min similarity values: min={:.4f}, max={:.4f}, mean={:.4f}'.format(num_topk, min_value.min(), min_value.max(), min_value.mean()))

        max_mask_matrix = torch.zeros(self.num_nodes, dtype=torch.bool, device=self.device)
        min_mask_matrix = torch.zeros(self.num_nodes, dtype=torch.bool, device=self.device)

        max_mask_matrix[max_index] = 1
        min_mask_matrix[min_index] = 1

        return max_mask_matrix, min_mask_matrix

    def Anchor_Guided_Anomaly_Scoring(self, weighted_features, alpha, beta, max_mask_matrix, min_mask_matrix):
        positive_index = torch.nonzero(max_mask_matrix == True).squeeze(1).tolist()
        positive_anchor_nodes = weighted_features[positive_index]

        negative_index = torch.nonzero(min_mask_matrix == True).squeeze(1).tolist()
        negative_anchor_nodes = weighted_features[negative_index]

        positive_distances = torch.cdist(weighted_features, positive_anchor_nodes)
        negative_distances = torch.cdist(weighted_features, negative_anchor_nodes)

        positive_mean = torch.mean(positive_distances, dim=-1)
        negative_mean = torch.mean(negative_distances, dim=-1)

        scores = (alpha * positive_mean) - (beta * negative_mean)

        return scores

    def binarize_sparse_tensor(self, adj):
        adj = adj.coalesce()

        binary_values = torch.ones(adj._nnz(), dtype=adj.values().dtype, device=adj.values().device)

        binary_adj = torch.sparse_coo_tensor(indices=adj.indices(), values=binary_values, size=adj.size()).coalesce()

        return binary_adj
