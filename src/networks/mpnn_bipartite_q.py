import torch
import torch.nn as nn
import torch.nn.functional as F


class MPNN(nn.Module):
    def __init__(self,
                 n_obs_in_g=1,
                 n_obs_in_p=1,
                 n_layers=3,
                 n_features=64,
                 tied_weights=False,
                 n_hid_readout=[], ):

        super().__init__()

        self.n_obs_in_g = n_obs_in_g
        self.n_obs_in_p = n_obs_in_p
        self.n_layers = n_layers
        self.n_features = n_features
        self.tied_weights = tied_weights

        self.node_init_embedding_layer_g = nn.Sequential(
            nn.Linear(n_obs_in_g, n_features, bias=False),
            nn.ReLU()
        )
        self.node_init_embedding_layer_p = nn.Sequential(
            nn.Linear(n_obs_in_p, n_features, bias=False),
            nn.ReLU()
        )

        self.edge_embedding_layer_g = EdgeAndNodeEmbeddingLayer(n_obs_in_p, n_features)
        self.edge_embedding_layer_p = EdgeAndNodeEmbeddingLayer(n_obs_in_g, n_features)

        if self.tied_weights:
            self.update_node_embedding_layer_g = UpdateNodeEmbeddingLayer(n_features)
            self.update_node_embedding_layer_p = UpdateNodeEmbeddingLayer(n_features)
        else:
            self.update_node_embedding_layer_g = nn.ModuleList(
                [UpdateNodeEmbeddingLayer(n_features) for _ in range(self.n_layers)])
            self.update_node_embedding_layer_p = nn.ModuleList(
                [UpdateNodeEmbeddingLayer(n_features) for _ in range(self.n_layers)])

        self.readout_layer = ReadoutLayer(n_features, n_hid_readout)

    @torch.no_grad()
    def get_normalisation(self, adj):
        norm = torch.sum((adj != 0), dim=-1).unsqueeze(-1)
        norm_fill_1 = norm.clone()
        norm_fill_1[norm == 0] = 1
        return norm.float(), norm_fill_1.float()

    def forward(self, obs):
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)

        # obs.transpose_(-1, -2)

        # Calculate features to be used in the MPNN
        node_features_g_first = obs[:, :, 0:self.n_obs_in_g-1]

        # Get graph adj matrix.
        adj = obs[:, :, self.n_obs_in_g-1:]
        adj_T = adj.clone()
        adj_T.transpose_(-1, -2)
        # adj_conns = (adj != 0).type(torch.FloatTensor).to(adj.device)

        gnum = adj.shape[1]
        pnum = adj.shape[2]
        norm_g, norm_fill_1_g = self.get_normalisation(adj)
        norm_p, norm_fill_1_p = self.get_normalisation(adj_T)
        node_features_g = torch.cat([node_features_g_first, norm_g/pnum], dim=-1)
        node_features_p = norm_p / gnum

        # norm = self.get_normalisation(adj)

        init_node_embeddings_g = self.node_init_embedding_layer_g(node_features_g)
        init_node_embeddings_p = self.node_init_embedding_layer_p(node_features_p)

        edge_embeddings_g = self.edge_embedding_layer_g(node_features_p, adj, norm_g, norm_fill_1_g)
        edge_embeddings_p = self.edge_embedding_layer_p(node_features_g, adj_T, norm_p, norm_fill_1_p)

        # Initialise embeddings.
        current_node_embeddings_g = init_node_embeddings_g
        current_node_embeddings_p = init_node_embeddings_p

        if self.tied_weights:
            for _ in range(self.n_layers):
                next_node_embeddings_g = self.update_node_embedding_layer_g(current_node_embeddings_g,
                                                                            current_node_embeddings_p,
                                                                            edge_embeddings_g,
                                                                            norm_fill_1_g,
                                                                            adj)
                next_node_embeddings_p = self.update_node_embedding_layer_p(current_node_embeddings_p,
                                                                            current_node_embeddings_g,
                                                                            edge_embeddings_p,
                                                                            norm_fill_1_p,
                                                                            adj_T)
                current_node_embeddings_g = next_node_embeddings_g
                current_node_embeddings_p = next_node_embeddings_p
        else:
            for i in range(self.n_layers):
                next_node_embeddings_g = self.update_node_embedding_layer_g[i](current_node_embeddings_g,
                                                                               current_node_embeddings_p,
                                                                               edge_embeddings_g,
                                                                               norm_fill_1_g,
                                                                               adj)
                next_node_embeddings_p = self.update_node_embedding_layer_p[i](current_node_embeddings_p,
                                                                               current_node_embeddings_g,
                                                                               edge_embeddings_p,
                                                                               norm_fill_1_p,
                                                                               adj_T)
                current_node_embeddings_g = next_node_embeddings_g
                current_node_embeddings_p = next_node_embeddings_p

        out = self.readout_layer(current_node_embeddings_g, current_node_embeddings_p)
        out = out.squeeze()

        return out


class EdgeAndNodeEmbeddingLayer(nn.Module):

    def __init__(self, n_obs_in, n_features):
        super().__init__()
        self.n_obs_in = n_obs_in
        self.n_features = n_features

        self.edge_embedding_NN = nn.Linear(n_obs_in, n_features - 1, bias=False)
        self.edge_feature_NN = nn.Linear(n_features, n_features, bias=False)

    def forward(self, node_features, adj, norm, norm_fill_1):
        # edge_features = torch.cat([adj.unsqueeze(-1),
        #                            node_features.unsqueeze(-2).transpose(-2, -3).repeat(1, adj.shape[-2], 1, 1)],
        #                           dim=-1)
        edge_features = node_features.unsqueeze(-2).transpose(-2, -3).repeat(1, adj.shape[-2], 1, 1)

        edge_features *= (adj.unsqueeze(-1) != 0).float()

        edge_features_unrolled = torch.reshape(edge_features, (edge_features.shape[0], edge_features.shape[1] * edge_features.shape[2], edge_features.shape[-1]))
        embedded_edges_unrolled = F.relu(self.edge_embedding_NN(edge_features_unrolled))
        embedded_edges_rolled = torch.reshape(embedded_edges_unrolled,
                                              (adj.shape[0], adj.shape[1], adj.shape[2], self.n_features - 1))
        embedded_edges = embedded_edges_rolled.sum(dim=2) / norm_fill_1

        edge_embeddings = F.relu(self.edge_feature_NN(torch.cat([embedded_edges, norm / norm_fill_1.max()], dim=-1)))

        return edge_embeddings


class UpdateNodeEmbeddingLayer(nn.Module):

    def __init__(self, n_features):
        super().__init__()

        self.message_layer = nn.Linear(2 * n_features, n_features, bias=False)
        self.update_layer = nn.Linear(2 * n_features, n_features, bias=False)

    def forward(self, current_node_embeddings, current_node_embeddings_opp, edge_embeddings, norm, adj):
        node_embeddings_aggregated = torch.matmul(adj, current_node_embeddings_opp) / norm

        message = F.relu(self.message_layer(torch.cat([node_embeddings_aggregated, edge_embeddings], dim=-1)))
        new_node_embeddings = F.relu(self.update_layer(torch.cat([current_node_embeddings, message], dim=-1)))

        return new_node_embeddings


class ReadoutLayer(nn.Module):

    def __init__(self, n_features, n_hid=[], bias_pool=False, bias_readout=True):

        super().__init__()

        self.layer_pooled_g = nn.Linear(int(n_features), int(n_features), bias=bias_pool)
        self.layer_pooled_p = nn.Linear(int(n_features), int(n_features), bias=bias_pool)

        if type(n_hid) != list:
            n_hid = [n_hid]

        n_hid = [2*n_features] + n_hid + [1]

        self.layers_readout = []
        for n_in, n_out in list(zip(n_hid, n_hid[1:])):
            layer = nn.Linear(n_in, n_out, bias=bias_readout)
            self.layers_readout.append(layer)

        self.layers_readout = nn.ModuleList(self.layers_readout)

    def forward(self, node_embeddings_g, node_embeddings_p):

        f_local = node_embeddings_g

        h_pooled_g = self.layer_pooled_g(node_embeddings_g.sum(dim=1) / node_embeddings_g.shape[1])
        h_pooled_p = self.layer_pooled_p(node_embeddings_p.sum(dim=1) / node_embeddings_p.shape[1])
        h_pooled = F.relu(h_pooled_g + h_pooled_p)
        f_pooled = h_pooled.repeat(1, 1, node_embeddings_g.shape[1]).view(node_embeddings_g.shape)

        features = F.relu(torch.cat([f_pooled, f_local], dim=-1))
        # features = F.relu(h_pooled_g + h_pooled_p)

        for i, layer in enumerate(self.layers_readout):
            features = layer(features)
            if i < len(self.layers_readout) - 1:
                features = F.relu(features)
            else:
                out = features

        return out