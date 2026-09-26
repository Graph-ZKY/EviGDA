"""Shared GCN encoder and classification/evidence heads.

Parameter names deliberately match the supplied checkpoints. The legacy models
allocate bottlenecks but bypass them in forward(); keeping these inactive modules
allows strict loading without silently changing the source model's predictions.
"""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GraphBackbone(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, dropout, use_bn=False, ego_mix=0.0):
        super().__init__()
        self.dropout = dropout
        self.use_bn = use_bn
        if not 0.0 <= ego_mix <= 1.0:
            raise ValueError('ego_mix must lie in [0, 1].')
        self.ego_mix = ego_mix
        self.layers = nn.ModuleList([
            GCNConv(input_dim if index == 0 else hidden_dim, hidden_dim,
                    cached=True, normalize=True)
            for index in range(layers)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(layers)
        ]) if use_bn else None

    def forward(self, x, adjacency):
        for index, layer in enumerate(self.layers):
            aggregated = layer(x, adjacency)
            if self.ego_mix:
                # Explicit heterophily variant: preserve each node's own
                # transformed features using the same learned GCN weights.
                own_features = layer.lin(x)
                if layer.bias is not None:
                    own_features = own_features + layer.bias
                x = (1.0 - self.ego_mix) * aggregated + self.ego_mix * own_features
            else:
                x = aggregated
            if self.use_bn:
                x = self.bns[index](x)
            if index < len(self.layers) - 1:
                x = F.dropout(F.relu(x), p=self.dropout, training=self.training)
        return x


class Bottleneck(nn.Module):
    """Inactive checkpoint compatibility module; excluded from optimization."""
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)


class Head(nn.Module):
    def __init__(self, hidden_dim, classes):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, classes)

    def forward(self, x):
        return self.fc(x)


class EvidentialGCN(nn.Module):
    """Return (class logits, evidence logits, instance embeddings).

``batch`` is absent for node classification and maps nodes to graphs for
graph classification. A model instance expects a fixed target adjacency;
clear_cache() must be called before passing a different graph.
"""
    def __init__(self, input_dim, hidden_dim, layers, classes, dropout=0.0,
                 bottleneck_dim=None, use_bn=False, ego_mix=0.0):
        super().__init__()
        bottleneck_dim = bottleneck_dim or hidden_dim
        self.backbone = GraphBackbone(input_dim, hidden_dim, layers, dropout, use_bn, ego_mix)
        self.bn_cls = Bottleneck(hidden_dim, bottleneck_dim)
        self.bn_evi = Bottleneck(hidden_dim, bottleneck_dim)
        self.clf = Head(bottleneck_dim, classes)
        self.evi = Head(bottleneck_dim, classes)
        for module in (self.bn_cls, self.bn_evi):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def forward(self, x, adjacency, batch=None):
        features = self.backbone(x, adjacency)
        if batch is not None:
            features = global_mean_pool(features, batch)
        return self.clf(features), self.evi(features), features

    def clear_cache(self):
        for layer in self.backbone.layers:
            layer._cached_edge_index = None
            layer._cached_adj_t = None

    def active_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


def load_model(path, dropout, device, ego_mix=0.0):
    """Infer dimensions from weights, avoiding hard-coded layer mismatches."""
    state = torch.load(str(Path(path)), map_location='cpu')
    layer_ids = {int(key.split('.')[2]) for key in state
                 if key.startswith('backbone.layers.')}
    first = state['backbone.layers.0.lin.weight']
    architecture = dict(
        input_dim=first.shape[1], hidden_dim=first.shape[0],
        layers=len(layer_ids), classes=state['clf.fc.weight'].shape[0],
        bottleneck_dim=state['bn_cls.fc.weight'].shape[0],
        use_bn=any(key.startswith('backbone.bns.') for key in state),
        ego_mix=ego_mix,
    )
    model = EvidentialGCN(dropout=dropout, **architecture)
    model.load_state_dict(state, strict=True)
    return model.to(device), architecture
