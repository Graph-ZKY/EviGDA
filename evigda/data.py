from dataclasses import dataclass
from pathlib import Path

import torch
from torch_geometric.data import Batch, InMemoryDataset
from torch_sparse import SparseTensor


@dataclass
class UnlabeledInput:
    x: torch.Tensor
    adjacency: SparseTensor
    batch: object
    size: int


class StoredGraphs(InMemoryDataset):
    def __init__(self, path):
        super().__init__(root=None)
        self.data, self.slices = torch.load(str(path), map_location='cpu')


def load_target(root, config, device):
    path = Path(root) / 'data' / config['family'] / config['target'] / 'processed' / 'data.pt'
    if not path.is_file():
        raise FileNotFoundError('Required supplied dataset is missing: {}'.format(path))
    if config['task'] == 'graph':
        dataset = StoredGraphs(path)
        graphs = [dataset.get(index) for index in range(len(dataset))]
        packed = Batch.from_data_list(graphs)
        batch = packed.batch.to(device)
        size = len(graphs)
    else:
        packed, _ = torch.load(str(path), map_location='cpu')
        batch = None
        size = packed.num_nodes
    labels = packed.y.reshape(-1).long().cpu().clone()
    if labels.numel() != size:
        raise ValueError('Label count does not match evaluation sample count.')
    x = packed.x.float().to(device)
    edges = packed.edge_index.long()
    adjacency = SparseTensor(row=edges[1], col=edges[0],
                             sparse_sizes=(x.shape[0], x.shape[0])).to(device)
    del packed
    return UnlabeledInput(x, adjacency, batch, size), labels, path
