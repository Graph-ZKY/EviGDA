"""Index-stable EMA memory and chunked cosine nearest-neighbor retrieval."""
import torch
from torch.nn import functional as F


class PredictionMemory:
    def __init__(self, features, probabilities, momentum=0.9, warm_start=True,
                 exclude_self=True, chunk_size=1024):
        self.momentum = momentum
        self.exclude_self = exclude_self
        self.chunk_size = chunk_size
        self.features = torch.rand_like(features)
        self.probabilities = torch.full_like(probabilities, 1.0 / probabilities.shape[1])
        if warm_start:
            self.features.copy_(features)
            self.probabilities.copy_(self.sharpen(probabilities))

    @staticmethod
    def sharpen(probabilities):
        squared = probabilities.square()
        # Original class-balanced squared-probability normalization (column-wise).
        return squared / squared.sum(0, keepdim=True).clamp_min(1e-8)

    @torch.no_grad()
    def update(self, features, probabilities):
        weight = self.momentum
        self.features.mul_(1 - weight).add_(features, alpha=weight)
        self.probabilities.mul_(1 - weight).add_(self.sharpen(probabilities), alpha=weight)

    @torch.no_grad()
    def pseudo_labels(self, features, neighbors):
        query = F.normalize(features, dim=1)
        reference = F.normalize(self.features, dim=1)
        size = query.shape[0]
        if size < 2:
            return self.probabilities.argmax(1)
        neighbors = min(neighbors, size - 1)
        outputs = []
        for start in range(0, size, self.chunk_size):
            end = min(start + self.chunk_size, size)
            similarities = query[start:end] @ reference.t()
            if self.exclude_self:
                rows = torch.arange(end - start, device=query.device)
                similarities[rows, rows + start] = -float('inf')
                indices = similarities.topk(neighbors, dim=1).indices
            else:
                indices = similarities.topk(neighbors + 1, dim=1).indices[:, 1:]
            outputs.append(self.probabilities[indices].mean(1).argmax(1))
        return torch.cat(outputs)
