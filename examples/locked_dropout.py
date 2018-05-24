import torch
import torch.nn as nn
from torch.autograd import Variable

class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5, seq_dim=0):
        if not self.training or not dropout:
            return x

        if seq_dim == 0:
            size = (1, x.size(1), x.size(2))
        elif seq_dim == 1:
            size = (x.size(0), 1, x.size(2))

        m = x.data.new(*size).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout)

        mask = mask.expand_as(x)
        return mask * x


class StickyDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask, dropout=0.5, seq_dim=0):
        if x is None or not self.training or not dropout:
            return x, mask

        if seq_dim == 0:
            size = (1, x.size(1), x.size(2))
        elif seq_dim == 1:
            size = (x.size(0), 1, x.size(2))

        if mask is None:
            m = x.data.new(*size).bernoulli_(1 - dropout)
            mask = Variable(m, requires_grad=False) / (1 - dropout)
            mask = mask.expand_as(x)
        return mask * x, mask
