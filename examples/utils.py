import math
import torch
from torch.autograd import Variable


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if h is None:
        return h
    if type(h) in [Variable, torch.Tensor, torch.FloatTensor]:
        return Variable(h.data)
    else:
        return [repackage_hidden(v) for v in h]


def batchify(data, bsz, args):
    """Carve corpus up into batches"""
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda(args.device)
    return data


def get_batch(source, i, args, seq_len=None, evaluation=False):
    """Fetch batch from source"""
    seq_len = min(seq_len if seq_len else args.seq_len, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target


def ppl(x):
    """Perplexity score"""
    try:
        return math.exp(x)
    except OverflowError:
        return math.nan
