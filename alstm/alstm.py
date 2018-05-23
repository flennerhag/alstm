"""adaptive LSTM

PyTorch implementation of the adaptive LSTM (https://arxiv.org/abs/1805.08574).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable
from torch.nn._functions.thnn import rnnFusedPointwise as fusedBackend

from .utils import Project, VariationalDropout, chunk, convert

# pylint: disable=too-many-locals,too-many-arguments,redefined-builtin


def alstm_cell(input, hidden, adapt, weights, bias=None):
    """The adaptive LSTM Cell for one time step."""
    hx, cx = hidden

    hidden_size, input_size = hidden.size(1), input.size(1)
    chunks = [input_size + hidden_size, 8 * hidden_size]
    if bias is not None:
        chunks.append(4 * hidden_size)

    adapt = chunk(adapt, chunks, 1)

    input = torch.cat([input, hx], 1) * adapt.pop(0)
    gates = F.linear(input, weights) * adapt.pop(0)

    igates, hgates = gates.chunk(2, 1)
    if bias is not None:
        hgates = hgates + bias * adapt.pop(0)

    if input.is_cuda:
        state = fusedBackend.LSTMFused.apply
        return state(igates, hgates, cx)

    gates = igates + hgates
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = F.sigmoid(ingate)
    forgetgate = F.sigmoid(forgetgate)
    cellgate = F.tanh(cellgate)
    outgate = F.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * F.tanh(cy)

    return hy, cy


class aLSTMCell(nn.modules.rnn.RNNCellBase):

    """Adaptive LSTM Cell
    """

    def __init__(self, input_size, hidden_size, use_bias=True):
        super(aLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.weights = Parameter(torch.Tensor(8 * hidden_size, hidden_size + input_size))
        if use_bias:
            self.bias = Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialization of parameters"""
        nn.init.orthogonal(self.weights)
        if self.use_bias:
            self.bias.data.zero_()
            # Forget gate bias initialization
            self.bias.data[self.hidden_size:2*self.hidden_size] += 1

    def forward(self, input, hx, adapt):
        """Run aLSTM for one time step with given input and policy"""
        return alstm_cell(input, hx, adapt, self.weights, self.bias)


class aLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, adapt_size, output_size=None,
                 nlayers=1, dropout_hidden=None, dropout_adapt=None,
                 batch_first=False, bias=True):
        super(aLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.adapt_size = adapt_size
        self.output_size = output_size if output_size else hidden_size
        self.nlayers = nlayers
        self.dropout_hidden = dropout_hidden
        self.dropout_adapt = dropout_adapt
        self.batch_first = batch_first
        self.bias = bias

        psz, alyr, elyr, flyr = [], [], [], []
        for l in range(nlayers):
            if l == 0:
                ninp, nhid = input_size, hidden_size

            if l == nlayers - 1:
                ninp, nhid = hidden_size, output_size
            if nlayers == 1:
                ninp, nhid = input_size, output_size

            # policy latent variable
            ain = adapt_size + ninp + nhid if nlayers != 1 else ninp + nhid
            alyr.append(nn.LSTMCell(ain, adapt_size))

            # sub-policy projection
            ipsz = ninp + nhid
            opsz = 8 * nhid if not bias else 12 * nhid
            psz.append(ipsz + opsz)
            elyr.append(Project(adapt_size, psz[-1]))

            # aLSTM
            flyr.append(aLSTMCell(ninp, nhid, use_bias=bias))

        self.adapt_layers = nn.ModuleList(alyr)
        self.project_layers = nn.ModuleList(elyr)
        self.alstm_layers = nn.ModuleList(flyr)
        self.policy_sizes = psz

    def forward(self, input, hidden=None):
        """run aLSTM over a batch of sequences."""
        if self.batch_first:
            input = input.transpose(0, 1)

        if hidden is None:
            hidden = self.init_hidden(input.size(1))

        hidden = convert(hidden, list)

        adaptive_hidden, alstm_hidden = hidden

        dropout = False
        if self.training and self.dropout:
            dropout = True
            lsz = [h[0].size() for h in alstm_hidden]
            asz = [h[0].size() for h in adaptive_hidden]
            dropout_alstm = VariationalDropout(
                input.data, self.dropout_hidden, lsz)
            dropout_adaptive = VariationalDropout(
                input.data, self.dropout_adaptive, asz)

        output = []
        for x in input:
            for l in range(self.nlayers):
                alyr = self.adapt_layers[l]
                elyr = self.project_layers[l]
                flyr = self.alstm_layers[l]
                ahx, ahc = adaptive_hidden[l]
                fhx, fhc = alstm_hidden[l]

                if self.nlayers != 1:
                    ax = torch.cat([x, fhx, adaptive_hidden[l-1][0]], 1)
                else:
                    ax = torch.cat([x, fhx], 1)

                ahx, ahc = alyr(ax, (ahx, ahc))

                if dropout:
                    ahx = dropout_adaptive(ahx, l)
                    ax = ahx
                else:
                    ax = ahx

                ahe = elyr(ax)
                fhx, fhc = flyr(x, (fhx, fhc), ahe)

                if l == self.nlayers - 1:
                    output.append(fhx)

                if dropout:
                    fhx = dropout_alstm(fhx, l)

                adaptive_hidden[l] = [ahx, ahc]
                alstm_hidden[l] = [fhx, fhc]

                x = fhx
            ###
        ###
        hidden = (adaptive_hidden, alstm_hidden)
        output = torch.stack(output, 1 if self.batch_first else 0)

        hidden = convert(hidden, tuple)
        return output, hidden

    def init_hidden(self, bsz):
        """Utility for initializing hidden states (to zero)"""
        asz = self.adapt_size
        osz = self.output_size
        hsz = self.hidden_size
        weight = next(self.parameters()).data

        def hidden(out):
            return Variable(weight.new(bsz, out).zero_())

        ah = [(hidden(asz), hidden(asz)) for _ in range(self.nlayers)]
        fh = [(hidden(hsz if l != self.nlayers - 1 else osz),
               hidden(hsz if l != self.nlayers - 1 else osz))
              for l in range(self.nlayers)]
        return ah, fh
