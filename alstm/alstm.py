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


def alstm_cell(input, hidden, adapt, weight_ih, weight_hh, bias=None):
    """The adaptive LSTM Cell for one time step."""
    hx, cx = hidden

    hidden_size, input_size = hx.size(1), input.size(1)
    chunks = [input_size, hidden_size, 4 * hidden_size, 4 * hidden_size]
    if bias is not None:
        chunks.append(4 * hidden_size)

    adapt = chunk(adapt, chunks, 1)

    input = input * adapt.pop(0)
    hx = hx * adapt.pop(0)
    igates = F.linear(input, weight_ih) * adapt.pop(0)
    hgates = F.linear(hx, weight_hh) * adapt.pop(0)

    if bias is not None:
        igates = igates + bias * adapt.pop(0)

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
        self.weight_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        if use_bias:
            self.bias = Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialization of parameters"""
        nn.init.orthogonal(self.weight_ih)
        nn.init.orthogonal(self.weight_hh)
        if self.use_bias:
            self.bias.data.zero_()
            # Forget gate bias initialization
            self.bias.data[self.hidden_size:2*self.hidden_size] += 1

    def forward(self, input, hx, adapt):
        """Run aLSTM for one time step with given input and policy"""
        return alstm_cell(input, hx, adapt, self.weight_ih, self.weight_hh, self.bias)


class aLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, adapt_size, output_size=None,
                 nlayers=1, dropout_alstm=None, dropout_adapt=None,
                 batch_first=False, bias=True):
        super(aLSTM, self).__init__()
        output_size = output_size if output_size is not None else hidden_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.adapt_size = adapt_size
        self.output_size = output_size
        self.nlayers = nlayers
        self.dropout_alstm = dropout_alstm
        self.dropout_adapt = dropout_adapt
        self.batch_first = batch_first
        self.bias = bias

        psz, alyr, elyr, flyr = [], [], [], []
        for l in range(nlayers):
            if l == 0:
                ninp, nhid = input_size, hidden_size
            elif l == nlayers - 1:
                ninp, nhid = hidden_size, output_size
            else:
                ninp, nhid = hidden_size, hidden_size
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

        adapt_hidden, alstm_hidden = hidden

        dropout_alstm = self._dropout(input.data, alstm_hidden, self.dropout_alstm)
        dropout_adapt = self._dropout(input.data, adapt_hidden, self.dropout_adapt)

        output = []
        for x in input:
            for l in range(self.nlayers):
                alyr = self.adapt_layers[l]
                elyr = self.project_layers[l]
                flyr = self.alstm_layers[l]
                ahx, ahc = adapt_hidden[l]
                fhx, fhc = alstm_hidden[l]

                if self.nlayers != 1:
                    ax = torch.cat([x, fhx, adapt_hidden[l-1][0]], 1)
                else:
                    ax = torch.cat([x, fhx], 1)

                ahx, ahc = alyr(ax, (ahx, ahc))
                ahx = dropout_adapt(ahx, l)
                ahe = elyr(ahx)

                fhx, fhc = flyr(x, (fhx, fhc), ahe)
                if l == self.nlayers - 1:
                    output.append(fhx)

                fhx = dropout_alstm(fhx, l)

                adapt_hidden[l] = [ahx, ahc]
                alstm_hidden[l] = [fhx, fhc]

                x = fhx
            ###
        ###
        hidden = (adapt_hidden, alstm_hidden)
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

    def _dropout(self, data_source, hiddens, dropout_rates):
        if self.training and dropout_rates:
            msz = [h[0].size() for h in hiddens]
            return VariationalDropout(data_source, dropout_rates, msz)
        return lambda x, l: x
