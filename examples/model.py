"""PyTorch Language Model

Generic PyTorch Language Model that can runs on top of an RNN class.
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop


def get_model(args):
    """Return the specified model"""
    return RNNModel(
        rnn_type=args.model,
        ntoken=args.ntokens,
        ninp=args.emsize,
        nhid=args.nhid,
        npar=args.npar,
        nlayers=args.nlayers,
        dropouth=args.dropouth,
        dropouti=args.dropouti,
        dropoute=args.dropoute,
        dropouto=args.dropouto,
        dropouta=args.dropouta,
        tie_weights=args.tied
    )


class RNNModel(nn.Module):

    """RNN Language Model

    Container module with an encoder, a recurrent module, and a decoder.
    """

    def __init__(self, rnn_type, ntoken, ninp, nhid, npar, nlayers, dropoutl=0,
                 dropouto=0.6, dropouth=0.5, dropouti=0.5, dropoute=0.1,
                 dropouta=0.1, wdrop=0, tie_weights=False):
        super(RNNModel, self).__init__()
        self.rnn_type = rnn_type
        self.ntoken = ntoken
        self.ninp = ninp
        self.nhid = nhid
        self.npar = npar
        self.nlayers = nlayers
        self.dropoutl = dropoutl
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.dropouto = dropouto
        self.dropouta = dropouta
        self.wdrop = wdrop
        self.tie_weights = tie_weights

        self.lockdrop = LockedDropout()
        self.edrop = nn.Dropout(dropoute)
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.ldrop = nn.Dropout(dropoutl)
        self.odrop = nn.Dropout(dropouto)
        self.encoder = nn.Embedding(ntoken, ninp)

        if rnn_type == 'ALSTM':
            from alstm import aLSTM
            self.rnns = aLSTM(ninp, nhid, npar, ninp, nlayers,
                              dropout_alstm=dropouth, dropout_adapt=dropouta)

        elif rnn_type == 'LSTM':
            self.rnns = [nn.LSTM(ninp if l == 0 else nhid,
                                 nhid if l != nlayers - 1 else ninp, 1,
                                 dropout=0)
                         for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop)
                             for rnn in self.rnns]

        elif rnn_type == 'GRU':
            self.rnns = [nn.GRU(ninp if l == 0 else nhid,
                                nhid if l != nlayers - 1 else ninp, 1,
                                dropout=0)
                         for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop)
                             for rnn in self.rnns]

        elif rnn_type == 'QRNN':
            from torchqrnn import QRNNLayer
            self.rnns = [QRNNLayer(input_size=ninp if l == 0 else nhid,
                                   hidden_size=nhid if l != nlayers - 1 else ninp,
                                   save_prev_x=False, zoneout=0,
                                   window=1 if l == 0 else 1, output_gate=True)
                         for l in range(nlayers)]
        else:
            raise NotImplementedError("Model type not implemented")

        if rnn_type != 'ALSTM':
            self.rnns = torch.nn.ModuleList(self.rnns)

        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            self.decoder.weight = self.encoder.weight

        self.init_weights()

    def reset(self):
        if self.rnn_type == 'QRNN':
            [r.reset() for r in self.rnns]

    def init_weights(self):
        """Initialize Embedding weights"""
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, return_h=False):
        """Run Language model on given input"""
        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        emb = self.lockdrop(emb, self.dropouti)

        output, hidden, raw_outputs, outputs = self._forward(emb, hidden)

        output = self.lockdrop(output, self.dropouto)
        outputs.append(output)

        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        result = decoded.view(output.size(0), output.size(1), decoded.size(1))

        if return_h:
            return result, hidden, raw_outputs, outputs
        return result, hidden

    def _forward(self, emb, hidden):
        if self.rnn_type == 'ALSTM':
            output, hidden, output_all, output_all_raw = self.rnns(emb, hidden, return_all=True)
            return output, hidden, output_all_raw, output_all

        # Original AWD-LSTM code
        raw_output = emb
        new_hidden = []
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden
        return output, hidden, raw_outputs, outputs

    def init_hidden(self, bsz):
        if self.rnn_type == 'ALSM':
            return None

        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return [(Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else self.ninp).zero_()),
                    Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else self.ninp).zero_()))
                    for l in range(self.nlayers)]
        elif self.rnn_type == 'QRNN' or self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
            return [Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else self.ninp).zero_())
                    for l in range(self.nlayers)]
