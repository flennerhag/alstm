"""PyTorch Training Script for language modeling
"""
import os
import argparse
import logging
import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import data
import model as model

from utils import batchify, get_batch, repackage_hidden, ppl

###############################################################################
parser = argparse.ArgumentParser(
    description='PyTorch Language Model Training Script'
)

###############################################################################
parser.add_argument(
    '--model', type=str, default='ALSTM', help='type of recurrent net'
)
parser.add_argument(
    '--emsize', type=int, default=400, help='size of word embeddings'
)
parser.add_argument(
    '--nhid', type=int, default=1000, help='number of hidden units per layer'
)
parser.add_argument(
    '--npar', type=int, default=100, help='number of adaptation units per layer'
)
parser.add_argument(
    '--nlayers', type=int, default=2, help='number of layers'
)
parser.add_argument(
    '--lr', type=float, default=0.003, help='initial learning rate'
)
parser.add_argument(
    '--cut-rate', type=float, default=4, help='learning rate cut factor'
)
parser.add_argument(
    '--cut-steps', type=int, nargs='+', default=[100, 180], help='learning rate cut epochs'
)
parser.add_argument(
    '--clip', type=float, default=0, help='gradient clipping'
)
parser.add_argument(
    '--epochs', type=int, default=200, help='upper epoch limit'
)
parser.add_argument(
    '--batch_size', type=int, default=64, help='batch size'
)
parser.add_argument(
    '--seq-len', type=int, default=35, help='sequence length'
)
parser.add_argument(
    '--var-seq', action='store_true', help='use variable sequence length'
)
parser.add_argument(
    '--dropouth', type=float, default=0.5, help='dropout for hidden state between layers (0 = no dropout)'
)
parser.add_argument(
    '--dropouti', type=float, default=0.5, help='dropout for input embedding layer (0 = no dropout)'
)
parser.add_argument(
    '--dropouto', type=float, default=0.4, help='dropout for final output embedding layer (0 = no dropout)'
)
parser.add_argument(
    '--dropoute', type=float, default=0.1, help='dropout to remove words from embedding layer (0 = no dropout)'
)
parser.add_argument(
    '--dropouta', type=float, default=0.0, help='dropout to adaptive hidden layer (0 = no dropout)'
)
parser.add_argument(
    '--tied', action='store_false', help='untie the word embedding and softmax weights'
)
parser.add_argument(
    '--seed', type=int, default=2987, help='random seed'
)
parser.add_argument(
    '--wdecay', type=float, default=1e-6, help='weight decay applied to all weights'
)
parser.add_argument(
    '--cuda', action='store_false', help='do not use CUDA'
)
parser.add_argument(
    '--log-interval', type=int, default=200, help='report interval'
)
parser.add_argument(
    '--device', default=0, type=int, help='GPU device'
)
parser.add_argument(
    '--parallel', action='store_true', help='Run model with DataParallel'
)
parser.add_argument(
    '--save', default=False, action='store_true', help='save log and best model'
)
parser.add_argument(
    '--resume', type=str, nargs='+', help='resume training of saved model. Specify path, start epoch count, best val loss'
)
parser.add_argument(
    '--overwrite', action='store_true', help="overwrite log/checkpoint in case of conflict"
)
parser.add_argument(
    '--data', type=str, default='./data/penn', help='path to data root directory'
)


args = parser.parse_args()

if args.resume:
    args.resume[1] = int(args.resume[1])
    args.resume[2] = float(args.resume[2])

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run "
              "with --cuda")
    else:
        torch.cuda.set_device(args.device)
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Logging
###############################################################################

for p in ['./logs', './checkpoints']:
    if not os.path.exists(p):
        os.mkdir(p)

# Format by date and params
date = time.gmtime()
_date = '%s-%s-%s-%s' % (
    str(date.tm_year), str(date.tm_mon),
    str(date.tm_mday), str(date.tm_hour))

model_specs = (
    'm:%s em:%i nh:%i na:%i nl:%i lr:%.3f de:%.2f di:%.2f dh:%.2f da:%.2f do:%.2f') % (
        args.model, args.emsize, args.nhid, args.npar, args.nlayers, args.lr,
        args.dropoute, args.dropouti, args.dropouth, args.dropouta, args.dropouto)
if args.resume:
    model_specs += ' resumed'

long_logname = '%s %s' % (_date, model_specs)
short_logname = model_specs
if args.save:
    log_path = os.path.join(os.getcwd(), 'logs', long_logname + '.log')
    ckpt_path = os.path.join(os.getcwd(), 'checkpoints', long_logname + '.model')
else:
    ckpt_path = os.path.join(os.getcwd(), 'checkpoints', 'tmp.model')
    log_path = os.path.join(os.getcwd(), 'logs', 'tmp.log')

for p in [log_path, ckpt_path]:
    if os.path.exists(p):
        if not args.save or args.overwrite:
            os.unlink(p)
        else:
            raise OSError("Log and/or checkpoint exists. Run with --overwrite")

logger = logging.getLogger(short_logname)
logger.setLevel(logging.INFO)

file_formatter = logging.Formatter('%(asctime)s - %(message)s', "%Y-%m-%d-%H:%M:%S")
stream_formatter = logging.Formatter('%(message)s')

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(stream_formatter)
logger.addHandler(ch)

fh = logging.FileHandler(log_path)
fh.setLevel(logging.INFO)
fh.setFormatter(file_formatter)
logger.addHandler(fh)


###############################################################################
# Load data
###############################################################################
corpus = data.Corpus(args.data)

eval_batch_size = 10
test_batch_size = 1
train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)

###############################################################################
# Build the model
###############################################################################
args.ntokens = ntokens = len(corpus.dictionary)

if args.resume:

    def load_params():
        print('Loading saved model (%s)...' % args.resume[0], end='')
        global model, optimizer

        model, optimizer = torch.load(args.resume[0])

        if model.rnn_type == 'LSTM':
            model.rnns.flatten_parameters()

        for d in ['e', 'i', 'h', 'a', 'o']:
            drp = getattr(args, 'dropout' + d)
            setattr(model, 'dropout' + d, drp)

        if model.rnn_type == 'ALSTM':
            setattr(model.rnns, 'dropout_alstm', args.dropouth)
            setattr(model.rnns, 'dropout_adapt', args.dropouta)

        optimizer.param_groups[0]['lr'] = args.lr
        optimizer.param_groups[0]['weight_decay'] = args.wdecay
        print('done')

    load_params()

else:
    model = model.get_model(args)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=(0., 0.999),
        eps=1e-9, weight_decay=args.wdecay
    )

if args.cuda:
    model.cuda(args.device)

if args.parallel:
    model = torch.nn.DataParallel(model)
    model.init_hidden = model.module.init_hidden

criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def evaluate(model, data_source, batch_size=10):
    model.eval()
    if args.model == 'QRNN':
        model.reset()

    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.seq_len):
        data, targets = get_batch(data_source, i, args, evaluation=True)

        output = model(data, hidden)
        if isinstance(output, tuple):
            output, hidden = output

        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source)


def test():
    if os.path.exists(ckpt_path):
        with open(ckpt_path, 'rb') as f:
            tmp_mod, _ = torch.load(f)
    else:
        print("Checkpoint does not exist at %s" % ckpt_path)

    if args.cuda:
        tmp_mod.cuda(args.device)

    if tmp_mod.rnn_type == 'LSTM':
        for rnn in tmp_mod.rnns:
            rnn.flatten_parameters()

    test_loss = evaluate(tmp_mod, test_data, test_batch_size)
    logger.info(
        'TEST | loss {:5.2f} | ppl {:8.2f}'.format(
            test_loss, ppl(test_loss)))


def train():

    def getseq():
        lr_original = optimizer.param_groups[0]['lr']
        if args.var_seq:
            # Vary sequence length
            seq_len = args.seq_len if np.random.random() < 0.95 else args.seq_len / 2.
            seq_len =  max(5, int(np.random.normal(seq_len, 5)))
            optimizer.param_groups[0]['lr'] = lr_original * seq_len / args.seq_len
        else:
            seq_len = args.seq_len
        data, targets = get_batch(train_data, i, args, seq_len=seq_len)
        return data, targets, seq_len, lr_original

    if args.model == 'QRNN':
        model.reset()

    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    batch, i = 0, 0
    while i < train_data.size(0) - 1 - 1:
        model.train()
        data, targets, seq_len, lro = getseq()

        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()

        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        if args.clip:
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()
        total_loss += loss.data

        # Ensure learning rate is reset (only applicable with var_seq)
        optimizer.param_groups[0]['lr'] = lro

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            logger.info(
                'TRAIN | epoch {:3d} | {:5d}/{:5d} batches | lr {:01.8f} '
                '| ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch,
                    batch,
                    len(train_data) // args.seq_len,
                    optimizer.param_groups[0]['lr'],
                    elapsed * 1000 / args.log_interval,
                    cur_loss,
                    ppl(cur_loss)
                )
            )
            total_loss = 0
            start_time = time.time()
        ###
        batch += 1
        i += seq_len


if __name__ == '__main__':

    total_params = sum(
        x.size()[0] * x.size()[1] if len(x.size()) > 1 else
        x.size()[0] for x in model.parameters())
    args.params = total_params
    logger.info(args)

    lr = args.lr
    stored_loss = np.inf if not args.resume else args.resume[2]
    epochs = range(1, args.epochs + 1) if not args.resume else range(1 + args.resume[1], 1 + args.resume[1] + args.epochs)
    strikes = 0
    for epoch in epochs:
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(model, val_data, eval_batch_size)
        logger.info(
            'VAL | epoch {:3d} | time: {:5.2f}s | loss {:5.2f} '
            '| ppl {:8.2f}'.format(
                epoch, (time.time() - epoch_start_time), val_loss,
                ppl(val_loss)))

        if val_loss < stored_loss:
            with open(ckpt_path, 'wb') as f:
                torch.save([model, optimizer], f)
                stored_loss = val_loss

        if args.cut_steps and epoch % args.cut_steps[0] == 0:
            args.cut_steps.pop(0)
            optimizer.param_groups[0]['lr'] /= args.cut_rate

    test()
