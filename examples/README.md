# Adaptive LSTM for Language Modeling 

This repo replicates the experiments in the original paper ([https://arxiv.org/abs/1805.08574](https://arxiv.org/abs/1805.08574)). The training code base is derive from
[https://github.com/salesforce/awd-lstm-lm](https://github.com/salesforce/awd-lstm-lm).

<div align="center">
<img src="valcurve.jpg" width="100%"><br><br>
</div>
<hr>

## Setup

Install [PyTorch](https://pytorch.org/) and [alstm](https://github.com/flennerhag/alstm). Create ``log`` and ``checkpoints`` directory in the root of the ``examples`` directory:

```bash
cd examples; mkdir log; mkdir checkpoints
```

Download the data you want to use (``p`` for Penn Treebank and ``w`` for Wikitext-2):

```bash
getdata.sh -pw 
```

## Train

### Penn Treebank

To train the aLSTM on Penn Treebank, run

```bash
python main.py --model ALSTM --epochs 190 --emsize 400 --nhid 1150 --nlayers 2 --npar 100 --dropouth 0.25 --dropoute 0.16 --dropouti 0.6 --dropouto 0.6 --dropouta 0.1 --wdecay 1e-6 --device 1 --var-seq --seq-len 70 --batch_size 20 --cut-steps 100 160 --cut-rate 10 --save
```

This will give you val / test scores of ``58.7`` / ``56.6``.

### Wikitext-2

To train the aLSTM on Wikitext-2, run

```bash
python main.py --model ALSTM --epochs 190 --emsize 400 --nhid 1500 --nlayers 2 --npar 100 --dropouth 0.3 --dropoute 0.16 --dropouti 0.6 --dropouto 0.6 --dropouta 0.1 --wdecay 1e-6 --device 1 --var-seq --seq-len 70 --batch_size 20 --cut-steps 100 160 180 --cut-rate 10 --save --data data/wikitext-2
```

This will give you val / test scores of ``67.7`` / ``64.8``.

The API for these language models is the same as that of the [AWD-LSTM](https://github.com/salesforce/awd-lstm-lm), so you can use any post-processing scripts they have if you want to fine tune, add a [neural cache](https://arxiv.org/abs/1612.04426) or generate samples. 

