# Adaptive LSTM (aLSTM)

[PyTorch](https://pytorch.org/) implementation of the adaptive LSTM ([https://arxiv.org/abs/1805.08574](https://arxiv.org/abs/1805.08574)). 

aLSTM is an extension of the standard LSTM that implements adaptive parameterization. 
Adaptive parameterization increases model flexibility given a parameter budget, allowing
more flexible and statistically efficient models. The aLSTM converges faster
than the LSTM and reaches better generalizing performance. It is also very stable; no need to
use gradient clipping, even for sequences of up to thousands of terms. 
 
If you use this code or our results in your research, please cite

```
@article{Flennerhag:2018alstm,
  title   = {{Breaking the Activation Function Bottleneck through Adaptive Parameterization}},
  author  = {Flennerhag, Sebastian and Hujun, Yin and Keane, John and Elliot, Mark},
  journal = {{arXiv preprint, arXiv:1805.08574}},
  year    = {2018}
}
```

## Requirements

This implementation should run on any [PyTorch](https://pytorch.org/) version. It has been tested for v2–v4. To install:

```bash
git clone https://github.com/flennerhag/alstm; cd alstm
python setup.py install
```

## Usage

This implementation follows the LSTM implementation in the official (and constantly changing) 
[PyTorch repo](https://github.com/pytorch/pytorch). You have an ``alstm_cell`` function and its ``aLSTMCell``
module wrapper. These apply to a given time step. The ``aLSTM`` class provides an end-user API with 
variational dropout and our hybrid RHN-LSTM adaptation model for multi-layer aLSTMs.  

```python
import torch
from torch.autograd import Variable
from alstm import aLSTM

seq_len, batch_size, input_size, hidden_size, adapt_size, output_size, = 20, 5, 8, 10, 3, 7

alstm = aLSTM(input_size, hidden_size, adapt_size, output_size, nlayers=2)

X = Variable(torch.rand(seq_len, batch_size, hidden_size))
out, hidden = alstm(X) 
``` 

## Examples

To replicate the original experiments of the [aLSTM paper](https://arxiv.org/abs/1805.08574) see 
[examples](https://github.com/flennerhag/alstm/tree/master/examples).

## Contributions

If you spot a bug, think the docs are useless or have an idea for an extension, don't hesitate to send a PR! 
If your contribution is substantial, please raise an issue first to check that it is in line with the 
scope of this repo.  Quick wins that would be great to have are:

- Support for bidirectional aLSTM
- Support PyTorch's PackedSequence
