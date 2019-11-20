# Factorized-TDNN

PyTorch implementation of the Factorized TDNN (TDNN-F) from ["Semi-Orthogonal Low-Rank Matrix Factorization for Deep Neural Networks"](http://danielpovey.com/files/2018_interspeech_tdnnf.pdf)[1]. This is also known as TDNN-F in nnet3 of [Kaldi](https://github.com/kaldi-asr/kaldi).

![model_fig](figures/ftdnn.png?raw=true "ftdnn diag") Taken from [1]

A TDNN-F layer is implemented in the class `FTDNNLayer` of `models.py`. To be specific to the description in [1], it is an implementation of the **"3-stage splicing"** implementation, in which three convolutions are used in sequence, with the first two being constrained to be semi-orthogonal. These convolutions are followed by a ReLU and then BatchNorm layer. The semi-orthogonal constraint is the **"floating case"** in [1]. (TODO: implement the scaled case like in Kaldi)

# Usage

## `FTDNNLayer`

This `FTDNNLayer` of `models.py` is used as follows:

```python
import torch
from models import FTDNNLayer, SOrthConv

tdnn_f = FTDNNLayer(1280, 512, 256, context_size=2, dilations=[2,2,2], paddings=[1,1,1])
# This is a sequence of three 2x1 convolutions
# dimensions go from 1280 -> 256 -> 256 -> 512
# dilations and paddings handles how much to dilate and pad each convolution
# Having these configurable is to ensure the sequence length stays the same

test_input = torch.rand(5, 100, 1280)
# inputs to the FTDNNLayer must be (batch_size, seq_len, in_dim)

tdnn_f(test_input).shape # returns (5, 100, 512)

tdnn_f.step_semi_orth() # The key method to constrain the first two convolutions, perform after every SGD step

tdnn_f.orth_error() # This returns the orth error of the constrained convs, useful for debugging
```

## `SOrthConv`

The components of `FTDNNLayer` which have the semi-orthogonal constraint are based around the class `SOrthConv`, which is essentially a `nn.Conv1d` with a `.step_semi_orth()` method to perform the semi-orthogonal update as in [1].

```python
sorth_conv = SOrthConv(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, padding_mode='zeros')
```

The implementation of the `.step_semi_orth()` method has been made to be as close to `ConstrainOrthonormalInternal` from [nnet-utils.cc](https://github.com/kaldi-asr/kaldi/blob/master/src/nnet3/nnet-utils.cc) in Kaldi's `nnet3` module.

# Extras

Also included in this repo in `models.py` is the following:
 * `FTDNN`: Factorized TDNN x-vector architecture (FTDNN) up to the embedding layer seen in  ["State-of-the-art speaker recognition with neural network embeddings in NIST SRE18 and Speakers in the Wild evaluations"](https://www.sciencedirect.com/science/article/pii/S0885230819302700)[2]. (This is not EXACTLY the same, but should be close enough).
 * `SharedDimScaleDropout`: The shared dimension scaled dropout described in [1] and in Kaldi:
     * Instead of randomly setting inputs to 0, use a continuous dropout scale.
     * For a dropout 'strength' alpha, multiply inputs inputs by a mask sampled from the uniform distribution on the interval [1 - 2 \* alpha, 1 + 2 \* alpha].
     * Share dropout masks along a dimension, such as time. From [1]: "If, for instance, a dimension is zeroed on a particular frame it will be zeroed on all frames of that sequence".

![model_fig](figures/ftdnn_arch.png?raw=true "ftdnn arch") The FTDNN x-vector architecture description taken from [2]. Up until layer 12 is implemented in `FTDNN` in `models.py`.


# Demo [WIP]

An demonstration of the `FTDNN` model being trained can be seen in the following output log (code not included, TODO: basic experiment demo):

```
exp/sp_ftdnn_bl: Wed Nov 20 14:21:15 2019: [10/120000]   C-Loss:21.9116, AvgLoss:21.6991, lr: 0.2, bs: 400
Orth error: 22.44341427081963
exp/sp_ftdnn_bl: Wed Nov 20 14:21:29 2019: [20/120000]   C-Loss:21.6260, AvgLoss:21.7459, lr: 0.2, bs: 400
Orth error: 8.235212338215206
exp/sp_ftdnn_bl: Wed Nov 20 14:21:43 2019: [30/120000]   C-Loss:21.7663, AvgLoss:21.7525, lr: 0.2, bs: 400
Orth error: 1.2611256236341433
exp/sp_ftdnn_bl: Wed Nov 20 14:21:56 2019: [40/120000]   C-Loss:21.6153, AvgLoss:21.6527, lr: 0.2, bs: 400
Orth error: 0.005309408872562926
exp/sp_ftdnn_bl: Wed Nov 20 14:22:14 2019: [50/120000]   C-Loss:21.0997, AvgLoss:21.5722, lr: 0.2, bs: 400
Orth error: 0.005543942232179688
exp/sp_ftdnn_bl: Wed Nov 20 14:22:26 2019: [60/120000]   C-Loss:21.2629, AvgLoss:21.5222, lr: 0.2, bs: 400
Orth error: 0.004769200691953301
exp/sp_ftdnn_bl: Wed Nov 20 14:22:40 2019: [70/120000]   C-Loss:20.9551, AvgLoss:21.4158, lr: 0.2, bs: 400
Orth error: 0.006055477493646322
exp/sp_ftdnn_bl: Wed Nov 20 14:22:56 2019: [80/120000]   C-Loss:20.4425, AvgLoss:21.3274, lr: 0.2, bs: 400
Orth error: 0.009634702852054033
exp/sp_ftdnn_bl: Wed Nov 20 14:23:09 2019: [90/120000]   C-Loss:21.0025, AvgLoss:21.2727, lr: 0.2, bs: 400
Orth error: 0.00611297079740325
exp/sp_ftdnn_bl: Wed Nov 20 14:23:25 2019: [100/120000]          C-Loss:20.6145, AvgLoss:21.1736, lr: 0.2, bs: 400
Orth error: 0.008151484609697945
exp/sp_ftdnn_bl: Wed Nov 20 14:23:38 2019: [110/120000]          C-Loss:20.1985, AvgLoss:21.0890, lr: 0.2, bs: 400
Orth error: 0.0072971017434610985
exp/sp_ftdnn_bl: Wed Nov 20 14:23:53 2019: [120/120000]          C-Loss:20.5698, AvgLoss:21.0300, lr: 0.2, bs: 400
Orth error: 0.00629939052669215
exp/sp_ftdnn_bl: Wed Nov 20 14:24:08 2019: [130/120000]          C-Loss:20.2024, AvgLoss:20.9425, lr: 0.2, bs: 400
Orth error: 0.008707787481398555
exp/sp_ftdnn_bl: Wed Nov 20 14:24:21 2019: [140/120000]          C-Loss:19.7034, AvgLoss:20.8641, lr: 0.2, bs: 400
Orth error: 0.010941843771433923
exp/sp_ftdnn_bl: Wed Nov 20 14:24:37 2019: [150/120000]          C-Loss:19.9718, AvgLoss:20.8035, lr: 0.2, bs: 400
Orth error: 0.00768740743296803
```

The FTDNN x-vector architecture seems to train successfully, and most importantly the Orth error is minimized.

# TODOs

* Implement 'scaled' case of semi-orthogonal constraint
* Refactor so that seq_len is final dim (or not?)
* Simple experiment/toy demo

# References

```
[1]
@inproceedings{Povey2018,
  author={Daniel Povey and Gaofeng Cheng and Yiming Wang and Ke Li and Hainan Xu and Mahsa Yarmohammadi and Sanjeev Khudanpur},
  title={Semi-Orthogonal Low-Rank Matrix Factorization for Deep Neural Networks},
  year=2018,
  booktitle={Proc. Interspeech 2018},
  pages={3743--3747},
  doi={10.21437/Interspeech.2018-1417},
  url={http://dx.doi.org/10.21437/Interspeech.2018-1417}
}
```

```
[2]
@article{VILLALBA2020101026,
    title = "State-of-the-art speaker recognition with neural network embeddings in NIST SRE18 and Speakers in the Wild evaluations",
    journal = "Computer Speech & Language",
    volume = "60",
    pages = "101026",
    year = "2020",
    issn = "0885-2308",
    doi = "https://doi.org/10.1016/j.csl.2019.101026",
    url = "http://www.sciencedirect.com/science/article/pii/S0885230819302700",
    author = "Jesús Villalba and Nanxin Chen and David Snyder and Daniel Garcia-Romero and Alan McCree and Gregory Sell and Jonas Borgstrom and Leibny Paola García-Perera and Fred Richardson and Réda Dehak and Pedro A. Torres-Carrasquillo and Najim Dehak"
}
```
