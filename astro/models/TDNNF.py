from pytorch_tdnn.tdnnf import TDNNF as TDNNFLayer
import torch.nn as nn
import torch
import torch.nn.functional as F
import random
from torch.nn.utils.parametrizations import orthogonal as ortho
from math import ceil


class TDNNF(nn.Module):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--tdnn-num-tdnn-layers', type=int, default=9)
        parser.add_argument('--tdnn-stride', type=int, default=3)
        parser.add_argument('--tdnn-bottleneck-dim', type=int, default=96)
        parser.add_argument('--tdnn-hidden-dim', type=int, default=768)
        parser.add_argument('--tdnn-prefinal-dim', type=int, default=192)

    def __init__(self, idim,
        num_tdnn_layers=9,
        tdnn_stride=3,
        bottleneck_dim=96,
        hidden_dim=768,
        prefinal_dim=192,
    ):
        super(TDNNF, self).__init__()
        self.odim = hidden_dim 
        self.tdnn_stride = tdnn_stride 
        # The CNN layers are hard-coded for now
        self.cnn_layers = nn.ModuleList()
        # Layer 1 (Conv2d(in_channels, out_channels, kernel_size))
        # LayerNorm over features (height) and channel
        self.cnn_layers.append(nn.Conv2d(1, 48, 3, stride=1, padding='same'))
        self.cnn_layers.append(nn.Conv2d(48, 48, 3, stride=1, padding='same'))
        self.cnn_layers.append(nn.Conv2d(48, 64, 3, stride=(2, 1), padding=(2, 1)))
        self.cnn_layers.append(nn.Conv2d(64, 64, 3, stride=1, padding='same'))
        self.cnn_layers.append(nn.Conv2d(64, 64, 3, stride=(2, 1), padding=(2, 1)))
        self.cnn_layers.append(nn.Conv2d(64, 128, 3, stride=(2, 1), padding=(2, 1)))
       
        x = torch.randn(3, 200, idim)
        y = self._cnn_forward(x)
        first_tdnn_dim = y.size(1) * y.size(2)
        
        self.tdnn_layers = nn.ModuleList()
        for l in range(num_tdnn_layers):
            idim = first_tdnn_dim if l == 0 else hidden_dim
            tdnn_stride = tdnn_stride if l == 0 else 1
            self.tdnn_layers.append(
                TDNNFLayer(
                    idim,
                    hidden_dim,
                    bottleneck_dim,
                    tdnn_stride,  
                )
            )

        self.prefinal = nn.Linear(hidden_dim, prefinal_dim, bias=False)
        self.final = ortho(nn.Linear(prefinal_dim, hidden_dim, bias=False))
         
    def forward(self, x, input_lens):
        # x -- B x T x D  
        # input_lens -- B x 1 
        # Convert x to B x C x D x T for convolution
        x = self._cnn_forward(x)  
        # We need to B x (C * D) x T as input to TDNNF layers
        x = x.contiguous().view(x.size(0), x.size(1) * x.size(2), x.size(3))
        for layer_idx, layer in enumerate(self.tdnn_layers):
            # Do a semi-orthogonal step every 4 iterations on average
            semi_ortho_step = self.training and (random.uniform(0, 1) < 0.25)
            x = layer(x, semi_ortho_step=semi_ortho_step)
            # We subsample just the first time
            if layer_idx == 0:
                x = x[:, :, ::self.tdnn_stride]
        x = x.transpose(1, 2)
        x = self.prefinal(x)
        x = self.final(x)
        output_lens = [ int(ceil(i / self.tdnn_stride)) for i in input_lens ]  
         
        return x, torch.Tensor(output_lens).to(torch.long)

    def _cnn_forward(self, x):
        # Convert x to B x C x D x T for convolution
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        for layer in self.cnn_layers:
            # Conv2d
            x = F.relu(layer(x))
            # Do layernorm. Transpose for layer norm
            x = x.transpose(1, 3)
            x = F.layer_norm(x, [x.size(-2), x.size(-1)])
            # Transpose back to original
            x = x.transpose(1, 3)
        return x


