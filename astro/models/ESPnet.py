# Espnet imports
from espnet2.asr.encoder.conformer_encoder import ConformerEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F


class ESPnet(nn.Module):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--conformer-num-blocks', type=int, default=6)
        parser.add_argument('--conformer-hidden-dim', type=int, default=2048)
        parser.add_argument('--conformer-final-dim', type=int, default=256)  
        
    def __init__(self, idim, num_blocks=6, hidden_dim=2048, output_dim=256):
        super(ESPnet, self).__init__()
        self.encoder = ConformerEncoder(idim,
            num_blocks=num_blocks,
            output_size=output_dim,
            linear_units=hidden_dim,
        )
        self.odim = output_dim
        
    def forward(self, x, input_lens):
        x, olens, _ = self.encoder(x, input_lens)
        return x, olens.to(torch.int32) # x used to be here y, x, olens



