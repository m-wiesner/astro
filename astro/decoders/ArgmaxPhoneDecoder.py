import torch.nn as nn

###############################################################################
#                          Argmax Decoder
#
# This class defines the argmax decoder class
###############################################################################
class ArgmaxPhoneDecoder(nn.Module):
    @classmethod
    def from_args(cls, args): 
        phones = []
        with open(args.phones, 'r', encoding='utf-8') as f:
            for l in f:
                p, _ = l.strip().split(None, 1)
                phones.append(p)
        return cls(phones)

    def __init__(self, phones):
        self.phones = phones
    
    def __call__(self, nnet_outputs, lengths=None):
        # Assumes the nnet_outputs are (BxTxD) and we take the argmax over the
        # D different output classes.
        preds = nnet_outputs.argmax(-1)
        return [
            ' '.join(
                map(
                    lambda x: str(x), #self.phones[x], 
                    preds[i].unique_consecutive()[preds[i].unique_consecutive() != 0].tolist()
                )
            )
            for i in range(preds.size(0)) 
        ]

