import torch.nn as nn
import torch.nn.functional as F
import torch


###############################################################################
#                        The Loss function (CTC)
#
# We just use the pytorch, built-in CTC objective.
###############################################################################
class CTC(nn.Module):
    '''
        Implements the CTCLoss function.
    '''
    def __init__(self, mdl, odim, pad_id=-1, blank_idx=0):
        '''
            The CTC loss function.
            Inputs:
                :param idim: the input dimension of the linear classifier
                :param odim: the output dimension (# classes) of the classifier
                :param blank_idx: the index in sp of the blank symbol
                :return: CTCLoss instance
        '''
        super(CTC, self).__init__()
        self.pad_id = pad_id
        self.blank_idx = blank_idx
        # model is assume to be in the astro format which includes the output
        # dimensions
        self.mdl = mdl
        self.classifier = nn.Linear(mdl.odim, odim)  
        self.decoding = False
    
    def forward(self, b):
        x, input_lens, targets = b.input, b.metadata['input_lens'], b.targets 
        outputs, olens = self.mdl(x, input_lens) 
        targets = targets[0]
        # Get the output logits from the underlying model 
        if outputs.size(0) != olens.size(0):
            return None
    
        # CTC assumes we have normalized scores
        outputs = self.classifier(outputs)
        lprobs = F.log_softmax(outputs, dim=-1).transpose(0, 1).contiguous()
        
        if self.decoding:
            return lprobs.transpose(0, 1), olens 
        # The input lengths are assumed to also be an output of the model
        input_lengths = olens

        # Get the output indices over which the CTC loss will be computed. 
        # Indices where the output token are the pad symbol are ignored
        pad_mask = (targets != self.pad_id)
        targets_flat = targets.masked_select(pad_mask)

        # The target lengths are the number of indices in the target tensors
        # that were not the pad symbol.
        target_lengths = pad_mask.sum(-1)

        # Compute the CTC loss
        with torch.backends.cudnn.flags(enabled=False):
            loss = F.ctc_loss(
               lprobs,
               targets_flat,
               input_lengths,
               target_lengths,
               blank=self.blank_idx,
               reduction='sum',
               zero_infinity=True,
            )
        return loss, outputs, input_lengths

    def decode(self):
        self.decoding = True

    def freeze_mdl(self):
        for p in self.mdl.parameters():
            p.requires_grad = False

    def unfreeze_mdl(self):
        for p in self.mdl.parameters():
            p.requires_grad = True
