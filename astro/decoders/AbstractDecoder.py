import torch.nn as nn


###############################################################################
#                           Abstract Decoder Class
#
# This class defines the abstract decoder class
###############################################################################
class AbstractDecoder(nn.Module):
    @staticmethod
    def add_args(parser):
        pass

    def from_args(cls, args):
        raise NotImplementedError
    
    def __init__(self):
        super(AbstractDecoder, self).__init__()

    def __call__(self, nnet_outputs):
        '''
            :param nnet_output: the nnet scores for each class (B x T xD)
            :return: A list of lists. B output predictions each a list of words
        '''
        raise NotImplementedError

