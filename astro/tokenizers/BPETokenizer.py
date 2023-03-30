import torch
from lhotse import CutSet
from .BaseTokenizer import BaseTokenizer


class BPETokenizer(BaseTokenizer):
    '''
        Defines the collating and tokenization for ASR using
        sentencepiece units.
    '''
    def __init__(
        self,
        bpe,
    ):
        '''
            Inputs:
                :bpe: The sentence piece model (bpe tokenizer to use) 
                :return: BPETokenizer object
        '''
        self.bpe = bpe
        self.vocab_size = bpe.vocab_size()

    def __call__(self, cuts: CutSet):
        token_sequences = [
            self.prepare_transcript(sup.text)
            for cut in cuts for sup in cut.supervisions
        ]

        # Get the max length in the batch (needed for padding)
        seqs = [
            seq + [-1] * (max_len - len(seq))
            for seq in token_sequences
        ]

        # Create a tensor and return it along with sequence lengths
        tokens_batch = torch.LongTensor(seqs)
        tokens_lens = torch.IntTensor([len(seq) for seq in seq])
        return [tokens_batch,], tokens_lens
    
    def prepare_transcript(self, text):
        '''
            Make bpes
        '''
        return self.bpe.encode(text, out_type=int)

    def tokens_to_syms(self, tokens, token_type=None):
        return self.bpe.decode(tokens) 
