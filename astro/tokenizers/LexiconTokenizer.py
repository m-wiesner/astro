import torch
import random
from lhotse import CutSet
import re
from .BaseTokenizer import BaseTokenizer


class LexiconTokenizer(BaseTokenizer):
    '''
       Defines the collating and tokenization for ASR training using a lexicon
       model using lhotse cuts 
    '''
    def __init__(
        self,
        lexicon,
        phones,
    ):
        '''
            Inputs:
                :lexicon: The lexicon used for phone tokenization
                :phones: The list of phones used (<eps> at the beginning)
                :return: BPE Collater object
        '''
        self.lexicon = lexicon
        self.phones = phones 
        self.words = ['<eps>'] + sorted(lexicon.keys())
        self.vocab_size = len(self.phones) 

    def __call__(self, cuts: CutSet):
        # Encode the text
        token_sequences = [
            self.prepare_transcript(re.sub(r'(<noise>)([^ ]+)', r'\1 \2', sup.text))
            for cut in cuts for sup in cut.supervisions
        ]
        
        # Get the max length in the batch (needed for padding).
        max_len = len(max(token_sequences, key=len))
        
        seqs = [
            seq + [-1] * (max_len - len(seq))
            for seq in token_sequences
        ]

        # Create a tensor and return it along with the sequence lengths
        tokens_batch = torch.LongTensor(seqs) 
        tokens_lens = torch.IntTensor([len(seq) for seq in token_sequences])
        return [tokens_batch,], tokens_lens

    def prepare_transcript(self, text):
        '''
           Make phonemes 
        '''
        transcript = []
        for w in text.split():
            transcript.extend(
                [self.phones.index(p) for p in random.choice(list(self.lexicon.get(w, set([("NSN",)]))))]
            )
        return transcript
    
    def tokens_to_syms(self, tokens, token_type='word'):
        '''
            Inputs:
                :param tokens: list of lists of tokens to convert to output
                    symbols.
                :param token_type: the type (word, phone) of the output symbol.
        '''
        if token_type not in ('word', 'phone'):
            raise ValueError("Expected token_type to be 'word' or 'phone'")
       
        lookup_table = self.phones if token_type == "phone" else self.words 
        # For back compatibility
        if self.words[0] != "<eps>":
            self.words = ["<eps>"] + self.words
        #other_words = []
        #with open('data/lang/words.txt', 'r') as f:
        #    for l in f:
        #        w, i = l.strip().split(None, 1)
        #        other_words.append(w)
        output = []
        for seq in tokens:
            output.append(' '.join(map(lambda x : lookup_table[x], seq)))
        return output
    
    def words_to_phones(self, words):
        phones = []
        for w in words.split():
            for p in random.choice(list(self.lexicon.get(w, set([("NSN",)])))):
                phones.append(p)
        return ' '.join(phones)
