import torch
from lhotse.dataset import OnTheFlyFeatures
from lhotse.dataset.input_strategies import BatchIO, PrecomputedFeatures
from lhotse import Fbank, FbankConfig
from lhotse import CutSet
from lhotse.workarounds import Hdf5MemoryIssueFix
from lhotse.dataset.speech_recognition import validate_for_asr
from lhotse.utils import ifnone


class Minibatch(object):
    '''
        Defines the minibatch object

        Inputs:
            :param b: A dictionary with fields input, targets, and metadata
    '''
    def __init__(self, b):
        self.input = b['input']
        self.targets = b['targets']
        self.metadata = b['metadata']

    def to(self, device):
        targets = [t.to(device) for t in self.targets]
        self.input = self.input.to(device)
        self.targets = targets
        self.metadata = self.metadata
       
        
class AstroDataset(torch.utils.data.Dataset):
    '''
        A pytorch Dataset designed to support training ASR systems with lhotse. 
    '''
    def __init__(self, tokenizer,
        cut_transforms=None,
        input_transforms=None,
        num_mel=80,
        use_feats=False,
    ):
        '''
            Inputs:
                :param tokenizer: the sentencepiece model for bpe tokenization
                :param cut_transforms: the input transformations to apply
                :return: ASRDataset instance
        '''
        self.tokenizer = tokenizer
        self.cut_transforms = ifnone(cut_transforms, [])
        self.input_transforms = ifnone(input_transforms, [])
        self.hdf5_fix = Hdf5MemoryIssueFix(reset_interval=100)
        if use_feats:
            self.extractor = PrecomputedFeatures()
        else:
            self.extractor = OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=num_mel)))

    def __getitem__(self, cuts: CutSet) -> Minibatch:
        validate_for_asr(cuts)
        self.hdf5_fix.update()
        cuts = cuts.sort_by_duration(ascending=False)
        for tnfm in self.cut_transforms:
            cuts = tnfm(cuts)
        
        feats, feat_lens = self.extractor(cuts)
        for tnfm in self.input_transforms:
            feats = tnfm(feats)

        tokens, token_lens = self.tokenizer(cuts)
        sup_ints = self.extractor.supervision_intervals(cuts)
        metadata = {
            'input_lens': feat_lens,
            'target_lens': token_lens,
            'utt_ids': [s.recording_id for cut in cuts for s in cut.supervisions],
            'text': [s.text for cut in cuts for s in cut.supervisions],
            'ids': [s.id for cut in cuts for s in cut.supervisions], 
            'intervals': sup_ints, 
        }
        targets = [t for t in tokens]
        return Minibatch(
            {
                'input': feats,
                'targets': targets,
                'metadata': metadata,
            }
        )
