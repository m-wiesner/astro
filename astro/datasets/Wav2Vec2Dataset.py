from .AstroDataset import Minibatch

import torch 

from lhotse.dataset import OnTheFlyFeatures
from lhotse.dataset.input_strategies import BatchIO, PrecomputedFeatures
from lhotse import Fbank, FbankConfig
from lhotse import CutSet
from lhotse.workarounds import Hdf5MemoryIssueFix
from lhotse.dataset.speech_recognition import validate_for_asr
from lhotse.utils import ifnone
from lhotse.dataset.input_strategies import AudioSamples


class Wav2Vec2Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.hdf5_fix = Hdf5MemoryIssueFix(reset_interval=100)
        self.input_method = AudioSamples()

    def __getitem__(self, cuts: CutSet) -> Minibatch:
        validate_for_asr(cuts)
        self.hdf5_fix.update()
        cuts = cuts.sort_by_duration(ascending=False)
        
        inputs, _ = self.input_method(cuts)
        supervision_intervals = self.input_method.supervision_intervals(cuts)
        tokens, token_lens = self.tokenizer(cuts)
        metadata = {
            'input_lens': supervision_intervals['num_samples'],
            'target_lens': token_lens,
            'utt_ids': [s.recording_id for cut in cuts for s in cut.supervisions],
            'text': [s.text for cut in cuts for s in cut.supervisions],
            'ids': [s.id for cut in cuts for s in cut.supervisions],
            'intervals': supervision_intervals,
        }
        targets = [t for t in tokens]
        return Minibatch(
            {
                'input': inputs,
                'targets': targets,
                'metadata': metadata,  
            }
        )
