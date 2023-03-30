from lhotse import Fbank, FbankConfig, LilcomChunkyWriter
from lhotse import CutSet, RecordingSet, SupervisionSet

partitions = [
    'train-clean-100',
    'train-clean-360',
    'train-other-500',
    'dev-clean',
    'dev-other',
    'test-clean',
    'test-other',
]

def prepare_fbank_cuts(manifests, part, num_mel_bins, odir,
    speed_perturb=False, num_jobs=80,
):
    if part not in partitions:
        raise ValueError(f"Expected {part} to be an element in {partitions}")
    extractor = Fbank(FbankConfig(num_mel_bins=num_mel_bins))
    recos = RecordingSet.from_jsonl(
        f'{manifests}/librispeech_recordings_{part}.jsonl.gz'
    )
    sups = SupervisionSet.from_jsonl(
        f'{manifests}/librispeech_supervisions_{part}.jsonl.gz'
    )
    
    cut_set = CutSet.from_manifests(
        recordings=recos,
        supervisions=sups,
    )
    if speed_perturb:
        cut_set = (
            cut_set
            + cut_set.perturb_speed(0.9)
            + cut_set.perturb_speed(1.1)
        )

    cut_set = cut_set.compute_and_store_features(
        extractor=extractor,
        storage_path=f"{odir}/librispeech_feats_{part}",
        num_jobs=num_jobs,
        storage_type=LilcomChunkyWriter,
    )
    
    cut_set.to_file(f"{odir}/librispeech_cuts_{part}.jsonl.gz")


