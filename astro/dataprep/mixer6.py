from lhotse import Fbank, FbankConfig, LilcomChunkyWriter
from lhotse import CutSet, RecordingSet, SupervisionSet
from contextlib import contextmanager
import torch


torch.set_num_threads(1)
torch.set_num_interop_threads(1)


partitions = [
    'train_intv',
    'train_call',
    'dev_a',
    'test',
]


@contextmanager
def get_executor():
    import plz
    from distributed import Client
    with plz.setup_cluster(grid='coe') as cluster:
        cluster.scale(80)
        yield Client(cluster)
    return


def prepare_fbank_cuts(manifests, part, num_mel_bins, odir,
    speed_perturb=False, num_jobs=80,
):
    if part not in partitions:
        raise ValueError(f"Expected {part} to be an element in {partitions}")
    extractor = Fbank(FbankConfig(num_mel_bins=num_mel_bins))
    recos = RecordingSet.from_jsonl(
        f'{manifests}/recordings_{part}.jsonl.gz'
    )
    sups = SupervisionSet.from_jsonl(
        f'{manifests}/supervisions_{part}.jsonl.gz'
    )
    
    cut_set = CutSet.from_manifests(
        recordings=recos,
        supervisions=sups,
    )
    cut_set = cut_set.trim_to_supervisions(keep_overlapping=False) 
    if speed_perturb:
        cut_set = (
            cut_set
            + cut_set.perturb_speed(0.9)
            + cut_set.perturb_speed(1.1)
        )

    with get_executor() as ex:
        cut_set = cut_set.compute_and_store_features(
            extractor=extractor,
            storage_path=f"{odir}/mixer6_feats_{part}",
            num_jobs=num_jobs,
            storage_type=LilcomChunkyWriter,
            executor=ex,
        )
    
        cut_set.to_file(f"{odir}/mixer6_cuts_{part}.jsonl.gz")


