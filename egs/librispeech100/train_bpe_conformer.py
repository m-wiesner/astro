# ASTRO imports
from astro.models.ESPnet import ESPnet
from astro.iterators.train_phone import train_one_epoch
from astro.iterators.valid_phone import valid_phone
from astro.tokenizers.BPETokenizer import BPETokenizer
from astro.datasets.AstroDataset import AstroDataset
from astro.dataprep.librispeech import prepare_fbank_cuts
from astro.criteria.CTC import CTC

# Pytorch imports
import torch
#from torch.optim.swa_utils import AveragedModel, SWALR
import torch.multiprocessing as mp
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Lhotse imports
from lhotse.utils import fix_random_seed
from lhotse.recipes import prepare_librispeech 
from lhotse import load_manifest_lazy
from lhotse.dataset import SpecAugment, CutConcatenate, DynamicBucketingSampler

# Python imports
import argparse
import os
import sys
from pathlib import Path
import logging
from tqdm import tqdm
from itertools import chain
import editdistance


class _SeedWorkers:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        fix_random_seed(self.seed + worker_id)


def setup_dist(rank, world_size, master_port=None, use_ddp_launch=False):
    """
    rank and world_size are used only if use_ddp_launch is False.
    """
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"

    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = (
            "12354" if master_port is None else str(master_port)
        )

    if use_ddp_launch is False:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    else:
        dist.init_process_group("nccl")


def train(args, phones, loss_fn, optim, scaler, lr_sched,
    train_dloader, dev_dloader, device,
):
    '''
        The main training loop.
        Inputs:
            :param args: namespace with the commandline arguments
            :param phones: a list of the phones used in the lexiconf
            :param mdl: the neural network model
            :param loss_fn: the loss_fn, (astro/criteria/CTC.py for instance)
            :param optim: the optimizer used for training
            :param lr_sched: the learning rate scheduler for the optimizer
            :param train_dloader: the pytorch data loader
            :param dev_dloader: the pytorch dev data loader
            :param device: the device (cpu / cuda) on which to train
    '''
    # This list is just a queue, which keeps track of the paths of the last few
    # models from the previous args.keep_last epochs. To save disk space, we
    # remove all epochs prior to this.
    saved_epochs = []
    
    # The main training loop (from start_epoch to the total number of epochs)
    for e in range(args.start_epoch, args.epochs):
        logging.info(f"Epoch {e+1} of {args.epochs}")
        # We need to set the epoch in order to shuffle the lhotse sampler
        train_dloader.sampler.set_epoch(e)
        
        # Train one epoch. This is the main training loop
        # astro/iterators/train_phone.py
        train_one_epoch(
            phones, loss_fn, optim, lr_sched, scaler, train_dloader,
            device, wer_logging=args.wer_logging, fp16=args.fp16,
            grad_thresh=args.grad_thresh, print_interval=args.print_interval,
        )

        # In DDP if the device index is 0, then we run a validation step
        if device.index == 0:
            if dev_dloader is not None:
                # This is the main validation loop
                # (astro/iterators/valid_phone.py)
                loss_val, wer = valid_phone(
                    phones, loss_fn, dev_dloader, device,
                    wer_logging=args.wer_logging,
                    fp16=args.fp16,
                )
            else:
                loss_val, wer = None, None

            wer_str = f"WER: {wer:.02f}" if args.wer_logging else ''
            logging_string = f'''
            --------------------------------------------
            Epoch: {e+1} Loss_Val: {loss_val} {wer_str}
            --------------------------------------------
            '''
            logging.info(logging_string)

            # Save trained model
            state_dict = {
                'criterion_and_model': loss_fn.module.state_dict(),
                'optimizer': optim.state_dict(),
                'scaler': scaler.state_dict() if scaler is not None else None,
                'lr_sched': lr_sched.state_dict(),
                'epoch': e,
                'loss': loss_val,
                'wer': wer,
            }

            if e % args.save_interval == 0:
                mdl_path = f'{args.expdir}/{e+1}.mdl'
                torch.save(state_dict, mdl_path)
                saved_epochs.insert(0, (mdl_path, e))

            # Remove old checkpoint
            if len(saved_epochs) > args.keep_last:
                mdl_path, e = saved_epochs.pop()
                if e % args.keep_interval != 0:
                    os.remove(mdl_path)


def get_args():
    '''
        Parse the input arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None, help='The path to the librispeech data')
    parser.add_argument('--world-size', type=int, default=4, help='Number of GPUs')
    parser.add_argument('--lexicon', type=str, default=None, help='The path to the librispeech pronunciation lexicon')
    parser.add_argument('--phones', type=str, default=None, help='The path to the phone list used in the lexicon. If it does not exist it will be created and dumped.')
    parser.add_argument('--max-duration', type=float, default=20.0, help='remove utterances that are longer than this duration from training')
    parser.add_argument('--min-duration', type=float, default=1.0, help='remove utterances that are shorter than this duration from training')
    parser.add_argument('--minibatch-duration', type=float, default=550.0, help='the total duration of utterances in the minibatch')   
    parser.add_argument('--num-buckets', type=int, default=30, help='total number of buckets for BucketingSampler')
    parser.add_argument('--num-workers', type=int, default=2, help='number of workers for the pytorch data loader')
    parser.add_argument('--lr', type=float, default=0.001, help='the maximum learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-06, help='optimizer weight decay')
    parser.add_argument('--epochs', type=int, default=50, help='the number of epochs of training')
    parser.add_argument('--grad-thresh', type=float, default=5.0, help='the gradient threshold for gradient clipping')
    parser.add_argument('--fp16', action='store_true', help='use 16bit precision')
    parser.add_argument('--datadir', type=str, default='./data', help='the directory where data, i.e., features and cuts will be stored')
    parser.add_argument('--expdir', type=str, default='.', help='the directory in which experiments and models will be stored')
    parser.add_argument('--print-interval', type=int, default=1, help='number of minibatches before prtining training statistics')
    parser.add_argument('--save-interval', type=int, default=1, help='number of epochs between saves of the model to disk')
    parser.add_argument('--keep-interval', type=int, default=10, help='keep each {keep_interval} checkpoints')
    parser.add_argument('--wer-logging', action='store_true', help='print out the WER or PER per minibatch as well a an example decoded utterance')
    parser.add_argument('--keep-last', type=int, default=5, help='keep the last {keep_last} models and delete the ones prior to this')
    parser.add_argument('--resume', type=str, default=None, help='the checkpoint (state_dict) from which to resume training')
    parser.add_argument('--num-mel', type=int, default=64, help='number of log mel filterbanks to use for speech features')
    parser.add_argument('--master-port', type=int, default=12345)
    parser.add_argument('--skip-data-prep', action='store_true', help='skip the data preparation part')
    parser.add_argument('--speed-perturb', action='store_true', help='use 3x speed perturbation in training')
    args, leftover = parser.parse_known_args()
    ESPnet.add_args(parser)
    parser.parse_args(leftover, namespace=args)
    return args


def main(rank, world_size, args):
    # Fix the random seed
    fix_random_seed(42)

    # Set up logging
    logging.basicConfig(
        format='%(asctime)s -- %(levelname)s:%(message)s',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(
                filename=f"{args.expdir}/log.{rank}",
                encoding='utf-8',
                mode='w',
            ),
        ],
    )
    # Set up distributed training (on multiple GPU) if the world_size,
    # i.e., num_gpus requested > 1
    if world_size > 1:
        setup_dist(rank, world_size, master_port=args.master_port)
    
    # Load the cuts (already packaged as feats + supervisions)
    cuts_train = load_manifest_lazy(
        f"{args.datadir}/librispeech_cuts_train-clean-100.jsonl.gz"
    )
    cuts_dev = load_manifest_lazy(
        f"{args.datadir}/librispeech_cuts_dev-clean.jsonl.gz"
    )

    cuts_train = cuts_train.filter(
        lambda c: args.min_duration <= c.duration < args.max_duration
    )

    # Load the lexicon and phones
       
    # Create the PhoneCollater from the lexicon and phones. The tokenizer is
    # responsible for converting the transcripts of a set of utterances into 
    # integers representing the tokens into which the transcripts were
    # decomposed. If the tokenizer already exists, then load it. Otherwise,
    # create the tokenizer and dump it to the exp directory.
    tokenizerpath = Path(f"{args.expdir}/tokenizer.bz2")
    if tokenizer.is_file():
        tokenizer = BPETokenizer.load(str(tokenizerpath))
        logging.warning(f"Loading prexisting tokenizer. To discard the current "
            "tokenizer, stop execution and remove {args.expdir}/tokenizer.bz2"
        ) 
    else:
        tokenizer = LexiconTokenizer(lexicon, phoneset)
        if rank == 0:
            tokenizer.serialize(f"{args.expdir}/tokenizer.bz2")

    # Create instance of the ASRDataset for use with the training cuts. We use
    # SpecAugment by default, but we do not use any time warping.
    # We also use CutConcatenation by default (See lhotse for details)
    input_transforms = [
        SpecAugment(
            time_warp_factor=0,
            num_frame_masks=10,
            features_mask_size=21,
            num_feature_masks=2,
            frames_mask_size=100,
        )
    ]

    # Now we create the datasets. Note that we do not use SpecAugment for the
    # dev dataset
    ds = AstroDataset(
        tokenizer,
        input_transforms=input_transforms,
        #cut_transforms=[CutConcatenate(duration_factor=1.0, gap=1.0)],
        num_mel=args.num_mel,
        use_feats=True,
    )
    ds_dev = AstroDataset(
        tokenizer,
        #cut_transforms=[CutConcatenate(duration_factor=1.0, gap=1.0)],
        num_mel=args.num_mel,
        use_feats=True,
    )

    # Create the training sampler and data loader
    train_sampler = DynamicBucketingSampler(
        cuts_train,
        max_duration=args.minibatch_duration,
        shuffle=True,
        num_buckets=args.num_buckets,
    )
    
    train_dloader = torch.utils.data.DataLoader(
        ds,
        sampler=train_sampler,
        batch_size=None,
        num_workers=args.num_workers,
        persistent_workers=False,
        worker_init_fn=_SeedWorkers(torch.randint(0, 100000, ()).item())
    )

    # Create the dev sampler and data loader
    dev_sampler = DynamicBucketingSampler(
        cuts_dev,
        max_duration=600,
        shuffle=False,
    )

    dev_dloader = torch.utils.data.DataLoader(
        ds_dev,
        sampler=dev_sampler,
        batch_size=None,
        num_workers=2,
        persistent_workers=False,
    )

    # Create the model and distribute across GPUs
    mdl = ESPnet(
        args.num_mel,
        num_blocks=args.conformer_num_blocks,
        hidden_dim=args.conformer_hidden_dim,
        output_dim=args.conformer_final_dim,
    )

    # Create the loss function. In ASTRO, the loss function has the linear
    # layer "built-in", so we need to know the dimension of the inputs to that
    # layer as well as the number of outputs of that layer. The inputs are the
    # dimension of the model outputs. The linear layer then outputs as many
    # phones as we have in our lexicon.
    loss_fn = CTC(mdl, len(phoneset)) 

    # Resume training from checkpoint (if provided)
    args.start_epoch = 0
    if args.resume is not None:
        params_dict = torch.load(args.resume, map_location='cpu')
        loss_fn.load_state_dict(params_dict['criterion_and_model'])
        args.start_epoch = mdl_dict['epoch'] + 1

    # DistributedDataParallel for multigpu training
    device = torch.device(
        f"cuda:{rank}" if torch.cuda.is_available() and args.world_size > 0 else "cpu"
    )

    loss_fn.to(device)
    if world_size > 1:
        loss_fn = DDP(loss_fn, device_ids=[rank])

    # Create the optimizer
    params = list(
        filter(
            lambda p: p.requires_grad,
            loss_fn.parameters(),
        )
    )
    optim = torch.optim.AdamW(
        params, lr=args.lr, weight_decay=args.weight_decay,
    )

    
    # Set up the learning rate scaler needed for fp16 computation if requested
    scaler = None
    if args.fp16:
        logging.info("Using fp16 operations")
        scaler = torch.cuda.amp.GradScaler()

    if args.resume is not None:
        optim.load_state_dict(mdl_dict['optimizer'])
        for state in optim.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
        # Create the LR-Schedule, possibly resuming from a checkpoint
        lr_sched = torch.optim.lr_scheduler.OneCycleLR(
            optim, max_lr=args.lr,
            steps_per_epoch=sum(1 for b in train_dloader.sampler),
            epochs=args.epochs,
        )
        lr_sched.load_state_dict(mdl_dict['lr_sched'])
        if args.fp16:
            scaler.load_state_dict(mdl_dict['scaler'])
    else:
        lr_sched = torch.optim.lr_scheduler.OneCycleLR(
            optim, max_lr=args.lr,
            steps_per_epoch=sum(1 for b in train_dloader.sampler),
            epochs=args.epochs,
        )

    # Run the main training loop now that everything is set up
    train(args, phoneset, loss_fn, optim, scaler, lr_sched,
        train_dloader, dev_dloader, device,
    ) 


if __name__ == "__main__":
    args = get_args()
    # Make expdir if it does not exist
    expdirpath = Path(args.expdir)
    expdirpath.absolute().mkdir(parents=True, exist_ok=True)

    # Make datadir if it does not exist
    datadirpath = Path(args.datadir)
    datadirpath.absolute().mkdir(parents=True, exist_ok=True) 
    
    # Create the a configurations file storing the experiment configuartions
    # if it doesn't yet exist.
    confpath = Path(f'{args.expdir}/conf.json')
    import json
    json.dump(
        vars(args),
        open(confpath, 'w'),
        indent=4, separators=(',', ': ')
    )
    with open(f'{args.expdir}/command.txt', 'w', encoding='utf-8') as f:
        print(sys.argv, file=f) 
   
    # Prepare data
    if not args.skip_data_prep:
        parts = [
            'train-clean-100', 'dev-clean',
            'dev-other', 'test-clean', 'test-other',
        ]
        for p in tqdm(parts):
            prepare_librispeech(
                args.data, p, num_jobs=1, output_dir=args.datadir,
            )
            sp = (p == 'train-clean-100' and args.speed_perturb)
            prepare_fbank_cuts(
                args.datadir, p, args.num_mel, args.datadir,
                speed_perturb=sp,
                num_jobs=80,
            )

    logging.basicConfig(level=logging.DEBUG)
    # spawn the word_size number of jobs
    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1:
        mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)
    else:
        main(rank=0, world_size=1, args=args)


    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

