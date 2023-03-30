# ASTRO imports
from astro.criteria.CTC import CTC
from astro.models.ESPnet import ESPnet
from astro.iterators.decode_phone import decode
from astro.decoders.K2Decoder import K2Decoder
from astro.tokenizers.LexiconTokenizer import LexiconTokenizer
from astro.datasets.AstroDataset import AstroDataset
from astro.lexicons.Lexicon import Lexicon

# Lhotse imports
from lhotse.utils import fix_random_seed
from lhotse.recipes import prepare_librispeech 
from lhotse import load_manifest_lazy
from lhotse.dataset import SpecAugment, CutConcatenate, DynamicBucketingSampler

import torch
import torch.nn as nn
import kaldilm

from pathlib import Path
import argparse
import json
import jiwer


parts = [
    'train-clean-100', 
    'dev-clean',
    'dev-other',
    'test-clean',
    'test-other',
]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, help='path to save pytorch model (as state dict)')
    parser.add_argument('--tokenizer', type=str, help='the astro tokenizer used for training') 
    parser.add_argument('--arpa-file', type=str, help='the arpa file corresponding to the lm used in decoding')
    parser.add_argument('--conf', type=str, help='path to json configuration file with training parameters')
    parser.add_argument('--datadir', type=str, help='path to datadir with cut manifests (precomputed features)')
    parser.add_argument('--part', type=str, help='name of the data partition to decode', choices=parts)
    parser.add_argument('--minibatch-duration', type=float, default=600.0, help='minibatch size')
    parser.add_argument('--decodedir', type=str, help='directory in which to dump decoding results')
    parser.add_argument('--k2', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--verbose', action='store_true')

    args, leftover = parser.parse_known_args()
    decoder = K2Decoder.add_args(parser)
    parser.parse_args(leftover, namespace=args)
    return parser.parse_args()  


def main(args): 
    device = torch.device("cuda:0")
    with open(args.conf, encoding='utf-8') as f:
        conf = json.load(f)
    
    cuts = load_manifest_lazy(
        f"{args.datadir}/librispeech_cuts_{args.part}.jsonl.gz"
    )

    # Create the decoder
    decoder = K2Decoder.from_args(args)
    if args.k2_graph is not None:
        graph_path = Path(args.k2_graph)
    if args.k2_graph is not None and not graph_path.is_file():
        torch.save(decoder.graph.as_dict(), f'{args.k2_graph}') 

    ds = AstroDataset(
        decoder.tokenizer,
        #cut_transforms=[CutConcatenate(duration_factor=1.0, gap=1.0)],
        num_mel=conf.get('num_mel', 64),
        use_feats=True,
    )

    # Create the dev sampler and data loader
    sampler = DynamicBucketingSampler(
        cuts,
        max_duration=600,
        shuffle=False,
    )

    dloader = torch.utils.data.DataLoader(
        ds,
        sampler=sampler,
        batch_size=None,
        num_workers=2,
    )

    mdl = ESPnet(
        conf.get('num_mel', 64),
        num_blocks=conf.get('conformer_num_blocks', 6),
        hidden_dim=conf.get('conformer_hidden_dim', 256),
        output_dim=conf.get('conformer_final_dim', 256),
    )

    loss_fn = CTC(mdl, len(decoder.tokenizer.phones))
    
    mdl_dict = torch.load(args.checkpoint, map_location='cpu')
    loss_fn.load_state_dict(mdl_dict['criterion_and_model'])
    loss_fn.to(device)
    ids, refs, hyps = decode(loss_fn, decoder, dloader, device,
        fp16=args.fp16, verbose=args.verbose
    )
    with open(decodedir / 'hyps', 'w', encoding='utf-8') as fh:
        with open(decodedir / 'refs', 'w', encoding='utf-8') as fr:
            for id, ref, hyp in zip(ids, refs, hyps):
                print(f'{id} {ref}', file=fr)
                print(f'{id} {hyp}', file=fh)
    with open(decodedir / 'wer', 'w', encoding='utf-8') as f:
        wer = jiwer.compute_measures(refs, hyps)
        print(f'{wer}', file=f)
    if args.verbose:
        print(wer)


if __name__ == "__main__":
    args = get_args()

    # Make the decodedir directory if it does not exist
    decodedir = Path(args.decodedir)
    decodedir.absolute().mkdir(parents=True, exist_ok=True)

    # Make datadir if it does not exist

    # Prepare the lexicon and LM
    if args.k2:
        if args.k2_graph is not None:
            k2_graph_path = Path(args.k2_graph) 
            if (
                not k2_graph_path.is_file() and
                (
                    args.k2_lexicon is None or
                    args.k2_lm is None or
                    args.k2_words is None
                )
            ):
                raise RuntimeError('''When the graph does not exist, then the 
                    lexicon, lm, and words for k2 decoding must be provided.
                    These are set with --k2-lm, --k2-lexicon, --k2-words. These
                    parameters are used to create the LM and lexicon FSTs.'''
                )     
        if not Path(args.k2_lexicon).is_file():
            lexicon = Lexicon.from_tokenizer(args.tokenizer)
            lexicon.add_disambig_symbols()
            lexicon_fst = lexicon.to_fst_no_sil(need_self_loops=True)
            torch.save(lexicon_fst.as_dict(), args.k2_lexicon)
            lexicon.word2id.to_file(args.k2_words)
        
        if not Path(args.k2_lm).is_file():
            if Path(args.arpa_file).is_file():
                G = kaldilm.arpa2fst(
                    args.arpa_file,
                    disambig_symbol="#0",
                    read_symbol_table=args.k2_words,
                )
                with open(args.k2_lm, 'w') as f:
                    print(G, file=f)
            else:
                raise NotImplementedError
    main(args) 
