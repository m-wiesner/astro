import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import logging
from astro.decoders.ArgmaxPhoneDecoder import ArgmaxPhoneDecoder
import contextlib


def decode(loss_fn, decoder, dev_dloader, device, fp16=True, verbose=True): 
    loss_val = 0.0
    loss_fn.eval()
    loss_fn.decode()
    ids, hyps, refs = [], [], []
    scale_context = autocast() if fp16 else contextlib.nullcontext()
    for batch_idx, b in enumerate(dev_dloader):
        b.to(device)
        with torch.no_grad(), scale_context:
            output, lengths = loss_fn(b)
            output_text = decoder(output, lengths=lengths)
            for i, text in enumerate(output_text):
                ref = b.metadata['text'][i]
                if decoder.token_type == 'phone':
                    ref = decoder.tokenizer.words_to_phones(ref)
                utt_id = b.metadata['ids'][i]
                refs.append(ref)
                hyps.append(text)
                ids.append(utt_id)
                if verbose:
                    print(f'hyp: {text}')
                    print(f'ref: {ref}', end="\n\n-----\n\n")
            del b
    return ids, refs, hyps

