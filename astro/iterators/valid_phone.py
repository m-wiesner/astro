import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import logging
from astro.decoders.ArgmaxPhoneDecoder import ArgmaxPhoneDecoder
import editdistance
import contextlib


def valid_phone(phones, loss_fn, dev_dloader, device, wer_logging=True, fp16=True): 
    if wer_logging:
        decoder = ArgmaxPhoneDecoder(phones)
    loss_val = 0.0
    loss_fn.eval()
    num_val = 0.0
    errors, num_ref = 0, 0
    scale_context = autocast() if fp16 else contextlib.nullcontext()
    for batch_idx, b in enumerate(dev_dloader):
        b.to(device)
        with torch.no_grad(), scale_context:
            loss, output, _ = loss_fn(b)
            loss_val += loss.sum().data.item()
            if wer_logging:
                output_text = decoder(output)
                for i, text in enumerate(output_text):
                    ref = list(
                        map(
                            lambda x: str(x.data.item()),
                            b.targets[0][i][b.targets[0][i] >= 0]
                        )
                    )
                    errors += editdistance.eval(ref, text.split())
                    num_ref += len(ref)
            del b
            num_val += 1.0
    loss_fn.train()
    loss_val /= num_val
    return loss_val, 100.0*errors / num_ref
    
