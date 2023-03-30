import torch
import torch.nn as nn
import logging
from itertools import chain
from astro.decoders.ArgmaxPhoneDecoder import ArgmaxPhoneDecoder
import editdistance
import contextlib


def train_one_epoch(phones, loss_fn, optim, lr_sched, scaler,
    train_dloader, device, wer_logging=True, fp16=True, grad_thresh=5.0,
    print_interval=1,
):
    '''
        The training loop for 1 epoch.

        Inputs:
            :param phones:   A BPE model for tokenization
            :param mdl:  A pytorch neural network model for ASR training
            :param loss_fn: A training loss
            :param optim: A pytorch optimizer
            :param lr_sched: A learning rate scheduler (with .step() method)
            :param train_dloader: A pytorch DataLoader
            :param device: the device (cpu / cuda) on which to run
    '''
    # Use fp16
    if fp16:
        logging.info("Using fp16 operations")
        scale_context = torch.cuda.amp.autocast()
    else:
        scale_context = contextlib.nullcontext()

    if wer_logging:
        decoder = ArgmaxPhoneDecoder(phones)

    loss_fn.train()
    
    #num_batches = sum(1 for b in train_dloader.sampler)
    #duration = sum(c.duration for b in train_dloader.sampler for c in b) / 3600
    #logging.info(f"Training on {duration} hr")
    for batch_idx, b in enumerate(train_dloader):
        batch_size = b.input.size(0)
        b.to(device) 
        logging.info(f'Progress: {batch_idx + 1}')
        continue;
        with scale_context:
            loss, outputs, _ = loss_fn(b)

        if loss is None:
            logging.warning("Length mismatch. Skipping ...")
            del b
            del loss
            continue;
        elif loss.isinf() or loss.isnan():
            logging.warning("NaN or Inf loss. Skipping ...")
        
        loss = loss.sum() / batch_size 
 
        if fp16:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        loss.detach()
        if fp16:
            scaler.unscale_(optim)
        grad_norm = nn.utils.clip_grad_norm_(loss_fn.parameters(), grad_thresh)
        if fp16:
            scaler.step(optim)
            scaler.update()
        else:
            optim.step()
        optim.zero_grad()
        lr_sched.step()
        
        if batch_idx % print_interval == 0: 
            logging.info(
                f'Progress: {batch_idx + 1} ' #/{num_batches} '
                f'Loss: {loss.data.item()} '
                f'Batchsize: {batch_size} '
                f'Gradnorm: {grad_norm} '
                f'LR: {lr_sched.get_last_lr()} '
                f"Scale: {scaler.get_scale() if scaler is not None else ''}"
            )
            if wer_logging:
                output_text = decoder(outputs)
                errors, num_ref = 0, 0
                for i, text in enumerate(output_text):
                    ref = list(
                        map(
                            lambda x: str(x.data.item()),
                            b.targets[0][i][b.targets[0][i] >= 0]
                        )
                    ) #phones[x]
                    if i == 0:
                        ref_print_out = ' '.join(ref)
                        hyp_print_out = text
                    errors += editdistance.eval(ref, text.split())
                    num_ref += len(ref)
                logging.info(f"hyp: {hyp_print_out}")
                logging.info(f"ref: {ref_print_out}")
                logging.info(f"WER: {100*errors/num_ref:.02f}")
        del b

