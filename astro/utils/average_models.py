import argparse
import torch
from pathlib import Path
from astro.models.ESPnet import ESPnet
from astro.criteria.CTC import CTC
import json


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-dir', type=str, required=True)
    parser.add_argument('--checkpoints', nargs='+', type=int)
    return parser.parse_args()


def main(args):
    checkpoint_dir = Path(args.checkpoint_dir)
    with open(str(checkpoint_dir / 'conf.json'), 'r', encoding='utf-8') as f:
        conf = json.load(f)
    
    state_dict = torch.load(
        str(checkpoint_dir / f'{args.checkpoints[0]}.mdl'),
        map_location='cpu'
    ) 
    #model_class = getattr(models, conf['model_type']) 
    mdl = ESPnet(
        conf['num_mel'],
        num_blocks=conf['conformer_num_blocks'],
        hidden_dim=conf['conformer_hidden_dim'],
        output_dim=conf['conformer_final_dim'],
    )

    phoneset_len = state_dict['criterion_and_model']['classifier.weight'].size(0)
    import pdb; pdb.set_trace()
    loss_fun = CTC(mdl, phoneset_len)
  
    new_dict = loss_fun.state_dict()
    for name, param in new_dict.items():
        if len(param.size()) > 0:
            param.mul_(0.0)

    fraction = 1.0 / len(args.checkpoints)
    for cp in args.checkpoints:
        state_dict = torch.load(
            str(checkpoint_dir / f'{cp}.mdl'),
            map_location=torch.device('cpu'),
        )
        for name, p in state_dict['criterion_and_model'].items():
            if name in new_dict:
                if len(p.size()) != 0:
                    new_dict[name].add_(p, alpha=fraction)
                else:
                    new_dict[name] = (p * fraction).type(new_dict[name].dtype)

    model_prefix = '_'.join(str(i) for i in args.checkpoints)
    torch.save(
        {'criterion_and_model': new_dict},
        str(checkpoint_dir / f"{model_prefix}.mdl")
    )


if __name__ == "__main__":
    args = get_args()
    main(args)
