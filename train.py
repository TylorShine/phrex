import argparse
import torch

from modules.common import load_config, load_model
from modules.dataset.loader import get_data_loaders
from modules.solver import train
from modules.decoder import Phrex


torch.backends.cudnn.benchmark = True


def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="path to the config file")
    return parser.parse_args(args=args, namespace=namespace)


if __name__ == '__main__':
    # parse commands
    cmd = parse_args()
    
    # load config
    args = load_config(cmd.config)
    print(' > config:', cmd.config)
    print(' >    exp:', args.env.expdir)

    # load model
    model = None
    if args.model.type == 'phrex':
        model = Phrex(
            sampling_rate=args.data.sampling_rate,
            block_size=args.data.block_size,
            in_channels=args.model.in_channels,
            hidden_channels=args.model.hidden_channels,
            out_channels=args.model.out_channels)
        # model.compile(mode="reduce-overhead")
            
    else:
        raise ValueError(f" [x] Unknown Model: {args.model.type}")
    
    
    # load model parameters
    optimizer = torch.optim.AdamW(model.parameters())
    initial_global_step, model, optimizer, states = load_model(args.env.expdir, model, optimizer, device=args.device)
    
    lr = args.train.lr if states is None else states['last_lr'][0]
    
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = args.train.lr
        param_group['lr'] = lr
        param_group['weight_decay'] = args.train.weight_decay
        
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.train.sched_factor, patience=args.train.sched_patience,
                                                        threshold=args.train.sched_threshold, threshold_mode=args.train.sched_threshold_mode,
                                                        cooldown=args.train.sched_cooldown, min_lr=args.train.sched_min_lr)
    if states is not None:
        sched_states = states.get('scheduler')
        if sched_states is not None:
            scheduler.best = sched_states['best']
            scheduler.cooldown_counter = sched_states['cooldown_counter']
            scheduler.num_bad_epochs = sched_states['num_bad_epochs']
            scheduler._last_lr = sched_states['_last_lr']
    else:
        scheduler._last_lr = (lr,)
    

    # device
    if args.device == 'cuda':
        torch.cuda.set_device(args.env.gpu_id)
    model.to(args.device)
    
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(args.device)


    # datas
    loaders = get_data_loaders(args)
    
    
    # run
    train(args, initial_global_step, (model, optimizer, scheduler), loaders['train'], loaders['test'])
    
