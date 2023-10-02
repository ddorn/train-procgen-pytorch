import argparse
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm

from common import set_global_seeds, set_global_log_levels
from run_utils import run_env

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='metrics', help='experiment name')
    parser.add_argument('--random_percent', type=float, default=0)
    parser.add_argument('--num_seeds', type=int, default=10)
    parser.add_argument('--log_file', type=str, default=None)
    parser.add_argument('--model_file', type=str, help="Can be either a path to a model file, or an "
                                                       "integer. Integer is interpreted as random_percent in training")
    parser.add_argument('--gpu', action='store_true', help='use gpu')
    # Misc.
    parser.add_argument('--gpu_device', type=int, default=int(0), required=False, help='visible device in CUDA')
    parser.add_argument('--start_level', type=int, default=int(0), help='start-level for environment')
    parser.add_argument('--start_level_seed', type=int, default=0)
    parser.add_argument('--seed_file', type=str, help="path to text file with env seeds to run on.")
    parser.add_argument('--agent_seed', type=int, default=random.randint(0, 999999), help='Seed for pytorch')
    parser.add_argument('--log_level', type=int, default=int(40), help='[10,20,30,40]')
    parser.add_argument('--progress', action='store_true', help='show progress bar')
    parser.add_argument('--reset_mode', type=str, default="inv_coin", help="Reset modes:"
                                                                           "- inv_coin returns when agent gets the inv coin OR finishes the level"
                                                                           "- complete returns when the agent finishes the level")
    args = parser.parse_args()

    # Seeds
    set_global_seeds(args.agent_seed)
    set_global_log_levels(args.log_level)

    if args.seed_file:
        print(f"Loading env seeds from {args.seed_file}")
        with open(args.seed_file) as f:
            seeds = f.read()
        seeds = [int(s) for s in seeds.split()]
    else:
        print(f"Running on env seeds {args.start_level_seed} to {args.start_level_seed + args.num_seeds}.")
        seeds = np.arange(args.num_seeds) + args.start_level_seed

    logfile = Path(args.log_file).resolve()
    logfile.parent.mkdir(parents=True, exist_ok=True)
    path_to_model_file = args.model_file

    print(f"Saving metrics to {str(logfile)}")
    print(f"Running coinrun with random_percent={args.random_percent}...")
    for env_seed in tqdm(seeds, disable=not args.progress):
        run_env(exp_name=args.exp_name,
                logfile=logfile,
                model_file=path_to_model_file,
                level_seed=env_seed,
                device='gpu' if args.gpu else 'cpu',
                gpu_device=args.gpu_device,
                random_percent=args.random_percent,
                reset_mode=args.reset_mode)
