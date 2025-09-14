import os
import json
import sys
import pprint

sys.path.append(".")
sys.path.append("..")

from options.train_options import TrainOptions
from training.coach import Coach


def main():
    opts = TrainOptions().parse()
    if not os.path.exists(opts.exp_dir):
        os.mkdir(opts.exp_dir)

    opts_dict = vars(opts)
    pprint.pprint(opts_dict)
    job_dir = os.path.join(opts.exp_dir, opts.job_name)
    os.makedirs(job_dir, exist_ok=True)
    with open(os.path.join(job_dir, 'opt.json'), 'w') as f:
        json.dump(opts_dict, f, indent=4, sort_keys=True)

    local_rank = int(os.environ['LOCAL_RANK'])

    coach = Coach(opts, local_rank)
    coach.train_ref_encoder()


if __name__ == '__main__':
    main()
