import os
import json
# import yaml
import pdb
from pprint import pprint
import itertools
import ast


def main(args):
    exp_dir = os.path.join("./fast-vol/experiments", args.exp_name) 
    os.mkdir(exp_dir)
    print(f"New experiment dir created at {exp_dir}")
    
    run_info_dict = {
        'exp_name':args.exp_name,
        'dataset_name':args.dataset_name,
        'data_root':args.data_root,
        'split_props':args.split_props,
        'use_subset':args.use_subset,
        'save_every':args.save_every,
        'loader_workers':args.loader_workers,
        'use_ddp':args.use_ddp,
        'gpus_per_worker':args.gpus_per_worker
    }
    pprint(run_info_dict)
    with open(os.path.join(exp_dir, "run_info.json"), 'w') as outfile:
        json.dump(run_info_dict, outfile)  
    
    # All hyperparams
    # hparam_grid = {
    #     'lr':[0.1],
    #     'batch_size':[128],
    #     'total_epochs':[2],
    #     'warmup_iters':[1000],
    #     'lr_milestones':[[0.5,0.75]],
    #     'momentum':[0.9],
    #     'weight_decay':[1e-4],
    #     'num_layers':[18],
    #     'bins':[100],
    #     'bbox_transform':['none'],
    #     'final_size':[256],
    #     'img_norm':['normal'],
    #     'optimizer':['sgd']
    # }
    hparam_grid = {
        'lr':[0.1],
        'batch_size':[128],
        'total_epochs':[20],
        'warmup_iters':[1000],
        'lr_milestones':[[0.5,0.75]],
        'momentum':[0.9],
        'weight_decay':[5e-4, 1e-3, 2e-3],
        'num_layers':[18],
        'bins':[100],
        'bbox_transform':['none'],
        'final_size':[256],
        'img_norm':['normal'],
        'optimizer':['sgd']
    }
    # Generate possible hparam combinations
    total_trials = 0
    print(str(hparam_grid.keys()))
    for hparam_instance in itertools.product(*hparam_grid.values()):
        print("    " + str(hparam_instance))
        total_trials += 1

    hparam_instances = []
    for trial,instance in enumerate(itertools.product(*hparam_grid.values())):
        hparam_dict = dict(zip(hparam_grid.keys(), instance))
        # Check for invalid hparams
        if hparam_dict['optimizer']=="adam":
            hparam_dict['momentum'] = 0.0
        hparam_instances.append(hparam_dict)
        # Scale lr w.r.t. batch size and number of GPUs used
        base_lr = hparam_dict['lr']
        bs = hparam_dict['batch_size']
        lr_scaler = 1.0
        if bs>128:
            lr_scaler *= int(bs/128)
        if run_info_dict['use_ddp']:
            lr_scaler *= run_info_dict['gpus_per_worker']
        hparam_dict['lr_scaler'] = lr_scaler

    with open(os.path.join(exp_dir, "hparams.json"), 'w') as outfile:
        json.dump(hparam_instances, outfile)    

    
    print(f"\nTotal trials = {total_trials}")
    print(f"EXP_NAME = {args.exp_name}")
    return


# See: https://developer.nvidia.com/blog/kubernetes-ai-hyperparameter-search-experiments/

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # Run info
    parser.add_argument("--exp_name",
                        help="Experiment name.",
                        type=str)
    parser.add_argument("--dataset_name",
                        help="Dataset name: ABO360 ('abo_spins', 'abo_catalog').",
                        type=str, default="abo_spins_per_object")
    parser.add_argument("--data_root",
                        help="Dataset root directory.",
                        type=str, default="/home/jovyan/fast-vol/ABO360/spins/small")
    parser.add_argument("--split_props",
                        help="Train/val/test proportions.",
                        type=float, nargs='+', default=[0.7,0.1,0.2])
    parser.add_argument("--use_subset",
                        help="Take 1/3 subset of data for faster training.",
                        action="store_true", default=False)
    parser.add_argument("--save_every",
                        help="Save model checkpoint every N epochs.",
                        type=int, default=1)
    
    parser.add_argument("--loader_workers",
                        help="Number of workers in one Dataloader.",
                        type=int, default=8)
    parser.add_argument("--use_ddp",
                        help="Flat to toggle use of DDP for distributed training.",
                        action="store_true", default=False)
    parser.add_argument("--gpus_per_worker",
                        help="Number of workers in one Dataloader.",
                        type=int, default=1)
    
    args = parser.parse_args()
    assert sum(args.split_props)==1.0
    main(args)