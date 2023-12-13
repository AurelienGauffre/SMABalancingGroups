# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# !/usr/bin/env python

import argparse
import json
import numpy as np
import os
import submitit
import sys
import time
import torch
import models
import itertools
from datasets import get_loaders
from setup_datasets import generate_metadata_SMA
from omegaconf import OmegaConf
# from deephyper.problem import HpProblem
# from deephyper.evaluator import Evaluator
# from deephyper.evaluator.callback import TqdmCallback
from utils import Tee, flatten_dictionary_for_wandb,results_to_log_dict

import wandb

def randl(l_):
    return l_[torch.randperm(len(l_))[0]]


def parse_args():
    parser = argparse.ArgumentParser(description='Training configurations.')
    parser.add_argument('--config', type=str, default='config.yaml')
    return parser.parse_args()


def run_experiment(args):
    if 'SMA' not in args:
        args.SMA = None
    wandb.init(project=args.wandb_project, config=flatten_dictionary_for_wandb(dict(args)))
    print(f'Method: {args.method}')
    #print(args)
    start_time = time.time()
    torch.manual_seed(args["init_seed"])
    np.random.seed(args["init_seed"])
    loaders = get_loaders(args["data_path"], args["dataset"], args["batch_size"], args["method"], args.SMA)

    sys.stdout = Tee(os.path.join(
        args["output_dir"], f'seed_{args["hparams_seed"]}_{args["init_seed"]}.out'), sys.stdout)
    sys.stderr = Tee(os.path.join(
        args["output_dir"], f'seed_{args["hparams_seed"]}_{args["init_seed"]}.err'), sys.stderr)
    checkpoint_file = os.path.join(
        args["output_dir"], f'seed_{args["hparams_seed"]}_{args["init_seed"]}.pt')
    best_checkpoint_file = os.path.join(
        args["output_dir"],
        f"seed_{args['hparams_seed']}_{args['init_seed']}.best.pt")

    model = {
        "erm": models.ERM,
        "suby": models.ERM,
        "subg": models.ERM,
        "rwy": models.ERM,
        "rwg": models.ERM,
        "dro": models.GroupDRO,
        "jtt": models.JTT
    }[args["method"]](args, loaders["tr"])

    last_epoch = 0
    best_selec_val = float('-inf')
    # Deactivate loading model for now
    # if os.path.exists(checkpoint_file):
    #     model.load(checkpoint_file)
    #     last_epoch = model.last_epoch
    #     best_selec_val = model.best_selec_val

    for epoch in range(last_epoch, args["num_epochs"]):
        if epoch == args["T"] + 1 and args["method"] == "jtt":
            loaders = get_loaders(
                args["data_path"],
                args["dataset"],
                args["batch_size"],
                args["method"],
                model.weights.tolist())

        for i, x, y, g in loaders["tr"]:
            model.update(i, x, y, g, epoch)

        result = {"epoch": epoch, "time": time.time() - start_time}
        if epoch % args.eval_every_n_epochs == 0:
            for loader_name, loader in loaders.items():
                print('yes')
                avg_acc, group_accs = model.accuracy(loader)
                result["acc_" + loader_name] = group_accs
                result["avg_acc_" + loader_name] = avg_acc

            selec_value = {
                "min_acc_va": min(result["acc_va"]),
                "avg_acc_va": result["avg_acc_va"],
            }[args["selector"]]

            if selec_value >= best_selec_val:
                model.best_selec_val = selec_value
                best_selec_val = selec_value
                model.save(best_checkpoint_file)

        # model.save(checkpoint_file) # Deactivate saving model for now
        # result["args"] = OmegaConf.to_container(result["args"], resolve=True)
        print(json.dumps(result))
        log_dict = results_to_log_dict(result)
        wandb.log(log_dict, step=epoch)
    wandb.finish()


if __name__ == "__main__":
    args_command_line = parse_args()
    config_dict = OmegaConf.to_container(
        OmegaConf.load(os.path.join('configs',
                                    args_command_line.config)))

    # commands = []
    # for hparams_seed in range(args["num_hparams_seeds"]):
    #     torch.manual_seed(hparams_seed)
    #     args["hparams_seed"] = hparams_seed
    #     args["dataset"] = randl(
    #         ["SMA"])
    #     args["method"] = randl(
    #         ["erm", "suby", "subg", "rwy", "rwg", "dro", "jtt"])
    #     # Global hyperparameters
    #     args["eta"] = 0.1
    #     args["lr"] = randl([1e-5, 1e-4, 1e-3])
    #     args["weight_decay"] = randl([1e-4, 1e-3, 1e-2, 1e-1, 1])
    #
    #     # Specific hyperparameters
    #     if args["dataset"] == "SMA":
    #         args["num_epochs"] = 50
    #         args["batch_size"] = 32 #randl([2, 4, 8, 16, 32])
    #         args["up"] = randl([4, 5, 6, 20, 50, 100])
    #         args["T"] = randl([40, 50, 60])
    #
    #     else :
    #         args["num_epochs"] = {
    #             "waterbirds": 300 + 60,
    #             "celeba": 50 + 10,
    #             "multinli": 5 + 2, 
    #             "civilcomments": 5 + 2
    #         }[args["dataset"]]
    #
    #         if args["dataset"] in ["waterbirds", "celeba"]:
    #             args["batch_size"] = randl([2, 4, 8])  # PERSO  [2, 4, 8,16, 32, 64, 128])
    #         else:
    #             args["batch_size"] = randl([2, 4, 8, 16, 32])
    #
    #         args["up"] = randl([4, 5, 6, 20, 50, 100])
    #         args["T"] = {
    #             "waterbirds": randl([40, 50, 60]),
    #             "celeba": randl([1, 5, 10]),
    #             "multinli": randl([1, 2]),
    #             "civilcomments": randl([1, 2])
    #         }[args["dataset"]]
    #
    #     for init_seed in range(args["num_init_seeds"]):
    #         args["init_seed"] = init_seed
    #         commands.append(dict(args))
    # torch.manual_seed(0)
    # commands = [commands[int(p)] for p in torch.randperm(len(commands))] #simple shuffle de la liste

    os.makedirs(config_dict["output_dir"], exist_ok=True)

    # The following code is a simple way to iterate over all the possible combinations of hyperparameters that are given
    # as lists in the config file
    list_keys = [k for k, v in config_dict.items() if isinstance(v, list)]  # Identify which keys have list values
    combinations = list(itertools.product(*(config_dict[k] for k in list_keys)))
    for values in combinations:
        for k, v in zip(list_keys, values):
            config_dict[k] = v
        command = OmegaConf.create(config_dict)
        run_experiment(command)

    # if args['slurm_partition'] is not None:
    #     executor = submitit.SlurmExecutor(folder=args['slurm_output_dir'])
    #     executor.update_parameters(
    #         time=args["max_time"],
    #         gpus_per_node=1,
    #         array_parallelism=512,
    #         cpus_per_task=4,
    #         partition=args["slurm_partition"])
    #     executor.map_array(run_experiment, commands)
    # else:
    #     for command in commands:
    #         run_experiment(command)
