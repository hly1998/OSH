from utils.tools import *
from model_backbone.network_xnor import ResNet_XNOR
from model_backbone.network_ir import ResNet_IR
from model_backbone.network_reactnet import ResNet_ReactNet
from model_backbone.network_bi2real import ResNet_Bi2Real
from model_backbone.network_recu import ResNet_Recu
from model_backbone.network_rbonn import ResNet_Rbonn
from model_backbone.network_rebnn import ResNet_Rebnn

import torch
import logging
import argparse
from model_hash.Ours_BNN import train_val

def get_argparser():
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type=str, default='imagenet', help='choose from imagenet,cifar100,nuswide_21')
        parser.add_argument('--optimizer', type=str, default='Adam')
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--weight_decay', type=float, default=10 ** -5)
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--epoch', type=int, default=700)
        parser.add_argument('--test_map', type=int, default=10)
        parser.add_argument('--stop_iter', type=int, default=7)
        parser.add_argument('--device', type=int, default=1)
        parser.add_argument('--bit', type=int, default=64)
        parser.add_argument('--bnn_model', type=str, default="ResNet_XNOR")
        parser.add_argument('--info', type=str, default="ours")
        parser.add_argument('--temp', type=float, default=0.2)

        return parser

def get_config(args):
    optimizer_map = {
        'SGD': torch.optim.SGD,
        'ASGD': torch.optim.ASGD,
        'Adam': torch.optim.Adam,
        'Adamax': torch.optim.Adamax,
        'Adagrad': torch.optim.Adagrad,
        'Adadelta': torch.optim.Adadelta,
        'Rprop': torch.optim.Rprop,
        'RMSprop': torch.optim.RMSprop
    }
    model_map = {
        "ResNet_XNOR": ResNet_XNOR,
        "ResNet_IR": ResNet_IR,
        "ResNet_ReactNet": ResNet_ReactNet,
        "ResNet_Bi2Real": ResNet_Bi2Real,
        "ResNet_Recu": ResNet_Recu,
        "ResNet_Rbonn": ResNet_Rbonn,
        "ResNet_Rebnn": ResNet_Rebnn
    }
    config = {
        "optimizer": {"type": optimizer_map[args.optimizer], "optim_params": {"lr": args.lr, "weight_decay": args.weight_decay}},
        "batch_size": args.batch_size,
        "net": model_map[args.bnn_model],
        "dataset": args.dataset,
        "epoch": args.epoch,
        "test_map": args.test_map,
        "device": torch.device("cuda:{}".format(args.device)),
        "bit": args.bit,
        "stop_iter": args.stop_iter,
        "resize_size": 256,
        "crop_size": 224,
        "info": args.info,
        "lambda1": 0.0001,
        "lambda2": 0.001,
        "temp": args.temp,
        "transformation_scale": 0.5,
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 32,
        "eval_epoch": 400,
        "max_norm": 5.0,
        "n_positive": 2,
        "transition_epoch": 100,
        "bnn_model": args.bnn_model
    }
    config = config_dataset(config)
    return config

if __name__ == "__main__":
    argparser = get_argparser()
    args = argparser.parse_args()
    config = get_config(args)
    logging.basicConfig(filename=f"logs/Ours_{args.bnn_model}_{config['dataset']}_{config['bit']}.log", level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    train_val(config)
