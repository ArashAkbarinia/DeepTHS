"""
PyTorch scripts to train/test orientation discrimination.
"""

import os
import sys

import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

from .datasets import dataloader_orientation
from .models import model_colour as networks
from .utils import report_utils, argument_handler
from .utils import common_routines, system_utils


def main(argv):
    args = argument_handler.master_arg_parser(argv, 'orientation_discrimination')
    args = common_routines.prepare_starting(args, 'orientation_discrimination')
    _main_worker(args)


def _main_worker(args):
    torch.cuda.set_device(args.gpu)
    model = networks.colour_discrimination_net(args)
    model = model.cuda(args.gpu)

    if args.test_net:
        if args.background is None:
            args.background = 128
        elif args.background.isnumeric():
            args.background = int(args.background)
        # TODO test with sinusoidal gratins
        return

    # loading the training set
    if args.train_dir is None:
        args.train_dir = args.data_dir + '/training_set/'
    train_dataset = dataloader_orientation.train_val_set(
        args.train_dir, args.target_size, args.preprocess
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None
    )

    # loading the validation set
    if args.validation_dir is None:
        args.validation_dir = args.data_dir + '/validation_set/'
    val_dataset = dataloader_orientation.train_val_set(
        args.validation_dir, args.target_size, args.preprocess
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    common_routines.do_epochs(args, common_routines.train_val, train_loader, val_loader, model)
