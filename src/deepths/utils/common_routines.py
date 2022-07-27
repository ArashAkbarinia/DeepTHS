"""

"""

import os
import numpy as np
import collections

import torch
from torch.utils.tensorboard import SummaryWriter

from . import system_utils
from ..models import model_utils


def _prepare_training(args, model):
    optimizer = _make_optimizer(args, model)

    initial_epoch = args.initial_epoch
    model_progress = []
    progress_path = os.path.join(args.output_dir, 'model_progress.csv')

    # optionally resume from a checkpoint
    if args.resume is not None:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location='cpu')
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

            initial_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            model = model.cuda(args.gpu)

            optimizer.load_state_dict(checkpoint['optimizer'])

            if os.path.exists(progress_path):
                model_progress = np.loadtxt(progress_path, delimiter=',')
                model_progress = model_progress.tolist()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    return model, optimizer, initial_epoch, model_progress, progress_path


def _make_optimizer(args, model):
    if args.classifier == 'nn':
        # if transfer_weights, only train the fc layer, otherwise all parameters
        if args.transfer_weights is None:
            params_to_optimize = [{'params': [p for p in model.parameters()]}]
        else:
            for p in model.features.parameters():
                p.requires_grad = False
            params_to_optimize = [{'params': [p for p in model.fc.parameters()]}]
        # optimiser
        optimizer = torch.optim.SGD(
            params_to_optimize, lr=args.learning_rate,
            momentum=args.momentum, weight_decay=args.weight_decay
        )
    else:
        optimizer = []
    return optimizer


def prepare_starting(args, task_folder):
    system_utils.set_random_environment(args.random_seed)

    if args.classifier != 'nn':
        args.epochs = 1
        args.print_freq = np.inf

    # preparing the output folder
    layer = args.transfer_weights[1]
    args.output_dir = '%s/%s/%s/%s/%s/%s/' % (
        args.output_dir, task_folder, args.dataset, args.architecture, args.experiment_name, layer
    )
    system_utils.create_dir(args.output_dir)

    args.mean, args.std = model_utils.get_mean_std(args.colour_space, args.vision_type)
    args.preprocess = (args.mean, args.std)

    # dumping all passed arguments to a json file
    if not args.test_net:
        system_utils.save_arguments(args)
    return args


def do_epochs(args, epoch_fun, train_loader, val_loader, model):
    model, optimizer, args.initial_epoch, model_prog, prog_path = _prepare_training(args, model)

    # create the tensorboard writers
    args.tb_writers = dict()
    for mode in ['train', 'val']:
        args.tb_writers[mode] = SummaryWriter(os.path.join(args.output_dir, mode))

    # training on epoch
    for epoch in range(args.initial_epoch, args.epochs):
        if args.classifier == 'nn':
            _adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train_log = epoch_fun(train_loader, model, optimizer, epoch, args)

        # evaluate on validation set
        validation_log = epoch_fun(val_loader, model, None, epoch, args)

        model_prog.append([*train_log, *validation_log[1:]])

        # remember best acc@1 and save checkpoint
        acc1 = validation_log[2]
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        # save the checkpoints
        system_utils.save_checkpoint(
            {
                'epoch': epoch,
                'arch': args.architecture,
                'transfer_weights': args.transfer_weights,
                'preprocessing': {'mean': args.mean, 'std': args.std},
                'state_dict': _extract_altered_state_dict(model, args.classifier),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict() if args.classifier == 'nn' else [],
                'target_size': args.target_size,
            },
            is_best, args
        )
        header = 'epoch,t_time,t_loss,t_top1,v_time,v_loss,v_top1'
        np.savetxt(prog_path, np.array(model_prog), delimiter=',', header=header)

    # closing the tensorboard writers
    for mode in args.tb_writers.keys():
        args.tb_writers[mode].close()


def _extract_altered_state_dict(model, classifier):
    altered_state_dict = collections.OrderedDict()
    for key, _ in model.named_buffers():
        altered_state_dict[key] = model.state_dict()[key]
    if classifier == 'nn':
        for key in ['fc.weight', 'fc.bias']:
            altered_state_dict[key] = model.state_dict()[key]
    return altered_state_dict


def _adjust_learning_rate(optimizer, epoch, args):
    lr = args.learning_rate * (0.1 ** (epoch // (args.epochs / 3)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
