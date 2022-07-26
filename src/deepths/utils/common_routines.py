"""

"""

import os
import numpy as np
import collections

from torch.utils.tensorboard import SummaryWriter

from . import system_utils


def do_epochs(args, epoch_fun, optimizer, train_loader, val_loader, model,
              model_progress, model_progress_path):
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

        model_progress.append([*train_log, *validation_log[1:]])

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
        np.savetxt(model_progress_path, np.array(model_progress), delimiter=',', header=header)

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
