"""
PyTorch scripts to train/test orientation discrimination.
"""

import os

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
        _test_sensitivity(args, model)
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


def _make_test_loader(args, rotation, grating_angle):
    kwargs = {'thetas': [grating_angle], 'rotation': rotation}
    db = dataloader_orientation.test_set(args.target_size, args.preprocess, **kwargs)

    return torch.utils.data.DataLoader(
        db, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True
    )


def _sensitivity_orientation(args, model, degree, direction):
    low = 0
    high = direction * 90
    mid = report_utils.compute_avg(low, high)

    attempt_i = 0
    # th=0.749 because test samples are 16, 12 correct equals 0.75 and test stops
    th = 0.749
    while True:
        db_loader = _make_test_loader(args, mid, degree)

        def name_gen(x):
            return 'img%03d_%03d_%d' % (x, degree, direction)

        _, accuracy = common_routines.train_val(
            db_loader, model, None, -1 - attempt_i, args, print_test=False, name_gen=name_gen
        )

        print(degree, mid, accuracy, low, high)
        new_low, new_mid, new_high = report_utils.midpoint(accuracy, low, mid, high, th=th)
        if new_mid is None or attempt_i == args.test_attempts:
            print('had to skip')
            break
        else:
            low, mid, high = new_low, new_mid, new_high
        attempt_i += 1
    return accuracy, mid


def _test_sensitivity(args, model):
    test_orientations = np.arange(0, 181, 15)

    # preparing the output file
    res_out_dir = os.path.join(args.output_dir, 'evals')
    out_file = '%s/%s_evolution.csv' % (res_out_dir, args.experiment_name)
    if os.path.exists(out_file):
        return
    system_utils.create_dir(res_out_dir)

    # preparing the tensorboard writer
    tb_path = os.path.join(args.output_dir, 'test_%s' % args.experiment_name)
    args.tb_writers = {'test': SummaryWriter(tb_path)}

    header = 'Orientation,P_Acc,N_Acc,P_Angle,N_Angle'
    all_results = []
    for degree in test_orientations:
        acc_p, ang_p = _sensitivity_orientation(args, model, degree, +1)
        acc_n, ang_n = _sensitivity_orientation(args, model, degree, -1)
        sensitivity = (ang_p + abs(ang_n)) / 2
        all_results.append([degree, acc_p, acc_n, ang_p, abs(ang_n)])
        np.savetxt(out_file, np.array(all_results), delimiter=',', fmt='%f', header=header)
        args.tb_writers['test'].add_scalar("{}".format('sensitivity'), sensitivity, degree)
    args.tb_writers['test'].close()
