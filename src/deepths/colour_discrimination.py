"""
PyTorch scripts to train/test colour discrimination.
"""

import os

import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

from .datasets import dataloader_colour
from .models import model_colour as networks
from .utils import report_utils, argument_handler
from .utils import common_routines, system_utils


def main(argv):
    args = argument_handler.master_arg_parser(argv, 'colour_discrimination')
    args = common_routines.prepare_starting(args, 'colour_discrimination')
    _main_worker(args)


def _main_worker(args):
    torch.cuda.set_device(args.gpu)
    model = networks.colour_discrimination_net(args)
    model = model.cuda(args.gpu)

    # setting the quadrant points
    if args.test_file is None:
        args.test_file = args.validation_dir + '/rgb_points.csv'
    test_pts = np.loadtxt(args.test_file, delimiter=',', dtype=str)

    args.test_pts = dataloader_colour.organise_test_points(test_pts)

    # defining validation set here so if only test don't do the rest
    if args.validation_dir is None:
        args.validation_dir = args.data_dir + '/validation_set/'

    if args.test_net:
        if args.background is None:
            args.background = 128
        elif args.background.isnumeric():
            args.background = int(args.background)
        if args.test_attempts > 0:
            _sensitivity_test_points(args, model)
        else:
            _accuracy_test_points(args, model)
        return

    # loading the validation set
    val_dataset = []
    for ref_pts in args.test_pts.values():
        others_colour = ref_pts['ffun'](np.expand_dims(ref_pts['ref'][:3], axis=(0, 1)))
        for ext_pts in ref_pts['ext']:
            target_colour = ref_pts['bfun'](np.expand_dims(ext_pts[:3], axis=(0, 1)))
            val_colours = {'target_colour': target_colour, 'others_colour': others_colour}
            val_dataset.append(dataloader_colour.val_set(
                args.validation_dir, args.target_size, args.preprocess, task=args.paradigm,
                **val_colours
            ))
    val_dataset = torch.utils.data.ConcatDataset(val_dataset)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    if args.train_dir is None:
        args.train_dir = args.data_dir + '/training_set/'

    # loading the training set
    train_kwargs = {'colour_dist': args.train_colours, **_common_db_params(args)}
    train_dataset = dataloader_colour.train_set(
        args.train_dir, args.target_size, args.preprocess, task=args.paradigm, **train_kwargs
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None
    )

    common_routines.do_epochs(args, common_routines.train_val, train_loader, val_loader, model)


def _common_db_params(args):
    return {'background': args.background, 'same_rotation': args.same_rotation}


def _sensitivity_test_points(args, model):
    for qname, qval in args.test_pts.items():
        for pt_ind in range(0, len(qval['ext'])):
            _sensitivity_test_point(args, model, qname, pt_ind)


def _accuracy_test_points(args, model):
    for qname, qval in args.test_pts.items():
        tosave = []
        for pt_ind in range(0, len(qval['ext'])):
            acc = _accuracy_test_point(args, model, qname, pt_ind)
            tosave.append([acc, *qval['ext'][pt_ind], qval['chns'][pt_ind]])
        output_file = os.path.join(args.output_dir, 'accuracy_%s.csv' % qname)
        chns_name = qval['space']
        header = 'acc,%s,%s,%s,chn' % (chns_name[0], chns_name[1], chns_name[2])
        np.savetxt(output_file, np.array(tosave), delimiter=',', fmt='%s', header=header)


def _make_test_loader(args, target_colour, others_colour):
    kwargs = {'target_colour': target_colour, 'others_colour': others_colour,
              **_common_db_params(args)}
    db = dataloader_colour.val_set(args.validation_dir, args.target_size, args.preprocess,
                                   task=args.paradigm, **kwargs)

    return torch.utils.data.DataLoader(
        db, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True
    )


def _accuracy_test_point(args, model, qname, pt_ind):
    qval = args.test_pts[qname]

    low = np.expand_dims(qval['ref'][:3], axis=(0, 1))
    high = np.expand_dims(qval['ext'][pt_ind][:3], axis=(0, 1))

    others_colour = qval['ffun'](low)
    target_colour = qval['ffun'](high)
    db_loader = _make_test_loader(args, target_colour, others_colour)

    _, accuracy = common_routines.train_val(db_loader, model, None, -1, args, print_test=False)
    print(qname, pt_ind, accuracy, low.squeeze(), high.squeeze())
    return accuracy


def _sensitivity_test_point(args, model, qname, pt_ind):
    bg_suffix = '_rnd' if type(args.background) is str else '_%.3d' % args.background
    res_out_dir = os.path.join(args.output_dir, 'evals_%s%s' % (args.experiment_name, bg_suffix))
    output_file = os.path.join(res_out_dir, 'evolution_%s_%d.csv' % (qname, pt_ind))
    if os.path.exists(output_file):
        return
    system_utils.create_dir(res_out_dir)
    tb_dir = os.path.join(args.output_dir, 'tests_%s%s' % (args.experiment_name, bg_suffix))
    args.tb_writers = {'test': SummaryWriter(os.path.join(tb_dir, '%s_%d' % (qname, pt_ind)))}

    qval = args.test_pts[qname]
    chns_name = qval['space']
    circ_chns = [0] if chns_name[0] == 'H' else []

    low = np.expand_dims(qval['ref'][:3], axis=(0, 1))
    high = np.expand_dims(qval['ext'][pt_ind][:3], axis=(0, 1))
    mid = report_utils.compute_avg(low, high, circ_chns)

    others_colour = qval['ffun'](low)

    all_results = []
    attempt_i = 0
    header = 'acc,%s,%s,%s,R,G,B' % (chns_name[0], chns_name[1], chns_name[2])

    th = 0.75 if args.paradigm == '2afc' else 0.625
    while True:
        target_colour = qval['ffun'](mid)
        db_loader = _make_test_loader(args, target_colour, others_colour)

        _, accuracy = common_routines.train_val(
            db_loader, model, None, -1 - attempt_i, args, print_test=False
        )
        print(qname, pt_ind, accuracy, attempt_i, low.squeeze(), mid.squeeze(), high.squeeze())

        all_results.append(np.array([accuracy, *mid.squeeze(), *target_colour.squeeze()]))

        new_low, new_mid, new_high = report_utils.midpoint(
            accuracy, low, mid, high, th=th, circ_chns=circ_chns
        )

        if new_low is None or attempt_i == args.test_attempts:
            print('had to skip')
            break
        else:
            low, mid, high = new_low, new_mid, new_high
        attempt_i += 1
    np.savetxt(output_file, np.array(all_results), delimiter=',', fmt='%f', header=header)
