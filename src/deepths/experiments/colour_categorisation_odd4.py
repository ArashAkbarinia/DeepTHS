"""
PyTorch scripts to train/test colour discrimination.
"""

import os
import sys

import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

from ..datasets.dataloader_colour import triple_colours_odd4
from ..models import model_colour as networks
from ..utils import argument_handler, common_routines, system_utils


def extra_args_fun(parser):
    specific_group = parser.add_argument_group('Experiment specific')

    specific_group.add_argument(
        '--focal_file',
        type=str,
        required=True,
        help='The path to the focal colours.'
    )
    specific_group.add_argument(
        '--test_inds',
        type=None,
        help='Which indices from test file to be tested.'
    )


def main(argv):
    args = argument_handler.master_arg_parser(argv, 'colour_discrimination', extra_args_fun)
    args = common_routines.prepare_starting(args, 'colour_discrimination')
    args.paradigm = 'odd4'
    _main_worker(args)


def _main_worker(args):
    torch.cuda.set_device(args.gpu)
    model = networks.colour_discrimination_net(args)
    model = model.cuda(args.gpu)

    # reading the test file
    test_colours = np.loadtxt(args.test_file, delimiter=',')
    if test_colours.shape[0] == 3:
        test_colours = test_colours.T
    if test_colours.shape[1] != 3:
        sys.exit('Unsupported test file %s with size %s' % (args.test_file, test_colours.shape))
    args.test_colours = test_colours

    # reading the focal file
    focal_colours = np.loadtxt(args.focal_file, delimiter=',')
    if focal_colours.shape[0] == 3:
        focal_colours = focal_colours.T
    if focal_colours.shape[1] != 3:
        sys.exit('Unsupported focal file %s with size %s' % (args.focal_file, focal_colours.shape))
    args.focal_colours = focal_colours

    # defining validation set here so if only test don't do the rest
    if args.validation_dir is None:
        args.validation_dir = args.data_dir + '/validation_set/'

    args.background = 128 if args.background is None else int(args.background)

    if args.test_inds is not None:
        test_inds = np.loadtxt(args.test_inds, dtype='int')
    else:
        test_inds = np.arange(args.test_colours.shape[0])
    for colour_ind in test_inds:
        _predict_i(args, model, colour_ind)
    return


def _make_test_loader(args, test_colour, ref_colours):
    kwargs = {'test_colour': test_colour, 'ref_colours': ref_colours, 'background': args.background}
    db = triple_colours_odd4(args.validation_dir, args.target_size, args.preprocess, **kwargs)

    return torch.utils.data.DataLoader(
        db, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True
    )


def _predict_i(args, model, colour_ind):
    print('Doing colour %.3d' % colour_ind)
    bg_suffix = '_%.3d' % args.background
    res_out_dir = os.path.join(args.output_dir, 'evals_%s%s' % (args.experiment_name, bg_suffix))
    output_file = os.path.join(res_out_dir, 'prediction_%.3d.csv' % colour_ind)
    if os.path.exists(output_file):
        return
    system_utils.create_dir(res_out_dir)
    tb_dir = os.path.join(args.output_dir, 'tests_%s%s' % (args.experiment_name, bg_suffix))
    args.tb_writers = {'test': SummaryWriter(os.path.join(tb_dir, '%.3d' % colour_ind))}

    test_colour = args.test_colours[colour_ind]
    all_results = []
    header = ''
    tb_ind = 1
    for ref_ind1 in range(args.focal_colours.shape[0] - 1):
        for ref_ind2 in range(ref_ind1 + 1, args.focal_colours.shape[0]):
            ref_colours = [args.focal_colours[ref_ind1], args.focal_colours[ref_ind2]]
            db_loader = _make_test_loader(args, test_colour, ref_colours)
            prediction1, _ = common_routines.train_val(
                db_loader, model, None, -tb_ind, args, print_test=False
            )
            all_results.append(prediction1[:, :4].argmax(axis=1))
            tb_ind += 1

            ref_colours = [args.focal_colours[ref_ind2], args.focal_colours[ref_ind1]]
            db_loader = _make_test_loader(args, test_colour, ref_colours)
            prediction2, _ = common_routines.train_val(
                db_loader, model, None, -tb_ind, args, print_test=False
            )
            all_results.append(prediction2[:, :4].argmax(axis=1))
            tb_ind += 1
            header = '%s,r1%dr2%d,r1%dr2%d' % (header, ref_ind1, ref_ind2, ref_ind2, ref_ind1)
    np.savetxt(output_file, np.array(all_results).T, delimiter=',', fmt='%f', header=header)
