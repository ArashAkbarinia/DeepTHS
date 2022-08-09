"""
Computing activation of kernels to different set of stimuli.
"""

import sys
import os
import numpy as np
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from .datasets import binary_shapes, dataset_utils
from .models import pretrained_models, model_utils, lesion_utils
from .utils import system_utils, common_routines, argument_handler


def _activation_db(db_loader, model, args, out_file, test_step, print_test=True):
    act_dict, rf_hooks = model_utils.register_model_hooks(
        model, args.architecture, args.transfer_weights[1:]
    )

    all_activations = []
    with torch.set_grad_enabled(False):
        for batch_ind, cu_batch in enumerate(db_loader):
            cu_batch[0] = cu_batch[0].to(next(model.parameters()).device)
            _ = model(cu_batch[0])

            if args.save_all:
                for img_ind in range(cu_batch[0].shape[0]):
                    img_base = cu_batch[1][img_ind][:-4]
                    tmp_acts = dict()
                    for layer_name, layer_act in act_dict.items():
                        current_acts = layer_act[img_ind].clone().cpu().numpy().squeeze()
                        tmp_acts[layer_name] = current_acts
                    save_path = '%s/%s%s.pickle' % (args.output_dir, img_base, out_file)
                    system_utils.write_pickle(save_path, tmp_acts)
            else:
                tmp_acts = dict()
                for layer_name, layer_act in act_dict.items():
                    current_acts = layer_act.clone().cpu().numpy().squeeze()
                    tmp_acts[layer_name] = [
                        np.mean(current_acts, axis=(1, 2)),
                        np.median(current_acts, axis=(1, 2)),
                        np.max(current_acts, axis=(1, 2)),
                    ]
            # acts_rads.append(tmp_acts)

            if batch_ind == 0:
                common_routines.tb_write_images(
                    args.tb_writers['test'], test_step, [cu_batch[0]], *args.preprocess
                )

            # printing the accuracy at certain intervals
            if print_test:
                print('Testing: [{0}/{1}]'.format(batch_ind, len(db_loader)))
            if batch_ind * len(cu_batch[0]) > args.val_samples:
                break

    return None


def _run_colour(args, model):
    test_colours = np.loadtxt(args.test_file, delimiter=',')
    if test_colours.shape[0] == 3:
        test_colours = test_colours.T
    if test_colours.shape[1] != 3:
        sys.exit('Unsupported test file %s with size %s' % (args.test_file, test_colours.shape))

    for colour_ind in range(test_colours.shape[0]):
        transform = dataset_utils.eval_preprocess(args.target_size, args.preprocess)
        colour = test_colours[colour_ind]
        db = binary_shapes.ShapeSingleOut(
            args.data_dir, transform=transform, colour=colour, background=args.background
        )
        db_loader = torch.utils.data.DataLoader(
            db, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True
        )
        out_file = '_colourind%.3d' % colour_ind
        _activation_db(db_loader, model, args, out_file, colour_ind)


def main(argv):
    args = argument_handler.activation_arg_parser(argv)

    args.output_dir = '%s/activations/%s/%s/%s/' % (
        args.output_dir, args.dataset, args.architecture, args.experiment_name
    )
    tb_path = os.path.join(args.output_dir, 'tbs')
    args.tb_writers = {'test': SummaryWriter(tb_path)}

    args.output_dir = os.path.join(args.output_dir, 'acts')
    system_utils.create_dir(args.output_dir)

    args.preprocess = model_utils.get_mean_std(args.colour_space, args.vision_type)

    model = pretrained_models.get_pretrained_model(args.architecture, args.transfer_weights[0])
    model = pretrained_models.get_backbone(args.architecture, model)
    model = lesion_utils.lesion_kernels(
        model, args.lesion_kernels, args.lesion_planes, args.lesion_lines
    )
    model.eval()
    model.cuda()

    # TODO: support different types of experiments
    if args.stimuli == 'grating_radius':
        pass
    elif args.stimuli == 'colour':
        _run_colour(args, model)
