"""
Computing activation of kernels to different set of stimuli.
"""

import sys
import os
import numpy as np
import glob

import torch
from torch.utils.tensorboard import SummaryWriter

from .datasets import binary_shapes, dataset_utils
from .models import readout, model_utils, lesion_utils
from .utils import system_utils, common_routines, argument_handler


def activation_distance_two_stimuli(db_loader, model, args, test_step, flatten=False,
                                    print_test=False):
    all_distances = []
    with torch.set_grad_enabled(False):
        for batch_ind, cu_batch in enumerate(db_loader):
            out0 = model(cu_batch[0], flatten=flatten)
            out1 = model(cu_batch[1], flatten=flatten)
            for img_ind in range(cu_batch[0].shape[0]):
                all_distances.append(torch.linalg.norm(out0[img_ind] - out1[img_ind]).item())

            if batch_ind == 0:
                common_routines.tb_write_images(
                    args.tb_writers['test'], test_step, [cu_batch[0]], *args.preprocess
                )

            # printing the accuracy at certain intervals
            if print_test:
                print('Testing: [{0}/{1}]'.format(batch_ind, len(db_loader)))
    return all_distances, np.mean(all_distances)


def _activation_db(db_loader, model, args, test_step, print_test=False):
    act_dict, rf_hooks = model_utils.register_model_hooks(
        model.backbone, args.architecture, args.transfer_weights[1:]
    )

    all_activations = []
    with torch.set_grad_enabled(False):
        for batch_ind, cu_batch in enumerate(db_loader):
            _ = model(cu_batch[0])

            for img_ind in range(cu_batch[0].shape[0]):
                img_base = cu_batch[1][img_ind][:-4]
                img_acts = dict()
                for layer_name, layer_act in act_dict.items():
                    current_acts = layer_act[img_ind].clone().cpu().numpy().squeeze()
                    img_acts[layer_name] = current_acts

                if args.save_all is not None:
                    save_path = '%s/%s%s.pickle' % (args.output_dir, img_base, args.save_all)
                    system_utils.write_pickle(save_path, img_acts)
                elif args.ref_dir is not None:
                    ref_files = sorted(glob.glob('%s/%s*.pickle' % (args.ref_dir, img_base)))
                    compare_ref = []
                    for pickle_file in ref_files:
                        ref_res = system_utils.read_pickle(pickle_file)
                        ref_eucs = dict()
                        for layer_name, layer_act in img_acts.items():
                            diff_ref = layer_act - ref_res[layer_name]
                            axis = tuple(np.arange(1, len(diff_ref.shape)))
                            euc_dist = np.sum(diff_ref ** 2, axis=axis) ** 0.5
                            ref_eucs[layer_name] = euc_dist
                        compare_ref.append(ref_eucs)
                    all_activations.append([compare_ref, img_base])

                # img_acts[layer_name] = [
                #     np.mean(current_acts, axis=(1, 2)),
                #     np.median(current_acts, axis=(1, 2)),
                #     np.max(current_acts, axis=(1, 2)),
                # ]

            if batch_ind == 0:
                common_routines.tb_write_images(
                    args.tb_writers['test'], test_step, [cu_batch[0]], *args.preprocess
                )

            # printing the accuracy at certain intervals
            if print_test:
                print('Testing: [{0}/{1}]'.format(batch_ind, len(db_loader)))
            if batch_ind * len(cu_batch[0]) > args.val_samples:
                break
    return all_activations


def _run_colour(args, model):
    test_colours = np.loadtxt(args.test_file, delimiter=',')
    if test_colours.shape[0] == 3:
        test_colours = test_colours.T
    if test_colours.shape[1] != 3:
        sys.exit('Unsupported test file %s with size %s' % (args.test_file, test_colours.shape))

    for colour_ind in range(test_colours.shape[0]):
        print('Performing colour %.3d' % colour_ind)
        args.save_all = '_colourind%.3d' % colour_ind if args.save_all else None
        if args.save_all is None:
            save_path = '%s/colour_ind%.3d.pickle' % (args.output_dir, colour_ind)
            if os.path.exists(save_path):
                continue

        transform = dataset_utils.eval_preprocess(args.target_size, args.preprocess)
        colour = test_colours[colour_ind]
        db = binary_shapes.ShapeSingleOut(
            args.data_dir, transform=transform, colour=colour, background=args.background
        )
        db_loader = torch.utils.data.DataLoader(
            db, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True
        )
        all_activations = _activation_db(db_loader, model, args, colour_ind)
        if args.save_all is None:
            system_utils.write_pickle(save_path, all_activations)


def main(argv):
    args = argument_handler.activation_arg_parser(argv)
    args.preprocess = model_utils.get_mean_std(args.colour_space, args.vision_type)

    args.output_dir = '%s/activations/%s/%s/%s/' % (
        args.output_dir, args.dataset, args.architecture, args.experiment_name
    )
    system_utils.create_dir(args.output_dir)
    system_utils.save_arguments(args)

    args.background = 128 if args.background is None else int(args.background)
    args.val_samples = np.inf if args.val_samples is None else args.val_samples

    tb_path = os.path.join(args.output_dir, 'tbs_%.3d' % args.background)
    args.tb_writers = {'test': SummaryWriter(tb_path)}

    args.output_dir = os.path.join(args.output_dir, 'acts_%.3d' % args.background)
    system_utils.create_dir(args.output_dir)

    model = readout.ActivationLoader(args.architecture, args.transfer_weights[0])
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
