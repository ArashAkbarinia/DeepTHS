"""
Computing activation of kernels to different set of stimuli.
"""

import os
import numpy as np
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from skimage import io

import torchvision.transforms as torch_transforms

from .datasets import stimuli_bank, binary_shapes, dataset_utils
from .models import pretrained_models, model_utils, lesion_utils
from .utils import report_utils, system_utils, common_routines, argument_handler


def run_gratings_radius(model, out_file, args):
    act_dict, rfhs = model_utils.resnet_hooks(model)

    max_rad = round(round(args.target_size / 2))

    mean, std = args.preprocess
    transform = torch_transforms.Compose(dataset_utils.post_transform(mean, std))

    all_activations = dict()
    with torch.no_grad():
        for i, (contrast) in enumerate(args.contrasts):
            acts_rads = []
            for grating_radius in range(1, max_rad, args.print_freq):
                img = stimuli_bank.circular_gratings(contrast, grating_radius)
                img = (img + 1) / 2
                # img = _prepapre_colour_space(img, args.colour_space, args.contrast_space)

                # making it pytorch friendly
                img = transform(img)
                img = torch.unsqueeze(img, 0)
                img = img.cuda()

                _ = model(img)
                tmp_acts = dict()
                for key, val in act_dict.items():
                    current_acts = val.clone().cpu().numpy().squeeze()
                    if args.save_all:
                        tmp_acts[key] = current_acts
                    else:
                        tmp_acts[key] = [
                            np.mean(current_acts, axis=(1, 2)),
                            np.median(current_acts, axis=(1, 2)),
                            np.max(current_acts, axis=(1, 2)),
                        ]
                acts_rads.append(tmp_acts)

                if args.visualise:
                    img_inv = report_utils.inv_normalise_tensor(img, mean, std)
                    img_inv = img_inv.detach().cpu().numpy().transpose(0, 2, 3, 1)
                    img_inv = np.concatenate(img_inv, axis=1)
                    save_path = '%s%.5d.png' % (out_file, i)
                    img_inv = np.uint8((img_inv.squeeze() * 255))
                    io.imsave(save_path, img_inv)

                print('Contrast %.2f [%d/%d]' % (contrast, grating_radius, max_rad))
            all_activations['con%.3d' % (contrast * 100)] = acts_rads

    save_path = out_file + '.pickle'
    system_utils.write_pickle(save_path, all_activations)
    return all_activations


def _activation_db(db_loader, model, args, print_test=True):
    ep_helper = common_routines.EpochHelper(args, model, None, -1)
    act_dict, rf_hooks = model_utils.resnet_hooks(ep_helper.model)

    all_activations = []
    end = time.time()

    with torch.set_grad_enabled(ep_helper.grad_status()):
        for batch_ind, cu_batch in enumerate(db_loader):
            # measure data loading time
            ep_helper.log_data_t.update(time.time() - end)

            cu_batch[0] = cu_batch[0].to(next(ep_helper.model.parameters()).device)
            _ = ep_helper.model(cu_batch[0])
            tmp_acts = dict()
            import pdb
            pdb.set_trace()
            for layer_name, layer_act in act_dict.items():
                current_acts = layer_act.clone().cpu().numpy().squeeze()
                if args.save_all:
                    tmp_acts[layer_name] = current_acts
                else:
                    tmp_acts[layer_name] = [
                        np.mean(current_acts, axis=(1, 2)),
                        np.median(current_acts, axis=(1, 2)),
                        np.max(current_acts, axis=(1, 2)),
                    ]
            # acts_rads.append(tmp_acts)

            if batch_ind == 0:
                ep_helper.tb_write_images(cu_batch[:-1], args.mean, args.std)

            # measure elapsed time
            ep_helper.log_batch_t.update(time.time() - end)
            end = time.time()

            # printing the accuracy at certain intervals
            if print_test:
                print('Testing: [{0}/{1}]'.format(batch_ind, len(db_loader)))
            if ep_helper.break_batch(batch_ind, cu_batch[0]):
                break

    ep_helper.finish_epoch()

    return None


def main(argv):
    args = argument_handler.activation_arg_parser(argv)

    res_out_dir = os.path.join(args.output_dir, 'activations_%s' % args.experiment_name)
    system_utils.create_dir(res_out_dir)

    tb_path = os.path.join(args.output_dir, 'act_%s' % args.experiment_name)
    args.tb_writers = {'test': SummaryWriter(tb_path)}

    args.preprocess = model_utils.get_mean_std(args.colour_space, args.vision_type)

    # TODO check it later in the loop
    out_file = '%s/%s' % (args.output_dir, args.experiment_name)
    if os.path.exists(out_file + '.pickle'):
        return

    model = pretrained_models.get_pretrained_model(args.architecture, args.transfer_weights)
    model = pretrained_models.get_backbone(args.architecture, model)
    model = lesion_utils.lesion_kernels(
        model, args.lesion_kernels, args.lesion_planes, args.lesion_lines
    )
    model.eval()
    model.cuda()

    # TODO: support different types of experiments
    if args.stimuli == 'grating_radius':
        _ = run_gratings_radius(model, out_file, args)
    else:
        transform = dataset_utils.eval_preprocess(args.target_size, args.preprocess)
        colour = np.array([1, 1, 0])
        db = binary_shapes.ShapeSingleOut(
            args.data_dir, transform=transform, colour=colour, background=args.background
        )
        db_loader = torch.utils.data.DataLoader(
            db, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True
        )
        _activation_db(db_loader, model, args)
