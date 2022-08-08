"""
Computing activation of kernels to different set of stimuli.
"""

import numpy as np
import os

import torch
from torch.utils.tensorboard import SummaryWriter

from skimage import io

import torchvision.transforms as torch_transforms

from .datasets import stimuli_bank, cv2_transforms
from .models import readout, model_utils, lesion_utils
from .utils import report_utils, system_utils, argument_handler
from . import colour_discrimination


def run_gratings_radius(model, out_file, args):
    act_dict, rfhs = model_utils.resnet_hooks(model)

    max_rad = round(round(args.target_size / 2))

    mean, std = args.preprocess
    transform = torch_transforms.Compose([
        cv2_transforms.ToTensor(),
        cv2_transforms.Normalize(mean, std),
    ])

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


def main(argv):
    args = argument_handler.activation_arg_parser(argv)

    res_out_dir = os.path.join(args.output_dir, 'activations_%s' % args.experiment_name)
    system_utils.create_dir(res_out_dir)

    tb_path = os.path.join(args.output_dir, 'act_%s' % args.experiment_name)
    args.tb_writers = {'test': SummaryWriter(tb_path)}

    args.mean, args.std = model_utils.get_mean_std(args.colour_space, args.vision_type)

    # TODO check it later in the loop
    out_file = '%s/%s' % (args.output_dir, args.experiment_name)
    if os.path.exists(out_file + '.pickle'):
        return

    model = readout.ReadOutNet(args.architecture, args.target_size, args.transfer_weights)
    model = lesion_utils.lesion_kernels(
        model, args.lesion_kernels, args.lesion_planes, args.lesion_lines
    )
    model.eval()
    model.cuda()

    # TODO: support different types of experiments
    if args.stimuli == 'grating_radius':
        _ = run_gratings_radius(model, out_file, args)
    else:
        pass
        # colour_discrimination._train_val(db_loader, model, None, -1, args, False)
