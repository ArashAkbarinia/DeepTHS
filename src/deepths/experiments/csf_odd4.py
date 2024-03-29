"""
Measuring the CSF with 4-choice odd-one-out task.
"""

import numpy as np
import os

import torch
from torch.utils.tensorboard import SummaryWriter

from ..datasets import dataloader_csf
from ..models import model_oddx, model_utils, lesion_utils
from ..utils import system_utils, report_utils, argument_handler
from ..odd_one_out import _train_val


def _luminance_out_gamut(illuminant, contrast):
    # testing whether this illuminant results values in the range of 0-1
    min_diff = contrast - 0.5
    max_diff = 0.5 - contrast
    return illuminant < min(min_diff, max_diff) or illuminant > max(min_diff, max_diff)


def _make_test_loader(args, contrast, l_wave):
    # unique params
    test_thetas = [0, 45, 90, 135]
    test_rhos = [0, 180]
    test_ps = [0]

    db_params = {
        'colour_space': args.colour_space,
        'vision_type': args.vision_type,
        'mask_image': args.mask_image,
        'grating_detector': args.grating_detector
    }
    test_samples = {
        'amp': [contrast], 'lambda_wave': [l_wave], 'theta': test_thetas,
        'rho': test_rhos, 'side': test_ps, 'illuminant': args.illuminant
    }
    db = dataloader_csf.test_set_odd4(
        args.target_size, (args.mean, args.std), test_samples, **db_params
    )
    db.contrast_space = args.contrast_space

    return torch.utils.data.DataLoader(
        db, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True
    )


def _gen_img_name(img_settings, img_ind):
    _, sf, angle, phase, side = img_settings[-1][img_ind]
    return '%.3d_%.3d_%.3d_%.3d' % (sf, side, angle, phase)


def _sensitivity_sf(args, model, l_wave, sf):
    low = 0
    high = 1
    mid = report_utils.compute_avg(low, high)

    res_sf = []
    attempt_i = 0
    psf = {'acc': [], 'contrast': []}

    # th=0.6249 because test samples are 32, 20 correct equals 0.625 and test stops
    th = 0.6249
    while True:
        db_loader = _make_test_loader(args, mid, l_wave)
        _, accuracy = _train_val(
            db_loader, model, None, -1 - attempt_i, args, name_gen_fun=_gen_img_name
        )
        psf['acc'].append(accuracy)
        psf['contrast'].append(int(mid * 1000))
        print(sf, mid, accuracy, low, high)
        res_sf.append(np.array([l_wave, sf, accuracy, mid]))
        new_low, new_mid, new_high = report_utils.midpoint(accuracy, low, mid, high, th=th)
        if new_mid is None or attempt_i == args.test_attempts:
            print('had to skip')
            break
        else:
            low, mid, high = new_low, new_mid, new_high
            if _luminance_out_gamut(args.illuminant, mid):
                print('Ill %.3f not possible for contrast %.3f' % (args.illuminant, mid))
                break
        attempt_i += 1
    return psf, res_sf


def main(argv):
    args = argument_handler.csf_test_arg_parser(argv)
    args.batch_size = 32
    args.workers = 2

    # which illuminant to test
    args.illuminant = 0 if args.illuminant is None else args.illuminant[0]
    ill_suffix = '' if args.illuminant == 0 else '_%d' % int(args.illuminant * 100)

    res_out_dir = os.path.join(args.output_dir, 'evals%s' % ill_suffix)
    out_file = '%s/%s_evolution.csv' % (res_out_dir, args.experiment_name)
    if os.path.exists(out_file):
        return
    system_utils.create_dir(res_out_dir)

    tb_path = os.path.join(args.output_dir, 'test_%s%s' % (args.experiment_name, ill_suffix))
    args.tb_writers = {'test': SummaryWriter(tb_path)}

    args.mean, args.std = model_utils.get_mean_std(args.colour_space, args.vision_type)

    # testing setting
    sf_base = (args.target_size * 0.5) / np.pi
    human_sfs = [i for i in range(1, int(args.target_size / 2) + 1) if args.target_size % i == 0]
    lambda_waves = [sf_base / e for e in human_sfs]

    # creating the model, args.test_net should be a path
    model = model_oddx.oddx_net(args)
    model = lesion_utils.lesion_kernels(
        model, args.lesion_kernels, args.lesion_planes, args.lesion_lines
    )
    model.eval()
    model.cuda()

    header = 'LambdaWave,SF,ACC,Contrast'
    all_results = []
    tb_writer = args.tb_writers['test']
    for i in range(len(lambda_waves)):
        psf_i, res_i = _sensitivity_sf(args, model, lambda_waves[i], human_sfs[i])
        all_results.extend(res_i)
        np.savetxt(out_file, np.array(all_results), delimiter=',', fmt='%f', header=header)
        tb_writer.add_scalar("{}".format('csf'), 1 / all_results[-1][-1], human_sfs[i])

        # making the psf
        psf_i['acc'] = np.array(psf_i['acc'])
        psf_i['contrast'] = np.array(psf_i['contrast'])
        for c in np.argsort(psf_i['contrast']):
            tb_writer.add_scalar(
                "{}_{:03d}".format('psf', human_sfs[i]), psf_i['acc'][c], psf_i['contrast'][c]
            )
    tb_writer.close()
