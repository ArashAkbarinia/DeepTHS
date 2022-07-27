"""
PyTorch contrast-discrimination training script for various datasets.
"""

import time
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from .datasets import dataloader
from .models import model_csf
from .utils import argument_handler
from .utils import common_routines


def main(argv):
    args = argument_handler.csf_train_arg_parser(argv)
    args = common_routines.prepare_starting(args, 'csf')
    _main_worker(args)


def _main_worker(args):
    # create model
    net_t = model_csf.GratingDetector if args.grating_detector else model_csf.ContrastDiscrimination
    model = net_t(args.architecture, args.target_size, args.transfer_weights, args.classifier)

    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)

    train_trans = []
    valid_trans = []
    both_trans = []

    # to have 100% deterministic behaviour train_params must be passed
    if args.train_params is not None:
        args.workers = 0
        shuffle = False
        args.illuminant = 0.0
    else:
        shuffle = True

    if args.sf_filter is not None and len(args.sf_filter) != 2:
        sys.exit('Length of the sf_filter must be two %s' % args.sf_filter)

    # loading the training set
    train_trans = [*both_trans, *train_trans]
    db_params = {
        'colour_space': args.colour_space,
        'vision_type': args.vision_type,
        'mask_image': args.mask_image,
        'contrasts': args.contrasts,
        'illuminant': args.illuminant,
        'train_params': args.train_params,
        'sf_filter': args.sf_filter,
        'contrast_space': args.contrast_space,
        'same_transforms': args.same_transforms,
        'grating_detector': args.grating_detector
    }
    if args.dataset in dataloader.NATURAL_DATASETS:
        path_or_sample = args.data_dir
    else:
        # this would be only for the grating dataset to generate
        path_or_sample = args.train_samples
    train_dataset = dataloader.train_set(
        args.dataset, args.target_size, preprocess=(args.mean, args.std),
        extra_transformation=train_trans, data_dir=path_or_sample, **db_params
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=shuffle,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=True
    )

    # validation always is random
    db_params['train_params'] = None
    # loading validation set
    valid_trans = [*both_trans, *valid_trans]
    validation_dataset = dataloader.validation_set(
        args.dataset, args.target_size, preprocess=(args.mean, args.std),
        extra_transformation=valid_trans, data_dir=path_or_sample, **db_params
    )

    val_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True
    )

    common_routines.do_epochs(args, _train_val, train_loader, val_loader, model)


def _gen_img_name(img_settings, img_ind):
    _, sf, angle, phase, _ = img_settings[img_ind]
    return '%.3d_%.3d_%.3d' % (sf, angle, phase)


def _train_val(db_loader, model, optimizer, epoch, args):
    # move this to the model itself
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    ep_helper = common_routines.EpochHelper(args, model, optimizer, epoch)

    end = time.time()

    with torch.set_grad_enabled(ep_helper.grad_status()):
        for batch_ind, cu_batch in enumerate(db_loader):
            # measure data loading time
            ep_helper.log_data_t.update(time.time() - end)

            if args.grating_detector:
                img0, target, img_settings = cu_batch
                img0 = img0.cuda(args.gpu, non_blocking=True)
                output = ep_helper.model(img0)
            else:
                img0, img1, target, img_settings = cu_batch
                img0 = img0.cuda(args.gpu, non_blocking=True)
                img1 = img1.cuda(args.gpu, non_blocking=True)
                # compute output
                output = ep_helper.model(img0, img1)

            if batch_ind == 0 and epoch >= -1:
                def name_gen(x): return _gen_img_name(cu_batch[-1], x)

                name_gen_f = name_gen if ep_helper.is_test else None
                ep_helper.tb_write_images(cu_batch[:-2], args.mean, args.std, name_gen_f)

            if ep_helper.all_xs is not None:
                ep_helper.all_xs.append(output.detach().cpu().numpy().copy())
                ep_helper.all_ys.append(target.detach().cpu().numpy().copy())
            else:
                target = target.cuda(args.gpu, non_blocking=True)
                loss = criterion(output, target)

                ep_helper.update_epoch(loss, output, target, img0)

            # measure elapsed time
            ep_helper.log_batch_t.update(time.time() - end)
            end = time.time()

            # printing the accuracy at certain intervals
            if batch_ind % args.print_freq == 0:
                ep_helper.print_epoch(db_loader, batch_ind)
            if ep_helper.break_batch(batch_ind, img0):
                break

    ep_helper.finish_epoch()

    return [epoch, ep_helper.log_batch_t.avg, ep_helper.log_loss.avg, ep_helper.log_acc.avg]
