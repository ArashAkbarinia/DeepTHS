"""
PyTorch contrast-discrimination training script for various datasets.
"""

import numpy as np
import time
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from sklearn import svm

from .datasets import dataloader
from .models import model_csf
from .utils import report_utils, system_utils, argument_handler
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


def _train_val(db_loader, model, optimizer, epoch, args):
    # move this to the model itself
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    batch_time = report_utils.AverageMeter()
    data_time = report_utils.AverageMeter()
    losses = report_utils.AverageMeter()
    top1 = report_utils.AverageMeter()

    is_train = optimizer is not None

    if is_train:
        model.train()
        num_samples = args.train_samples
        epoch_type = 'train'
    else:
        model.eval()
        num_samples = args.val_samples
        epoch_type = 'test' if epoch < 0 else 'val'
    tb_writer = args.tb_writers[epoch_type]

    end = time.time()

    all_xs = [] if args.classifier != 'nn' else None
    all_ys = [] if args.classifier != 'nn' else None

    epoch_detail = {'lcontrast': [], 'hcontrast': [], 'ill': [], 'chn': []}
    with torch.set_grad_enabled(is_train and args.classifier == 'nn'):
        for i, data in enumerate(db_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            if args.grating_detector:
                img0, target, _ = data
                img0 = img0.cuda(args.gpu, non_blocking=True)
                output = model(img0)
            else:
                img0, img1, target, img_settings = data
                img0 = img0.cuda(args.gpu, non_blocking=True)
                img1 = img1.cuda(args.gpu, non_blocking=True)
                # compute output
                output = model(img0, img1)

                # FIXME, do this for grating detector as well
                if epoch_type == 'train':
                    for iset in img_settings:
                        contrast0, contrast1, ill, chn = iset
                        epoch_detail['lcontrast'].append(min(contrast0, contrast1))
                        epoch_detail['hcontrast'].append(max(contrast0, contrast1))
                        epoch_detail['ill'].append(ill)
                        epoch_detail['chn'].append(chn)
                if i == 0 and epoch >= -1:
                    img_disp = torch.cat([img0, img1], dim=3)
                    img_inv = report_utils.inv_normalise_tensor(img_disp, args.mean, args.std)
                    for j in range(min(16, img0.shape[0])):
                        if epoch_type == 'test':
                            contrast, sf, angle, phase, _ = img_settings[j]
                            img_name = '%.3d_%.3d_%.3d' % (sf, angle, phase)
                        else:
                            img_name = 'img%03d' % j
                        tb_writer.add_image('{}'.format(img_name), img_inv[j], epoch)

            if all_xs is not None:
                all_xs.append(output.detach().cpu().numpy().copy())
                all_ys.append(target.detach().cpu().numpy().copy())
            else:
                target = target.cuda(args.gpu, non_blocking=True)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1 = report_utils.accuracy(output, target)
                losses.update(loss.item(), img0.size(0))
                top1.update(acc1[0].cpu().numpy()[0], img0.size(0))

                if is_train:
                    # compute gradient and do SGD step
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # printing the accuracy at certain intervals
            if i % args.print_freq == 0:
                print(
                    'Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        epoch, i, len(db_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1
                    )
                )
            if num_samples is not None and i * len(img0) > num_samples:
                break

    # the case of non NN classifier
    if all_xs is not None:
        all_xs = np.concatenate(np.array(all_xs), axis=0)
        all_ys = np.concatenate(np.array(all_ys), axis=0)
        if is_train:
            # currently only supporting svm
            max_iter = 100000
            if args.classifier == 'linear_svm':
                clf = svm.LinearSVC(max_iter=max_iter)
            elif args.classifier == 'svm':
                clf = svm.SVC(max_iter=max_iter)
            clf.fit(all_xs, all_ys)
            system_utils.write_pickle('%s/%s.pickle' % (args.output_dir, args.classifier), clf)
        else:
            clf = system_utils.read_pickle('%s/%s.pickle' % (args.output_dir, args.classifier))
        top1.update(np.mean(clf.predict(all_xs) == all_ys) * 100, len(all_xs))

    if not is_train:
        # printing the accuracy of the epoch
        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))

    if epoch_type == 'train':
        for key in epoch_detail.keys():
            epoch_detail[key] = np.array(epoch_detail[key])
        tb_writer.add_histogram("{}".format('low_contrast'), epoch_detail['lcontrast'], epoch)
        tb_writer.add_histogram("{}".format('high_contrast'), epoch_detail['hcontrast'], epoch)
        tb_writer.add_histogram("{}".format('illuminant'), epoch_detail['ill'], epoch)
        chns_occurance = dict()
        for chn_ind in [-1, 0, 1, 2]:
            chns_occurance['%d' % chn_ind] = np.sum(epoch_detail['chn'] == chn_ind)
        tb_writer.add_scalars("{}".format('chn'), chns_occurance, epoch)

    # writing to tensorboard
    if epoch_type != 'test':
        tb_writer.add_scalar("{}".format('loss'), losses.avg, epoch)
        tb_writer.add_scalar("{}".format('top1'), top1.avg, epoch)
        tb_writer.add_scalar("{}".format('time'), batch_time.avg, epoch)

    return [epoch, batch_time.avg, losses.avg, top1.avg]
