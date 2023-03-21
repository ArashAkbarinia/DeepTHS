"""
PyTorch scripts to train/test the odd-one-out task.
"""

import numpy as np
import sys
import os
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from .datasets import dataloader_oddx
from .models import model_oddx
from .utils import argument_handler
from .utils import common_routines, report_utils, system_utils


def main(argv):
    args = argument_handler.master_arg_parser(argv, 'odd_one_out')
    # FIXME args.paradigm
    if args.features_path is not None:
        if os.path.isfile(args.features_path):
            args.train_kwargs = system_utils.read_json(args.features_path)
            if 'single_img' in args.train_kwargs:
                args.single_img = args.train_kwargs['single_img']
        else:
            sys.exit('%s does not exist!' % args.features_path)
    else:
        args.train_kwargs = {
            'features': dataloader_oddx.FEATURES,
            'single_img': args.single_img
        }
    args = common_routines.prepare_starting(args, 'odd_one_out')
    _main_worker(args)


def _main_worker(args):
    torch.cuda.set_device(args.gpu)
    model = model_oddx.oddx_net(args, args.train_kwargs)
    model = model.cuda(args.gpu)

    # loading the training set
    train_dataset = dataloader_oddx.oddx_bg_folder(
        args.background, args.paradigm, args.target_size, args.preprocess, **args.train_kwargs
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None
    )

    common_routines.do_epochs(args, _train_val, train_loader, train_loader, model)


def _gen_img_name(gt_settings, img_ind):
    odd_ind, odd_class = gt_settings
    return 'gt_%.3d_%.3d' % (odd_class[img_ind], odd_ind[img_ind])


def _train_val(db_loader, model, optimizer, epoch, args, print_test=True):
    ep_helper = common_routines.EpochHelper(args, model, optimizer, epoch)
    log_acc_class = report_utils.AverageMeter()
    log_loss_ind = report_utils.AverageMeter()
    log_loss_class = report_utils.AverageMeter()
    criterion = ep_helper.model.loss_function

    all_predictions = []
    end = time.time()

    with torch.set_grad_enabled(ep_helper.grad_status()):
        for batch_ind, cu_batch in enumerate(db_loader):
            # measure data loading time
            ep_helper.log_data_t.update(time.time() - end)

            input_signal = cu_batch[:-2]
            # preparing the target
            odd_class = cu_batch[-1]
            odd_ind = cu_batch[-2]
            if len(input_signal) > 1:
                odd_ind_arr = torch.zeros(odd_ind.shape[0], len(input_signal))
                odd_ind_arr[torch.arange(odd_ind.shape[0]), odd_ind] = 1
            else:
                odd_ind_arr = odd_ind
            # moving them to CUDA
            odd_class = odd_class.cuda(args.gpu, non_blocking=True)
            odd_ind = odd_ind.cuda(args.gpu, non_blocking=True)
            odd_ind_arr = odd_ind_arr.cuda(args.gpu, non_blocking=True)
            output = ep_helper.model(*input_signal)

            if batch_ind == 0:
                def name_gen(x): return _gen_img_name(cu_batch[-2:], x)

                ep_helper.tb_write_images(input_signal, args.mean, args.std, name_gen)

            target = (odd_ind_arr, odd_class)

            ##
            loss_ind, loss_class = criterion(output, target)
            log_loss_ind.update(loss_ind.item(), cu_batch[0].size(0))
            loss = loss_ind if output[1] is None else 0.5 * loss_ind + 0.5 * loss_class
            ep_helper.log_loss.update(loss.item(), cu_batch[0].size(0))

            # measure accuracy and record loss
            acc_ind = report_utils.accuracy(output[0], odd_ind)
            ep_helper.log_acc.update(acc_ind[0].cpu().numpy()[0], cu_batch[0].size(0))
            if output[1] is not None:
                log_loss_class.update(loss_class.item(), cu_batch[0].size(0))
                acc_class = report_utils.accuracy(output[1], odd_class)
                log_acc_class.update(acc_class[0].cpu().numpy()[0], cu_batch[0].size(0))

            if ep_helper.is_train:
                # compute gradient and do SGD step
                ep_helper.optimizer.zero_grad()
                loss.backward()
                ep_helper.optimizer.step()
            ##

            # measure elapsed time
            ep_helper.log_batch_t.update(time.time() - end)
            end = time.time()

            # to use for correlations
            to_concatenate = [
                output[0].detach().cpu().numpy(),
                odd_ind.unsqueeze(dim=1).cpu().numpy(),
            ]
            if output[1] is not None:
                to_concatenate.append(output[1].detach().cpu().numpy())
                to_concatenate.append(odd_class.unsqueeze(dim=1).cpu().numpy())
            pred_outs = np.concatenate(to_concatenate, axis=1)
            # I'm not sure if this is all necessary, copied from keras
            if not isinstance(pred_outs, list):
                pred_outs = [pred_outs]

            if not all_predictions:
                for _ in pred_outs:
                    all_predictions.append([])

            for j, out in enumerate(pred_outs):
                all_predictions[j].append(out)

            # printing the accuracy at certain intervals
            if ep_helper.is_test and print_test:
                print('Testing: [{0}/{1}]'.format(batch_ind, len(db_loader)))
            elif batch_ind % args.print_freq == 0:
                ep_helper.print_epoch(db_loader, batch_ind, end="\t")
                print(
                    'Acc@Class {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Loss@Ind {loss_ind.val:.3f} ({loss_ind.avg:.3f})\t'
                    'Loss@Class {loss_class.val:.3f} ({loss_class.avg:.3f})\t'.format(
                        top1=log_acc_class, loss_ind=log_loss_ind, loss_class=log_loss_class,
                    )
                )
            if ep_helper.break_batch(batch_ind, cu_batch[0]):
                break

    ep_helper.finish_epoch()
    ep_helper.tb_writer.add_scalar("{}".format('loss_ind'), log_loss_ind.avg, epoch)
    ep_helper.tb_writer.add_scalar("{}".format('loss_class'), log_loss_class.avg, epoch)
    ep_helper.tb_writer.add_scalar("{}".format('acc_class'), log_acc_class.avg, epoch)

    if len(all_predictions) == 1:
        prediction_output = np.concatenate(all_predictions[0])
    else:
        prediction_output = [np.concatenate(out) for out in all_predictions]
    if ep_helper.is_test:
        accuracy = ep_helper.log_acc.avg / 100
        return prediction_output, accuracy
    return [epoch, ep_helper.log_batch_t.avg, ep_helper.log_loss.avg, ep_helper.log_acc.avg]
