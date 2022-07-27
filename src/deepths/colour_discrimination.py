"""
PyTorch scripts to train/test colour discrimination.
"""

import os
import numpy as np
import time

import torch

from .datasets import dataloader_colour
from .models import model_colour as networks
from .utils import report_utils, argument_handler, colour_spaces
from .utils import common_routines


def main(argv):
    args = argument_handler.colour_discrimination_arg_parser(argv)
    args = common_routines.prepare_starting(args, 'colour_discrimination')
    _main_worker(args)


def _main_worker(args):
    model = networks.colour_discrimination_net(
        args.paradigm, args.test_net, args.architecture, args.target_size,
        args.transfer_weights, args.classifier
    )

    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)

    # setting the quadrant points
    if args.pts_path is None:
        args.pts_path = args.validation_dir + '/rgb_points.csv'
    test_pts = np.loadtxt(args.pts_path, delimiter=',', dtype=str)

    args.test_pts = _organise_test_points(test_pts)

    # defining validation set here so if only test don't do the rest
    if args.validation_dir is None:
        args.validation_dir = args.data_dir + '/validation_set/'

    if args.test_net:
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

    common_routines.do_epochs(args, _train_val, train_loader, val_loader, model)


def _organise_test_points(test_pts):
    out_test_pts = dict()
    for test_pt in test_pts:
        pt_val = test_pt[:3].astype('float')
        test_pt_name = test_pt[-2]
        if 'ref_' == test_pt_name[:4]:
            test_pt_name = test_pt_name[4:]
            if test_pt[-1] == 'dkl':
                ffun = colour_spaces.dkl2rgb01
                bfun = colour_spaces.rgb012dkl
                chns_name = ['D', 'K', 'L']
            elif test_pt[-1] == 'hsv':
                ffun = colour_spaces.hsv012rgb01
                bfun = colour_spaces.rgb2hsv01
                chns_name = ['H', 'S', 'V']
            elif test_pt[-1] == 'rgb':
                ffun = lambda x: x
                bfun = lambda x: x
                chns_name = ['R', 'G', 'B']
            out_test_pts[test_pt_name] = {
                'ref': pt_val, 'ffun': ffun, 'bfun': bfun, 'space': chns_name, 'ext': [], 'chns': []
            }
        else:
            out_test_pts[test_pt_name]['ext'].append(pt_val)
            out_test_pts[test_pt_name]['chns'].append(test_pt[-1])
    return out_test_pts


def _train_val(db_loader, model, optimizer, epoch, args, print_test=True):
    ep_helper = common_routines.EpochHelper(args, model, optimizer, epoch)

    all_predictions = []
    end = time.time()

    with torch.set_grad_enabled(ep_helper.grad_status()):
        for batch_ind, cu_batch in enumerate(db_loader):
            # measure data loading time
            ep_helper.log_data_t.update(time.time() - end)

            if args.paradigm == '2afc':
                (img0, img1, target) = cu_batch
                img0 = img0.cuda(args.gpu, non_blocking=True)
                img1 = img1.cuda(args.gpu, non_blocking=True)
                target = target.unsqueeze(dim=1).float()
            else:
                (img0, img1, img2, img3, odd_ind) = cu_batch
                img0 = img0.cuda(args.gpu, non_blocking=True)
                img1 = img1.cuda(args.gpu, non_blocking=True)
                img2 = img2.cuda(args.gpu, non_blocking=True)
                img3 = img3.cuda(args.gpu, non_blocking=True)

                # preparing the target
                target = torch.zeros(odd_ind.shape[0], 4)
                target[torch.arange(odd_ind.shape[0]), odd_ind] = 1
                odd_ind = odd_ind.cuda(args.gpu, non_blocking=True)

            if ep_helper.all_xs is not None:
                ep_helper.all_xs.append(output.detach().cpu().numpy().copy())
                ep_helper.all_ys.append(target.detach().cpu().numpy().copy())
            else:
                target = target.cuda(args.gpu, non_blocking=True)

                # compute output
                if args.paradigm == '2afc':
                    output = ep_helper.model(img0, img1)
                    odd_ind = target
                else:
                    output = ep_helper.model(img0, img1, img2, img3)
                loss = ep_helper.model.loss_function(output, target)

                ep_helper.update_epoch(loss, output, odd_ind, img0)

            # measure elapsed time
            ep_helper.log_batch_t.update(time.time() - end)
            end = time.time()

            # to use for correlations
            if args.paradigm == '2afc':
                pred_outs = np.concatenate(
                    [output.detach().cpu().numpy(), odd_ind.cpu().numpy()], axis=1
                )
            else:
                pred_outs = np.concatenate(
                    [output.detach().cpu().numpy(), odd_ind.unsqueeze(dim=1).cpu().numpy()], axis=1
                )
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
                ep_helper.print_epoch(epoch, db_loader, batch_ind)
            if ep_helper.break_batch(batch_ind, img0):
                break

    ep_helper.finish_epoch()

    if len(all_predictions) == 1:
        prediction_output = np.concatenate(all_predictions[0])
    else:
        prediction_output = [np.concatenate(out) for out in all_predictions]
    if ep_helper.is_test:
        accuracy = ep_helper.log_acc.avg if ep_helper.log_acc.avg <= 1.0 else ep_helper.log_acc.avg / 100
        return prediction_output, accuracy
    return [epoch, ep_helper.log_batch_t.avg, ep_helper.log_loss.avg, ep_helper.log_acc.avg]


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
        output_file = os.path.join(args.output_dir, 'accuracy_%s.csv' % (qname))
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

    _, accuracy = _train_val(db_loader, model, None, -1, args, print_test=False)
    print(qname, pt_ind, accuracy, low.squeeze(), high.squeeze())
    return accuracy


def _sensitivity_test_point(args, model, qname, pt_ind):
    qval = args.test_pts[qname]
    chns_name = qval['space']
    circ_chns = [0] if chns_name[0] == 'H' else []
    output_file = os.path.join(args.output_dir, 'evolutoin_%s_%d.csv' % (qname, pt_ind))
    if os.path.exists((output_file)):
        return

    low = np.expand_dims(qval['ref'][:3], axis=(0, 1))
    high = np.expand_dims(qval['ext'][pt_ind][:3], axis=(0, 1))
    mid = report_utils.compute_avg(low, high, circ_chns)

    others_colour = qval['ffun'](low)

    all_results = []
    j = 0
    header = 'acc,%s,%s,%s,R,G,B' % (chns_name[0], chns_name[1], chns_name[2])

    th = 0.75 if args.paradigm == '2afc' else 0.625
    while True:
        target_colour = qval['ffun'](mid)
        db_loader = _make_test_loader(args, target_colour, others_colour)

        _, accuracy = _train_val(db_loader, model, None, -1, args, print_test=False)
        print(qname, pt_ind, accuracy, j, low.squeeze(), mid.squeeze(), high.squeeze())

        all_results.append(np.array([accuracy, *mid.squeeze(), *target_colour.squeeze()]))
        np.savetxt(output_file, np.array(all_results), delimiter=',', fmt='%f', header=header)

        new_low, new_mid, new_high = report_utils.midpoint(
            accuracy, low, mid, high, th=th, circ_chns=circ_chns
        )

        if new_low is None or j == args.test_attempts:
            print('had to skip')
            break
        else:
            low, mid, high = new_low, new_mid, new_high
        j += 1
