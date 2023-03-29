"""

"""

import os
import sys
import time

import numpy as np
import collections

import torch
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from sklearn import svm

from . import system_utils, report_utils
from ..models import model_utils


def _prepare_training(args, model):
    optimizer = _make_optimizer(args, model)
    milestones = [int(e * args.epochs) for e in [5, 8]]
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    initial_epoch = args.initial_epoch
    model_progress = []

    # optionally resume from a checkpoint
    if args.resume is not None:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location='cpu')
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

            initial_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            model = model.cuda(args.gpu)

            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])

            if os.path.exists(args.prog_path):
                model_progress = np.loadtxt(args.prog_path, delimiter=',')
                model_progress = model_progress.tolist()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    return model, optimizer, scheduler, initial_epoch, model_progress


def _make_optimizer(args, model):
    if args.classifier == 'nn':
        # if transfer_weights, only train the fc layer, otherwise all parameters
        if args.transfer_weights[0] is None or args.transfer_weights[0] == 'none':
            params_to_optimize = [{'params': [p for p in model.parameters()]}]
        else:
            for p in model.backbone.parameters():
                p.requires_grad = False
            params_to_optimize = [{'params': [p for p in model.fc.parameters()]}]
        # optimiser
        optimizer = torch.optim.SGD(
            params_to_optimize, lr=args.learning_rate,
            momentum=args.momentum, weight_decay=args.weight_decay
        )
    else:
        optimizer = []
    return optimizer


def prepare_starting(args, task_folder):
    system_utils.set_random_environment(args.random_seed)

    args.mean, args.std = model_utils.get_mean_std(args.colour_space, args.vision_type)
    args.preprocess = (args.mean, args.std)

    if args.test_net:
        return args

    if args.classifier != 'nn':
        args.epochs = 1
        args.print_freq = np.inf

    # preparing the output folder
    layer = args.transfer_weights[1] if len(args.transfer_weights) == 2 else 'mix'
    args.output_dir = '%s/%s/%s/%s/%s/%s/' % (
        args.output_dir, task_folder, args.dataset, args.architecture, args.experiment_name, layer
    )
    system_utils.create_dir(args.output_dir)

    # dumping all passed arguments to a json file
    system_utils.save_arguments(args)
    return args


def do_epochs(args, epoch_fun, train_loader, val_loader, model):
    args.prog_path = os.path.join(args.output_dir, 'model_progress.csv')
    model, optimizer, scheduler, args.initial_epoch, model_prog = _prepare_training(args, model)

    # create the tensorboard writers
    args.tb_writers = dict()
    for mode in ['train', 'val']:
        args.tb_writers[mode] = SummaryWriter(os.path.join(args.output_dir, mode))

    # TODO: for resume this is not correct, make the epoch helper here
    best_acc1 = 0
    # training on epoch
    for epoch in range(args.initial_epoch, args.epochs):
        # train for one epoch
        train_log = epoch_fun(train_loader, model, optimizer, epoch, args)

        # evaluate on validation set
        validation_log = epoch_fun(val_loader, model, None, epoch, args)

        model_prog.append([*train_log, *validation_log[1:]])

        # remember best acc@1 and save checkpoint
        acc1 = validation_log[2]
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if args.classifier == 'nn':
            scheduler.step()

        # save the checkpoints
        system_utils.save_checkpoint(
            {
                'epoch': epoch,
                'arch': args.architecture,
                'net': {'classifier': args.classifier, 'pooling': args.pooling,
                        'extra': args.net_params},
                'transfer_weights': args.transfer_weights,
                'preprocessing': {'mean': args.mean, 'std': args.std},
                'state_dict': _extract_altered_state_dict(model, args),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict() if args.classifier == 'nn' else [],
                'scheduler': scheduler.state_dict() if args.classifier == 'nn' else [],
                'target_size': args.target_size,
            },
            is_best, args
        )
        header = 'epoch,t_time,t_loss,t_top1,v_time,v_loss,v_top1'
        np.savetxt(args.prog_path, np.array(model_prog), delimiter=',', header=header)

    # closing the tensorboard writers
    for mode in args.tb_writers.keys():
        args.tb_writers[mode].close()


def _extract_altered_state_dict(model, args):
    if args.transfer_weights[0] == 'none':
        return model.state_dict()
    altered_state_dict = collections.OrderedDict()
    for key, _ in model.named_buffers():
        altered_state_dict[key] = model.state_dict()[key]
    if args.classifier == 'nn':
        for key in ['fc.weight', 'fc.bias']:
            altered_state_dict[key] = model.state_dict()[key]
    return altered_state_dict


class EpochHelper:
    def __init__(self, args, model, optimizer, epoch):
        self.log_batch_t = report_utils.AverageMeter()
        self.log_data_t = report_utils.AverageMeter()
        self.log_loss = report_utils.AverageMeter()
        self.log_acc = report_utils.AverageMeter()

        self.epoch = epoch
        self.model = model
        self.optimizer = optimizer
        self.classifier = args.classifier
        self.output_dir = args.output_dir

        self.is_train = optimizer is not None
        self.is_test = self.epoch < 0

        if self.is_train:
            self.model.train()
            if args.transfer_weights[0] not in [None, 'none']:
                self.model.backbone.eval()
            self.num_samples = args.train_samples
            self.epoch_type = 'train'
        else:
            self.model.eval()
            self.num_samples = args.val_samples
            self.epoch_type = 'test' if self.is_test else 'val'
        self.num_samples = np.inf if self.num_samples is None else self.num_samples
        self.tb_writer = args.tb_writers[self.epoch_type]

        self.all_xs = [] if self.classifier != 'nn' else None
        self.all_ys = [] if self.classifier != 'nn' else None

    def update_epoch(self, output, target, target_acc, batch, criterion):
        if self.all_xs is not None:
            self.all_xs.append(output.detach().cpu().numpy().copy())
            self.all_ys.append(target.detach().cpu().numpy().copy())
        else:
            # measure accuracy and record loss
            loss = criterion(output, target)
            self.log_loss.update(loss.item(), batch.size(0))
            acc1 = report_utils.accuracy(output, target_acc)
            self.log_acc.update(acc1[0].cpu().numpy()[0], batch.size(0))

            if self.is_train:
                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def grad_status(self):
        return self.is_train and self.classifier == 'nn'

    def break_batch(self, b_ind, batch):
        return not self.is_test and b_ind * len(batch) > self.num_samples

    def finish_epoch(self):
        # the case of non NN classifier
        if self.all_xs is not None:
            self.train_non_nn()

        if not self.is_train:
            # printing the accuracy of the epoch
            print(' * Acc@1 {top1.avg:.3f}'.format(top1=self.log_acc))
        print()

        # writing to tensorboard
        if self.epoch_type != 'test':
            self.tb_writer.add_scalar("{}".format('loss'), self.log_loss.avg, self.epoch)
            self.tb_writer.add_scalar("{}".format('top1'), self.log_acc.avg, self.epoch)
            self.tb_writer.add_scalar("{}".format('time'), self.log_batch_t.avg, self.epoch)

    def train_non_nn(self):
        all_xs = np.concatenate(np.array(self.all_xs), axis=0)
        all_ys = np.concatenate(np.array(self.all_ys), axis=0)
        pickle_path = '%s/%s.pickle' % (self.output_dir, self.classifier)
        if self.is_train:
            # currently only supporting svm
            max_iter = 100000
            if self.classifier == 'linear_svm':
                clf = svm.LinearSVC(max_iter=max_iter)
            elif self.classifier == 'svm':
                clf = svm.SVC(max_iter=max_iter)
            else:
                sys.exit('Unsupported type of linear classifier %s' % self.classifier)
            clf.fit(all_xs, all_ys)
            system_utils.write_pickle(pickle_path, clf)
        else:
            clf = system_utils.read_pickle(pickle_path)
        return self.log_acc.update(np.mean(clf.predict(all_xs) == all_ys) * 100, len(all_xs))

    def print_epoch(self, db_loader, batch_ind, end="\n"):
        print(
            '{0}: [{1}][{2}/{3}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                self.epoch_type, self.epoch, batch_ind, len(db_loader), batch_time=self.log_batch_t,
                data_time=self.log_data_t, loss=self.log_loss, top1=self.log_acc
            ), end=end
        )

    def tb_write_images(self, cu_batch, mean, std, name_gen=None):
        step = self.epoch if self.epoch >= 0 else -self.epoch - 1
        tb_write_images(self.tb_writer, step, cu_batch, mean, std, name_gen)


def tb_write_images(tb_writer, step, cu_batch, mean, std, name_gen=None):
    img_disp = torch.cat([*cu_batch], dim=3)
    img_inv = report_utils.inv_normalise_tensor(img_disp, mean, std)
    for j in range(min(16, cu_batch[0].shape[0])):
        img_name = name_gen(j) if name_gen is not None else 'img%03d' % j
        tb_writer.add_image('{}'.format(img_name), img_inv[j], step)


def train_val(db_loader, model, optimizer, epoch, args, print_test=True, name_gen=None):
    ep_helper = EpochHelper(args, model, optimizer, epoch)
    criterion = ep_helper.model.loss_function

    # FIXME ALL_TARGETS needs to be defined to have the same oerder as prediciton.
    all_predictions = []
    end = time.time()
    with torch.set_grad_enabled(ep_helper.grad_status()):
        for batch_ind, cu_batch in enumerate(db_loader):
            # measure data loading time
            ep_helper.log_data_t.update(time.time() - end)

            if args.paradigm == '2afc':
                target = cu_batch[-1].unsqueeze(dim=1).float()
                odd_ind = target
            else:
                odd_ind = cu_batch[-1]
                # preparing the target
                target = torch.zeros(odd_ind.shape[0], 4)
                target[torch.arange(odd_ind.shape[0]), cu_batch[-1]] = 1
            odd_ind = odd_ind.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            output = ep_helper.model(*cu_batch[:-1])

            if batch_ind == 0:
                ep_helper.tb_write_images(cu_batch[:-1], args.mean, args.std, name_gen)

            ep_helper.update_epoch(output, target, odd_ind, cu_batch[0], criterion)

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
                ep_helper.print_epoch(db_loader, batch_ind)
            if ep_helper.break_batch(batch_ind, cu_batch[0]):
                break

    ep_helper.finish_epoch()

    if len(all_predictions) == 1:
        prediction_output = np.concatenate(all_predictions[0])
    else:
        prediction_output = [np.concatenate(out) for out in all_predictions]
    if ep_helper.is_test:
        accuracy = ep_helper.log_acc.avg / 100
        return prediction_output, accuracy
    return [epoch, ep_helper.log_batch_t.avg, ep_helper.log_loss.avg, ep_helper.log_acc.avg]
