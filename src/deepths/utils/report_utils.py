"""
Utility routines for logging and reporting.
"""

import re
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy_preds(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        corrects = []
        for k in topk:
            corrects.append(correct[:k])
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res, corrects


def accuracy(output, target, topk=(1,)):
    if output.shape[1] == 1:
        acc = (output > 0) == target
        acc = [acc.float().mean(0, keepdim=True)[0]]
        return acc
    res, _ = accuracy_preds(output, target, topk=topk)
    return res


def inv_normalise_tensor(tensor, mean, std):
    tensor = tensor.clone()
    if type(mean) not in [tuple, list]:
        mean = tuple([mean for _ in range(tensor.shape[1])])
    if type(std) not in [tuple, list]:
        std = tuple([std for _ in range(tensor.shape[1])])
    # inverting the normalisation for each channel
    for i in range(tensor.shape[1]):
        tensor[:, i, ] = (tensor[:, i, ] * std[i]) + mean[i]
    tensor = tensor.clamp(0, 1)
    return tensor


def atof(value):
    try:
        return float(value)
    except ValueError:
        return value


def atoi(value):
    try:
        return int(value)
    except ValueError:
        return value


def natural_keys(text, delimiter=None, remove=None):
    """
    alist.sort(key=natural_keys) sorts in human order
    adapted from http://nedbatchelder.com/blog/200712/human_sorting.html
    """
    if remove is not None:
        text = text.replace(remove, '')
    if delimiter is None:
        return [atoi(c) for c in re.split(r'(\d+)', text)]
    else:
        return [atof(c) for c in text.split(delimiter)]


def min_max_normalise(x, low=0, high=1, minv=None, maxv=None):
    if minv is None:
        minv = x.min()
    if maxv is None:
        maxv = x.max()
    output = low + (x - minv) * (high - low) / (maxv - minv)
    return output


def mid_point(a, b, circular_chns=None):
    if circular_chns is None:
        circular_chns = []
    c = (a + b) / 2
    for i in circular_chns:
        c[0, 0, i] = circular_mid_point(a[0, 0, i], b[0, 0, i])
    return c


def circular_mid_point(a, b):
    mu = (a + b + 1) / 2 if abs(a - b) > 0.5 else (a + b) / 2
    if mu >= 1:
        mu = mu - 1
    return mu
