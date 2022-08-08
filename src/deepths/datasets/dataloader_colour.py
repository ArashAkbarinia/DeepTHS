"""
Dataloader for the colour discrimination task.
"""

import numpy as np
import random

from skimage import io

from .binary_shapes import ShapeMultipleOut, ShapeTrain
from . import dataset_utils


def _get_others_colour(target_colour):
    others_diff = [random.choice([1, -1]) * random.randint(1, 128) for _ in range(3)]
    others_colour = []
    for chn_ind in range(3):
        chn_colour = target_colour[chn_ind] + others_diff[chn_ind]
        if chn_colour < 0 or chn_colour > 255:
            chn_colour = target_colour[chn_ind] - others_diff[chn_ind]
        others_colour.append(chn_colour)
    return others_colour


class ShapeVal(ShapeMultipleOut):

    def __init__(self, root, transform=None, target_colour=None, others_colour=None, **kwargs):
        ShapeMultipleOut.__init__(self, root, transform=transform, **kwargs)
        if self.bg is None:
            self.bg = 128
        if self.same_rotation is None:
            self.same_rotation = True
        stimuli_path = '%s/validation.cvs' % self.root
        self.stimuli = np.loadtxt(stimuli_path, delimiter=',', dtype=int)
        self.target_colour = target_colour
        self.others_colour = others_colour

    def _prepare_test_imgs(self, masks):
        others_colour = self.others_colour.squeeze()
        target_colour = self.target_colour.squeeze()
        imgs = self._mul_out_imgs(masks, others_colour, target_colour, 'centre')
        return imgs

    def __len__(self):
        return len(self.stimuli)


class ShapeOddOneOutTrain(ShapeTrain):

    def __init__(self, root, transform=None, colour_dist=None, **kwargs):
        ShapeTrain.__init__(self, root, transform=transform, colour_dist=colour_dist, **kwargs)
        self.num_stimuli = 4

    def __getitem__(self, item):
        target_path = self.img_paths[item]
        other_paths = [target_path] * 3 if self.same_rotation else self._angle_paths(target_path, 3)
        masks = [io.imread(target_path), *[io.imread(opath) for opath in other_paths]]

        # set the colours
        target_colour = self._get_target_colour()
        others_colour = _get_others_colour(target_colour)

        imgs = self._mul_train_imgs(masks, others_colour, target_colour)

        inds = list(np.arange(0, self.num_stimuli))
        random.shuffle(inds)
        # the target is always added the first element in the imgs list
        target = inds.index(0)
        return imgs[inds[0]], imgs[inds[1]], imgs[inds[2]], imgs[inds[3]], target


class ShapeOddOneOutVal(ShapeVal):

    def __init__(self, root, transform=None, **kwargs):
        ShapeVal.__init__(self, root, transform=transform, **kwargs)
        self.num_stimuli = 4

    def __getitem__(self, item):
        # image names start from 1
        imgi = item + 1
        base_path = '%s/img_shape%d_angle' % (self.imgdir, imgi)
        target_path = '%s%d.png' % (base_path, self.stimuli[item, 0])
        if self.same_rotation:
            other_paths = [target_path] * 3
        else:
            other_paths = ['%s%d.png' % (base_path, self.stimuli[item, i]) for i in range(1, 4)]
        masks = [io.imread(target_path), *[io.imread(opath) for opath in other_paths]]

        imgs = self._prepare_test_imgs(masks)

        # the target is always added the first element in the imgs list
        target = self.stimuli[item, -1]
        inds = list(np.arange(0, self.num_stimuli))
        tmp_img = imgs[target].clone()
        imgs[target] = imgs[0].clone()
        imgs[0] = tmp_img.clone()
        return imgs[inds[0]], imgs[inds[1]], imgs[inds[2]], imgs[inds[3]], target


class Shape2AFCTrain(ShapeTrain):

    def __init__(self, root, transform=None, colour_dist=None, **kwargs):
        ShapeTrain.__init__(self, root, transform=transform, colour_dist=colour_dist, **kwargs)

    def __getitem__(self, item):
        target_path = self.img_paths[item]
        other_paths = target_path if self.same_rotation else self._angle_paths(target_path, 1)[0]
        masks = [io.imread(target_path), io.imread(other_paths)]

        # set the colours
        target_colour = self._get_target_colour()
        if random.random() < 0.5:
            target = 1
            others_colour = target_colour
        else:
            target = 0
            others_colour = _get_others_colour(target_colour)

        imgs = self._mul_train_imgs(masks, others_colour, target_colour)

        return imgs[0], imgs[1], target


class Shape2AFCVal(ShapeVal):

    def __init__(self, root, transform=None, **kwargs):
        ShapeVal.__init__(self, root, transform=transform, **kwargs)

    def __getitem__(self, item):
        # image names start from 1
        imgi = item + 1
        target_path = '%s/img_shape%d_angle%d.png' % (self.imgdir, imgi, self.stimuli[item, 0])
        if self.same_rotation:
            other_paths = target_path
        else:
            other_paths = '%s/img_shape%d_angle%d.png' % (self.imgdir, imgi, self.stimuli[item, 1])
        masks = [io.imread(target_path), io.imread(other_paths)]

        imgs = self._prepare_test_imgs(masks)

        # target doesn't have a meaning in this test, it's always False
        target = 0
        return imgs[0], imgs[1], target


def train_set(root, target_size, preprocess, task, **kwargs):
    db_fun = ShapeOddOneOutTrain if task == 'odd4' else Shape2AFCTrain
    transform = dataset_utils.train_preprocess(target_size, preprocess, (0.8, 1.0))
    return db_fun(root, transform, **kwargs)


def val_set(root, target_size, preprocess, task, **kwargs):
    db_fun = ShapeOddOneOutVal if task == 'odd4' else Shape2AFCVal
    transform = dataset_utils.eval_preprocess(target_size, preprocess)
    return db_fun(root, transform, **kwargs)
