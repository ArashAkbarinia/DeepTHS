"""
Dataloader for the colour discrimination task.
"""

import numpy as np
import sys
import random

from skimage import io

from .binary_shapes import ShapeMultipleOut, ShapeTrain, ShapeDataset
from . import dataset_utils
from ..utils import system_utils, colour_spaces


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
        imgs = self._mul_out_imgs(masks, others_colour, target_colour, self.bg, 'centre')
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
        others_colour = self._get_others_colour(target_colour)
        bg = self._unique_bg([target_colour, others_colour])
        imgs = self._mul_train_imgs(masks, others_colour, target_colour, bg)

        inds = dataset_utils.shuffle(list(np.arange(0, self.num_stimuli)))
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
        bg = self._unique_bg([target_colour, others_colour])
        imgs = self._mul_train_imgs(masks, others_colour, target_colour, bg)
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


class ShapeTripleColoursOdd4(ShapeDataset):
    def __init__(self, root, test_colour, ref_colours, transform=None, target=0, **kwargs):
        ShapeDataset.__init__(self, root, transform=transform, **kwargs)
        if self.bg is None:
            self.bg = 128
        self.stimuli = sorted(system_utils.image_in_folder(self.imgdir))
        self.test_colour = test_colour
        self.ref_colours = ref_colours
        self.target = target  # target can be irrelevant depending on the experiment

    def __getitem__(self, item):
        mask = dataset_utils.cv2_loader(self.stimuli[item])

        img0 = self._one_out_img(mask, self.ref_colours[0].squeeze(), self.bg, 'centre')
        img_test = self._one_out_img(mask, self.test_colour.squeeze(), self.bg, 'centre')
        img1 = self._one_out_img(mask, self.ref_colours[1].squeeze(), self.bg, 'centre')
        imgs = [img0, img_test, img_test, img1]

        if self.transform is not None:
            imgs = self.transform(imgs)
        return *imgs, self.target

    def __len__(self):
        return len(self.stimuli)


def organise_test_points(test_pts):
    out_test_pts = dict()
    for test_pt in test_pts:
        pt_val = test_pt[:3].astype('float')
        test_pt_name = test_pt[-2]
        if 'ref_' == test_pt_name[:4]:
            test_pt_name = test_pt_name[4:]
            if test_pt[-1] == 'dkl':
                ffun = colour_spaces.dkl012rgb01
                bfun = colour_spaces.rgb2dkl01
                chns_name = ['D', 'K', 'L']
            elif test_pt[-1] == 'hsv':
                ffun = colour_spaces.hsv012rgb01
                bfun = colour_spaces.rgb2hsv01
                chns_name = ['H', 'S', 'V']
            elif test_pt[-1] == 'xyy':
                ffun = colour_spaces.xyy2rgb
                bfun = colour_spaces.rgb2xyy
                chns_name = ['X', 'Y', 'Y']
            elif test_pt[-1] == 'ycc':
                ffun = colour_spaces.ycc012rgb01
                bfun = colour_spaces.rgb2ycc01
                chns_name = ['Y', 'C', 'C']
            elif test_pt[-1] == 'rgb':
                ffun = colour_spaces.identity
                bfun = colour_spaces.identity
                chns_name = ['R', 'G', 'B']
            else:
                sys.exit('Unsupported colour space %s' % test_pt[-1])
            out_test_pts[test_pt_name] = {
                'ref': pt_val, 'ffun': ffun, 'bfun': bfun, 'space': chns_name, 'ext': [], 'chns': []
            }
        else:
            out_test_pts[test_pt_name]['ext'].append(pt_val)
            out_test_pts[test_pt_name]['chns'].append(test_pt[-1])
    return out_test_pts


def train_set(root, target_size, preprocess, task, **kwargs):
    db_fun = ShapeOddOneOutTrain if task == 'odd4' else Shape2AFCTrain
    transform = dataset_utils.train_preprocess(target_size, preprocess, (0.8, 1.0), im2double=False)
    return db_fun(root, transform, **kwargs)


def val_set(root, target_size, preprocess, task, **kwargs):
    db_fun = ShapeOddOneOutVal if task == 'odd4' else Shape2AFCVal
    transform = dataset_utils.eval_preprocess(target_size, preprocess, im2double=False)
    return db_fun(root, transform, **kwargs)


def triple_colours_odd4(root, target_size, preprocess, **kwargs):
    transform = dataset_utils.eval_preprocess(target_size, preprocess)
    return ShapeTripleColoursOdd4(root, transform=transform, **kwargs)
