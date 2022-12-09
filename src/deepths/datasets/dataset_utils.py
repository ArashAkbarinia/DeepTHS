"""
Set of utility functions common across datasets.
"""

import numpy as np
import os
import sys
import random

import torchvision.transforms as torch_transforms
from torchvision import datasets as torch_datasets

import cv2

from . import cv2_transforms
from ..utils import colour_spaces


def random_colour(chns=3):
    return [random.randint(0, 255) for _ in range(chns)]


def unique_colours(num, exclude=None, chns=3):
    if exclude is None:
        exclude = []
    colours = []
    for i in range(num):
        while True:
            colour = random_colour(chns=chns)
            if colour not in colours and colour not in exclude:
                colours.append(colour)
                break
    return colours


def randint(low, high):
    low, high = int(low), int(high)
    return low if low >= high else np.random.randint(low, high)


def shuffle(arr):
    random.shuffle(arr)
    return arr


def background_img(bg_type, bg_size, num_chns=3):
    if type(bg_type) == str and os.path.exists(bg_type):
        bg_img = cv2.resize(cv2_loader(bg_type), bg_size, interpolation=cv2.INTER_NEAREST)
    elif bg_type == 'rnd_img':
        bg_img = np.random.randint(0, 256, (*bg_size, num_chns), dtype='uint8')
    else:
        if bg_type == 'uniform_achromatic':
            rnd_bg = np.random.randint(0, 256, dtype='uint8')
        elif bg_type == 'uniform_colour':
            rnd_bg = random_colour()
        elif type(bg_type) == str:
            rnd_bg = int(bg_type)
        else:
            rnd_bg = bg_type
        bg_img = np.zeros((*bg_size, num_chns), dtype='uint8') + rnd_bg
    return bg_img.astype('float32') / 255


def random_place(fg_size, bg_size):
    srow = random.randint(0, bg_size[0] - fg_size[0])
    scol = random.randint(0, bg_size[1] - fg_size[1])
    return srow, scol


def relative_place(fg_size, bg_size, pos):
    srow = pos[0] * (bg_size[0] - fg_size[0])
    scol = pos[1] * (bg_size[1] - fg_size[1])
    return int(srow), int(scol)


def centre_place(fg_size, bg_size):
    srow = (bg_size[0] - fg_size[0]) // 2
    scol = (bg_size[1] - fg_size[1]) // 2
    return srow, scol


def check_place_fun(place_fun):
    if isinstance(place_fun, str):
        return centre_place if place_fun == 'centre' else random_place
    return place_fun


def merge_fg_bg(bg, fg, place_fun, alpha=0):
    srow, scol = check_place_fun(place_fun)(fg.shape[:2], bg.shape[:2])
    return merge_fg_bg_at_loc(bg, fg, srow, scol, alpha)


def merge_fg_bg_at_loc(bg, fg, srow, scol, alpha=0):
    bg = bg.copy()
    erow = srow + fg.shape[0]
    ecol = scol + fg.shape[1]
    bg[srow:erow, scol:ecol] = (1 - alpha) * fg.copy() + alpha * bg[srow:erow, scol:ecol]
    return bg


def crop_fg_from_bg(bg, fg_size, srow, scol):
    erow = srow + fg_size[0]
    ecol = scol + fg_size[1]
    return bg[srow:erow, scol:ecol].copy()


def cv2_loader(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) > 2 else img


def rgb2opp_funs(colour_space):
    if colour_space == 'dkl':
        ffun = colour_spaces.rgb2dkl01
        bfun = colour_spaces.dkl012rgb01
    elif colour_space == 'yog':
        ffun = colour_spaces.rgb2yog01
        bfun = colour_spaces.yog012rgb01
    else:
        sys.exit('Unsupported colour space %s.' % colour_space)
    return ffun, bfun


def opp2rgb_funs(colour_space):
    ffun, bfun = rgb2opp_funs(colour_space)
    return bfun, ffun


def apply_vision_type_rgb(img, colour_space, vision_type):
    if 'grey' not in colour_space and vision_type != 'trichromat':
        ffun, bfun = rgb2opp_funs(colour_space)
        img = bfun(apply_vision_type(ffun(img), colour_space, vision_type))
    return img


def apply_vision_type(opp_img, colour_space, vision_type):
    if 'grey' not in colour_space and vision_type != 'trichromat':
        if vision_type == 'dichromat_rg':
            opp_img[:, :, 1] = 0.5
        elif vision_type == 'dichromat_yb':
            opp_img[:, :, 2] = 0.5
        elif vision_type == 'monochromat':
            opp_img[:, :, [1, 2]] = 0.5
        else:
            sys.exit('Vision type %s not supported' % vision_type)
    return opp_img


def eval_preprocess(target_size, preprocess):
    return torch_transforms.Compose([
        *pre_transform_eval(target_size),
        *post_transform(*preprocess)
    ])


def train_preprocess(target_size, preprocess, scale):
    return torch_transforms.Compose([
        *pre_transform_train(target_size, scale),
        *post_transform(*preprocess)
    ])


def pre_transform_train(target_size, scale):
    return [
        cv2_transforms.RandomResizedCrop(target_size, scale=scale),
        cv2_transforms.RandomHorizontalFlip(),
    ]


def pre_transform_eval(target_size):
    return [
        cv2_transforms.Resize(target_size),
        cv2_transforms.CenterCrop(target_size),
    ]


def post_transform(mean, std):
    return [
        cv2_transforms.ToTensor(),
        cv2_transforms.Normalize(mean, std),
    ]


class NoTargetFolder(torch_datasets.ImageFolder):

    def __getitem__(self, item):
        img, _ = super().__getitem__(item)
        return img


class ItemPathFolder(torch_datasets.ImageFolder):
    def __getitem__(self, item):
        path, _ = self.samples[item]
        return path
