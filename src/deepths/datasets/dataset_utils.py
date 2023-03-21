"""
Set of utility functions common across datasets.
"""

import numpy as np
import os
import sys
import random

from torch.utils import data as torch_data
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


def patch_img(img_size, num_colours, num_patches):
    img = np.zeros((*img_size, 3), dtype='uint8')
    colours = unique_colours(num_colours)
    patch_rows = int(np.ceil(img_size[0] / num_patches))
    patch_cols = int(np.ceil(img_size[1] / num_patches))
    for r_ind in range(num_patches):
        srow = r_ind * patch_rows
        erow = min(srow + patch_rows, img.shape[0])
        for c_ind in range(num_patches):
            patch_ind = c_ind + r_ind * num_patches
            scol = c_ind * patch_cols
            ecol = min(scol + patch_cols, img.shape[1])
            colour_ind = np.mod(patch_ind, num_colours)
            img[srow:erow, scol:ecol] = colours[colour_ind]
            if colour_ind == num_colours:
                random.shuffle(colours)
    return img


def uniform_img(bg_size, num_chns, value):
    return np.zeros((*bg_size, num_chns), dtype='uint8') + value


def background_img(bg_type, bg_size, num_chns=3):
    if type(bg_size) not in [tuple, list]:
        bg_size = (bg_size, bg_size)
    if type(bg_type) == np.ndarray:
        bg_img = cv2.resize(bg_type, bg_size, interpolation=cv2.INTER_NEAREST)
    elif type(bg_type) == str:
        if bg_type == 'uniform_achromatic':
            bg_img = uniform_img(bg_size, num_chns, np.random.randint(0, 256, dtype='uint8'))
        elif bg_type == 'uniform_colour':
            bg_img = uniform_img(bg_size, num_chns, random_colour())
        elif os.path.exists(bg_type):
            bg_img = cv2_loader(bg_type, num_chns)
            bg_img = cv2.resize(bg_img, bg_size, interpolation=cv2.INTER_NEAREST)
        elif bg_type == 'rnd_img':
            bg_img = np.random.randint(0, 256, (*bg_size, num_chns), dtype='uint8')
        elif 'patch_colour_' in bg_type:
            num_colours, num_patches = [int(item) for item in bg_type.split('_')[-2:]]
            bg_img = patch_img(bg_size, num_colours, num_patches)
        else:
            bg_img = uniform_img(bg_size, num_chns, int(bg_type))
    elif type(bg_type) is list and len(bg_type) == 3:
        bg_img = uniform_img(bg_size, num_chns, np.array(bg_type))
    else:
        bg_img = uniform_img(bg_size, num_chns, bg_type)
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


def cv2_loader_3chns(path):
    return cv2_loader(path, num_chns=3)


def cv2_loader(path, num_chns=None):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif num_chns is not None:
        img = np.repeat(img[:, :, np.newaxis], num_chns, axis=2)
    return img


def rotate_img(img, angle):
    (h, w) = img.shape[:2]
    (cx, cy) = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    return cv2.warpAffine(img, rot_mat, (w, h), flags=cv2.INTER_NEAREST)


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


class BackgroundGenerator(torch_data.Dataset):
    def __init__(self, background, target_size):
        self.background = background
        self.target_size = target_size

    def __getitem__(self, _):
        return (background_img(self.background, self.target_size) * 255).astype('uint8')

    def __len__(self):
        return 1000000


def make_bg_loader(background, target_size):
    if type(background) is str and os.path.isdir(background):
        scale = (0.5, 1.0)
        bg_transform = torch_transforms.Compose(pre_transform_train(target_size, scale))
        bg_db = NoTargetFolder(background, loader=cv2_loader_3chns)
    else:
        bg_db = BackgroundGenerator(background, target_size)
        bg_transform = None
    return bg_db, bg_transform
