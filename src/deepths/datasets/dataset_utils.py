"""
Set of utility functions common across datasets.
"""

import sys

import cv2

from ..utils import colour_spaces


def cv2_loader(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


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
