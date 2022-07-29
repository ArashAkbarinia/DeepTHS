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


def apply_vision_type(img, colour_space, vision_type):
    if 'grey' not in colour_space and vision_type != 'trichromat':
        opp_img = colour_spaces.rgb2dkl(img)
        if vision_type == 'dichromat_rg':
            opp_img[:, :, 1] = 0
        elif vision_type == 'dichromat_yb':
            opp_img[:, :, 2] = 0
        elif vision_type == 'monochromat':
            opp_img[:, :, [1, 2]] = 0
        else:
            sys.exit('Vision type %s not supported' % vision_type)
        img = colour_spaces.dkl2rgb(opp_img)
    return img
