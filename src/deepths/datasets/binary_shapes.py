"""
Creating PyTorch dataloader from a set of binary images.
"""

import os
import numpy as np

from skimage import io
import cv2

from torch.utils import data as torch_data


def _create_bg_img(bg, mask_size, full_size):
    if os.path.exists(bg):
        bg_img = io.imread(bg)
        mask_img = cv2.resize(bg_img, mask_size, interpolation=cv2.INTER_NEAREST)
        full_img = cv2.resize(bg_img, full_size, interpolation=cv2.INTER_NEAREST)
    elif bg == 'rnd_img':
        mask_img = np.random.randint(0, 256, (*mask_size, 3), dtype='uint8')
        full_img = np.random.randint(0, 256, (*full_size, 3), dtype='uint8')
    elif bg == 'rnd_uniform':
        rnd_bg = np.random.randint(0, 256, dtype='uint8')
        mask_img = np.zeros((*mask_size, 3), dtype='uint8') + rnd_bg
        full_img = np.zeros((*full_size, 3), dtype='uint8') + rnd_bg
    else:
        mask_img = np.zeros((*mask_size, 3), dtype='uint8') + int(bg)
        full_img = np.zeros((*full_size, 3), dtype='uint8') + int(bg)
    mask_img = mask_img.astype('float32') / 255
    full_img = full_img.astype('float32') / 255
    return mask_img, full_img


class ShapeDataset(torch_data.Dataset):
    def __init__(self, root, transform=None, background=None, same_rotation=None, **kwargs):
        self.root = root
        self.transform = transform
        self.target_size = (224, 224)
        self.mask_size = (128, 128)
        self.imgdir = '%s/shape2D/' % self.root
        self.bg = background
        self.same_rotation = same_rotation

    def _one_out_img(self, mask, current_colour, place_fun):
        mask = cv2.resize(mask, self.mask_size, interpolation=cv2.INTER_NEAREST)
        mask_img, img = _create_bg_img(self.bg, self.mask_size, self.target_size)

        for chn_ind in range(3):
            current_chn = mask_img[:, :, chn_ind]
            current_chn[mask == 255] = current_colour[chn_ind]

        srow, scol = place_fun(self.mask_size, self.target_size)
        erow = srow + self.mask_size[0]
        ecol = scol + self.mask_size[1]
        img[srow:erow, scol:ecol] = mask_img
        return img

    def _one_out_img_uint8(self, mask, current_colour, place_fun):
        img = self._one_out_img(mask, current_colour, place_fun)
        return (img * 255).astype('uint8')

    def _mul_out_imgs(self, masks, others_colour, target_colour, place_fun):
        imgs = []
        for mask_ind, mask in enumerate(masks):
            current_colour = target_colour if mask_ind == 0 else others_colour
            imgs.append(self._one_out_img(mask, current_colour, place_fun))
        if self.transform is not None:
            imgs = self.transform(imgs)
        return imgs
