"""
Creating PyTorch dataloader from a set of binary images.
"""

import os
import numpy as np
import glob
import random
import ntpath

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


def _random_place(mask_size, target_size):
    srow = random.randint(0, target_size[0] - mask_size[0])
    scol = random.randint(0, target_size[1] - mask_size[1])
    return srow, scol


class ShapeDataset(torch_data.Dataset):
    def __init__(self, root, transform=None, background=None, same_rotation=None,
                 target_size=None, mask_size=None, **kwargs):
        self.root = root
        self.transform = transform
        self.target_size = 224 if target_size is None else target_size
        if isinstance(self.target_size, int):
            self.target_size = (self.target_size, self.target_size)
        self.mask_size = 128 if mask_size is None else mask_size
        if isinstance(self.mask_size, int):
            self.mask_size = (self.mask_size, self.mask_size)
        self.imgdir = '%s/shape2D/' % self.root
        self.bg = background
        self.same_rotation = same_rotation

    def _one_out_img(self, mask, colour, place_fun):
        mask = cv2.resize(mask, self.mask_size, interpolation=cv2.INTER_NEAREST)
        mask_img, img = _create_bg_img(self.bg, self.mask_size, self.target_size)

        for chn_ind in range(3):
            current_chn = mask_img[:, :, chn_ind]
            current_chn[mask == 255] = colour[chn_ind]

        srow, scol = place_fun(self.mask_size, self.target_size)
        erow = srow + self.mask_size[0]
        ecol = scol + self.mask_size[1]
        img[srow:erow, scol:ecol] = mask_img
        return img

    def _one_out_img_uint8(self, mask, colour, place_fun):
        img = self._one_out_img(mask, colour, place_fun)
        return (img * 255).astype('uint8')

    def _mul_out_imgs(self, masks, others_colour, target_colour, place_fun):
        imgs = []
        for mask_ind, mask in enumerate(masks):
            current_colour = target_colour if mask_ind == 0 else others_colour
            imgs.append(self._one_out_img(mask, current_colour, place_fun))
        if self.transform is not None:
            imgs = self.transform(imgs)
        return imgs


class ShapeTrain(ShapeDataset):

    def __init__(self, root, transform=None, colour_dist=None, **kwargs):
        ShapeDataset.__init__(self, root, transform=transform, **kwargs)
        if self.bg is None:
            self.bg = 'rnd_uniform'
        if self.same_rotation is None:
            self.same_rotation = False
        self.angles = (1, 11)
        self.img_paths = sorted(glob.glob(self.imgdir + '*.png'))
        self.colour_dist = colour_dist
        if self.colour_dist is not None:
            self.colour_dist = np.loadtxt(self.colour_dist, delimiter=',', dtype=int)

    def _one_train_img(self, mask_img, target_colour):
        target_colour = np.array(target_colour).astype('float32') / 255
        return self._one_out_img_uint8(mask_img, target_colour, _random_place)

    def _mul_train_imgs(self, masks, others_colour, target_colour):
        others_colour = np.array(others_colour).astype('float32') / 255
        target_colour = np.array(target_colour).astype('float32') / 255
        imgs = self._mul_out_imgs(masks, others_colour, target_colour, _random_place)
        return imgs

    def _get_target_colour(self):
        if self.colour_dist is not None:
            rand_row = random.randint(0, len(self.colour_dist) - 1)
            target_colour = self.colour_dist[rand_row]
        else:
            target_colour = [random.randint(0, 255) for _ in range(3)]
        return target_colour

    def _angle_paths(self, path, samples):
        angle = int(ntpath.basename(path[:-4]).split('_')[-1].replace('angle', ''))
        ang_pool = list(np.arange(*self.angles))
        ang_pool.remove(angle)
        random.shuffle(ang_pool)
        org_angle = 'angle%d.png' % angle
        angle_paths = [path.replace(org_angle, 'angle%d.png' % ang_pool[i]) for i in range(samples)]
        return angle_paths

    def __len__(self):
        return len(self.img_paths)
