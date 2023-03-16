"""
Dataloader for the orientation discrimination task.
"""

import numpy as np

from skimage import io

from torch.utils import data as torch_data

from .binary_shapes import ShapeTrain
from . import dataset_utils, stimuli_bank


class ShapeOddOneOut(ShapeTrain):

    def __init__(self, root, transform=None, **kwargs):
        ShapeTrain.__init__(self, root, transform=transform, **kwargs)
        self.num_stimuli = 4

    def __getitem__(self, item):
        target_path = self.img_paths[item]
        masks = []
        for i in range(self.num_stimuli):
            mask_img = io.imread(target_path)
            rows, cols = mask_img.shape[:2]
            mask = np.zeros((rows, cols), dtype=mask_img.dtype)
            mask = dataset_utils.merge_fg_bg(mask, mask_img, 'centre')
            if i == 0:
                mask = dataset_utils.rotate_img(mask, np.random.randint(1, 180))
            masks.append(mask)

        # set the colours
        colour = self._get_target_colour()
        bg = self._unique_bg([colour])
        imgs = self._mul_train_imgs(masks, colour, colour, bg)

        inds = dataset_utils.shuffle(list(np.arange(0, self.num_stimuli)))
        # the target is always added the first element in the imgs list
        target = inds.index(0)
        imgs = [imgs[i] for i in inds]
        return *imgs, target


class SinusoidalGratings(torch_data.Dataset):
    def __init__(self, transform, target_size, rotation, thetas=None, phases=None, sfs=None):
        self.transform = transform
        self.target_size = target_size
        self.rotation = rotation
        self.thetas = thetas if thetas is not None else np.arange(0, 181, 15)
        self.phases = phases if phases is not None else [0, 90, 180]
        self.sfs = sfs if sfs is not None else [1, 2, 4, 6, 8, 16]
        self.num_stimuli = 4

    def __getitem__(self, item):
        imgs = []
        settings_shape = (len(self.thetas), len(self.phases), len(self.sfs))
        theta_i, phase_i, sf_i = np.unravel_index(item, settings_shape)
        theta = self.thetas[theta_i]
        for angle in [theta + self.rotation, theta, theta, theta]:
            sf_base = ((self.target_size / 2) / np.pi)
            lambda_wave = sf_base / self.sfs[sf_i]

            angle = np.deg2rad(angle)
            omega = [np.cos(angle), np.sin(angle)]
            # generating the gratings
            sinusoid_param = {
                'amp': 1.0, 'omega': omega, 'rho': np.deg2rad(self.phases[phase_i]),
                'img_size': self.target_size, 'lambda_wave': lambda_wave
            }
            img = stimuli_bank.sinusoid_grating(**sinusoid_param)
            img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
            imgs.append(((img + 1) / 2).copy())

        if self.transform:
            imgs = self.transform(imgs)

        inds = dataset_utils.shuffle(list(np.arange(0, self.num_stimuli)))
        # the target is always added the first element in the imgs list
        target = inds.index(0)
        imgs = [imgs[i] for i in inds]
        return *imgs, target

    def __len__(self):
        return len(self.thetas) * len(self.phases) * len(self.sfs)


def train_val_set(root, target_size, preprocess, **kwargs):
    transform = dataset_utils.eval_preprocess(target_size, preprocess)
    return ShapeOddOneOut(root, transform, **kwargs)


def test_set(target_size, preprocess, **kwargs):
    transform = dataset_utils.eval_preprocess(target_size, preprocess)
    return SinusoidalGratings(transform, target_size, **kwargs)
