"""
Data loader for contrast discrimination routine.
"""

import sys
import os
import numpy as np
import random

import cv2
from skimage import io
from torchvision import datasets as tdatasets
from torch.utils import data as torch_data
import torchvision.transforms as torch_transforms

from ..utils import colour_spaces, system_utils
from . import imutils
from . import stimuli_bank
from . import cv2_transforms
from .binary_shapes import ShapeTrain
from . import dataset_utils

NATURAL_DATASETS = ['imagenet', 'celeba', 'land', 'bw']


def _two_pairs_stimuli(img0, img1, con0, con1, p=0.5, contrast_target=None):
    imgs_cat = [img0, img1]
    max_contrast = np.argmax([con0, con1])
    if contrast_target is None:
        contrast_target = 0 if random.random() < p else 1
    if max_contrast != contrast_target:
        imgs_cat = imgs_cat[::-1]

    return (imgs_cat[0], imgs_cat[1]), contrast_target


def _prepare_grating_detector(img0, colour_space, vision_type, contrasts, mask_image,
                              pre_transform, post_transform, p, illuminant=0.0,
                              sf_filter=None, contrast_space='rgb'):
    # converting to range 0 to 1
    img0 = np.float32(img0) / 255
    img0 = dataset_utils.apply_vision_type_rgb(img0, colour_space, vision_type)

    if pre_transform is not None:
        [img0] = pre_transform([img0])

    if contrasts is None:
        contrast0 = random.uniform(0, 1)
    else:
        contrast0 = random.uniform(contrasts[0], contrasts[1])

    if 'grey' in colour_space:
        img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
        if colour_space == 'grey':
            img0 = np.expand_dims(img0, axis=2)
        elif colour_space == 'grey3':
            img0 = np.repeat(img0[:, :, np.newaxis], 3, axis=2)

    # applying SF filter
    if sf_filter is not None:
        hsf_cut, lsf_cut = sf_filter
        img0 = imutils.filter_img_sf(img0, hsf_cut=hsf_cut, lsf_cut=lsf_cut)

    # manipulating the contrast
    if contrast_space != 'rgb':
        img0 = colour_spaces.rgb2dkl01(img0)

    if random.random() < p:
        grating = _random_grating(img0.shape[0], contrast0)
        contrast_target = 1

        if contrast_space == 'lum':
            img0[:, :, 0] = (img0[:, :, 0] + grating) / 2
        elif contrast_space == 'rg':
            img0[:, :, 1] = (img0[:, :, 1] + grating) / 2
        elif contrast_space == 'yb':
            img0[:, :, 2] = (img0[:, :, 2] + grating) / 2
        elif contrast_space == 'rgb':
            grating = np.repeat(grating[:, :, np.newaxis], 3, axis=2)
            img0 = (img0 + grating) / 2
        elif contrast_space == 'dkl':
            chn = random.randint(0, 2)
            img0[:, :, chn] = (img0[:, :, chn] + grating) / 2
            for i in range(3):
                if i == chn:
                    continue
                if random.random() < 0.5:
                    img0[:, :, i] = (img0[:, :, i] + grating) / 2
    else:
        grating = 0.5
        contrast_target = 0
        img0 = (img0 + grating) / 2

    if contrast_space != 'rgb':
        img0 = colour_spaces.dkl012rgb01(img0)

    # multiplying by the illuminant
    if illuminant is None:
        illuminant = [1e-4, 1.0]
    if type(illuminant) in (list, tuple):
        if len(illuminant) == 1:
            ill_val = illuminant[0]
        else:
            ill_val = np.random.uniform(low=illuminant[0], high=illuminant[1])
    else:
        ill_val = illuminant
    # we simulate the illumination with multiplication
    # is ill_val is very small, the image becomes very dark
    img0 *= ill_val
    half_ill = ill_val / 2

    if mask_image == 'gaussian':
        img0 -= half_ill
        img0 = img0 * _gauss_img(img0.shape)
        img0 += half_ill

    if post_transform is not None:
        [img0] = post_transform([img0])

    return img0, contrast_target


def _prepare_stimuli(img0, colour_space, vision_type, contrasts, mask_image,
                     pre_transform, post_transform, same_transforms, p,
                     illuminant=0.0, current_param=None, sf_filter=None,
                     contrast_space='rgb', grating_detector=False, p_chn_wise=0.5):
    if grating_detector:
        if current_param:
            sys.exit('For grating_detector current_param cant be true')
        return _prepare_grating_detector(
            img0, colour_space, vision_type, contrasts, mask_image, pre_transform, post_transform,
            p, illuminant, sf_filter, contrast_space
        )

    # converting to range 0 to 1
    img0 = np.float32(img0) / 255
    img0 = dataset_utils.apply_vision_type_rgb(img0, colour_space, vision_type)
    # copying to img1
    img1 = img0.copy()

    contrast_target = None
    # if current_param is passed no randomness is generated on the fly
    if current_param is not None:
        # cropping
        srow0, scol0, srow1, scol1 = current_param['crops']
        img0 = img0[srow0:, scol0:, :]
        img1 = img1[srow1:, scol1:, :]
        # flipping
        hflip0, hflip1 = current_param['hflips']
        if hflip0 > 0.5:
            img0 = img0[:, ::-1, :]
        if hflip1 > 0.5:
            img1 = img1[:, ::-1, :]
        # contrast
        contrasts = current_param['contrasts']
        # side of high contrast
        contrast_target = 0 if current_param['ps'] < 0.5 else 1

    if pre_transform is not None:
        if same_transforms:
            img0, img1 = pre_transform([img0, img1])
        else:
            [img0] = pre_transform([img0])
            [img1] = pre_transform([img1])

    if contrasts is None:
        min_contrast = 0.004
        contrast0 = random.uniform(min_contrast, 1)
        contrast1 = random.uniform(min_contrast, 1)
    else:
        contrast0, contrast1 = contrasts

    if 'grey' in colour_space:
        img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        if colour_space == 'grey':
            img0 = np.expand_dims(img0, axis=2)
            img1 = np.expand_dims(img1, axis=2)
        elif colour_space == 'grey3':
            img0 = np.repeat(img0[:, :, np.newaxis], 3, axis=2)
            img1 = np.repeat(img1[:, :, np.newaxis], 3, axis=2)

    # applying SF filter
    if sf_filter is not None:
        hsf_cut, lsf_cut = sf_filter
        img0 = imutils.filter_img_sf(img0, hsf_cut=hsf_cut, lsf_cut=lsf_cut)
        img1 = imutils.filter_img_sf(img1, hsf_cut=hsf_cut, lsf_cut=lsf_cut)

    # manipulating the contrast
    if contrast_space == 'dkl':
        img0 = colour_spaces.rgb2dkl01(img0)
        img1 = colour_spaces.rgb2dkl01(img1)
        # probability distribution of which channel is changing
        p_chns = [0.6, 0.2, 0.2]
    else:
        p_chns = [1 / 3] * 3
    chn_wise = np.random.uniform() < p_chn_wise
    if chn_wise:
        chn = np.random.choice([0, 1, 2], p=p_chns)
        img0[:, :, chn] = imutils.adjust_contrast(img0[:, :, chn], contrast0)
        img1[:, :, chn] = imutils.adjust_contrast(img1[:, :, chn], contrast1)
    else:
        chn = -1
        img0 = imutils.adjust_contrast(img0, contrast0)
        img1 = imutils.adjust_contrast(img1, contrast1)
    if contrast_space == 'dkl':
        img0 = colour_spaces.dkl012rgb01(img0)
        img1 = colour_spaces.dkl012rgb01(img1)

    if mask_image == 'gaussian':
        mean_val = 0.5
        img0 -= mean_val
        img0 = img0 * _gauss_img(img0.shape)
        img0 += mean_val
        img1 -= mean_val
        img1 = img1 * _gauss_img(img1.shape)
        img1 += mean_val

    # getting the illuminant
    if illuminant is None:
        min_diff = min(img0.min() - 0, img1.min() - 0)
        max_diff = min(1 - img0.max(), 1 - img1.max())
        illuminant = [-min_diff, max_diff]
    if type(illuminant) in (list, tuple):
        if len(illuminant) == 1:
            ill_val = illuminant[0]
        else:
            ill_val = np.random.uniform(low=illuminant[0], high=illuminant[1])
    else:
        ill_val = illuminant

    # we simulate the illumination with addition
    img0 += ill_val
    img1 += ill_val

    if post_transform is not None:
        img0, img1 = post_transform([img0, img1])

    img_out, contrast_target = _two_pairs_stimuli(
        img0, img1, contrast0, contrast1, p, contrast_target=contrast_target
    )
    settings = np.array([contrast0, contrast1, ill_val, chn])
    return img_out, contrast_target, settings


def _gauss_img(img_size):
    midx = np.floor(img_size[1] / 2) + 1
    midy = np.floor(img_size[0] / 2) + 1
    y = np.linspace(img_size[0], 0, img_size[0]) - midy
    x = np.linspace(0, img_size[1], img_size[1]) - midx
    [x, y] = np.meshgrid(x, y)
    sigma = min(img_size[0], img_size[1]) / 6
    gauss_img = np.exp(-(np.power(x, 2) + np.power(y, 2)) / (2 * np.power(sigma, 2)))

    gauss_img = gauss_img / np.max(gauss_img)
    if len(img_size) > 2:
        gauss_img = np.repeat(gauss_img[:, :, np.newaxis], img_size[2], axis=2)
    return gauss_img


class AfcDataset(object):
    def __init__(self, post_transform=None, pre_transform=None, p=0.5, contrasts=None,
                 same_transforms=False, colour_space='grey', vision_type='trichromat',
                 mask_image=None, illuminant=0.0, train_params=None, sf_filter=None,
                 contrast_space='rgb', grating_detector=False):
        self.p = p
        self.contrasts = contrasts
        self.same_transforms = same_transforms
        self.colour_space = colour_space
        self.vision_type = vision_type
        self.mask_image = mask_image
        self.post_transform = post_transform
        self.pre_transform = pre_transform
        self.illuminant = illuminant
        self.train_params = train_params
        if self.train_params is not None:
            self.train_params = system_utils.read_pickle(train_params)
        self.img_counter = 0
        self.sf_filter = sf_filter
        self.contrast_space = contrast_space
        self.grating_detector = grating_detector


class CelebA(AfcDataset, tdatasets.CelebA):
    def __init__(self, afc_kwargs, celeba_kwargs):
        AfcDataset.__init__(self, **afc_kwargs)
        tdatasets.CelebA.__init__(self, **celeba_kwargs)
        self.loader = dataset_utils.cv2_loader_3chns

    def __getitem__(self, index):
        path = os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index])
        img0 = self.loader(path)

        img_out, contrast_target, img_settings = _prepare_stimuli(
            img0, self.colour_space, self.vision_type, self.contrasts, self.mask_image,
            self.pre_transform, self.post_transform, self.same_transforms, self.p,
            self.illuminant, sf_filter=self.sf_filter, contrast_space=self.contrast_space,
            grating_detector=self.grating_detector
        )

        if self.grating_detector:
            return img_out, contrast_target, img_settings
        else:
            return img_out[0], img_out[1], contrast_target, img_settings


class ImageFolder(AfcDataset, tdatasets.ImageFolder):
    def __init__(self, afc_kwargs, folder_kwargs):
        AfcDataset.__init__(self, **afc_kwargs)
        tdatasets.ImageFolder.__init__(self, **folder_kwargs)
        self.loader = dataset_utils.cv2_loader_3chns

    def __getitem__(self, index):
        current_param = None
        if self.train_params is not None:
            index = self.train_params['image_inds'][self.img_counter]
            current_param = {
                'ps': self.train_params['ps'][self.img_counter],
                'contrasts': self.train_params['contrasts'][self.img_counter],
                'hflips': self.train_params['hflips'][self.img_counter],
                'crops': self.train_params['crops'][self.img_counter]
            }
            self.img_counter += 1

        path, class_target = self.samples[index]
        img0 = self.loader(path)
        img_out, contrast_target, img_settings = _prepare_stimuli(
            img0, self.colour_space, self.vision_type, self.contrasts, self.mask_image,
            self.pre_transform, self.post_transform, self.same_transforms, self.p,
            self.illuminant, current_param=current_param, sf_filter=self.sf_filter,
            contrast_space=self.contrast_space, grating_detector=self.grating_detector
        )

        if self.grating_detector:
            return img_out, contrast_target, img_settings
        else:
            return img_out[0], img_out[1], contrast_target, img_settings


class BinaryShapes(AfcDataset, ShapeTrain):
    def __init__(self, afc_kwargs, shape_kwargs):
        AfcDataset.__init__(self, **afc_kwargs)
        ShapeTrain.__init__(self, **shape_kwargs)
        self.loader = dataset_utils.cv2_loader

    def __getitem__(self, index):
        current_param = None
        if self.train_params is not None:
            index = self.train_params['image_inds'][self.img_counter]
            current_param = {
                'ps': self.train_params['ps'][self.img_counter],
                'contrasts': self.train_params['contrasts'][self.img_counter],
                'hflips': self.train_params['hflips'][self.img_counter],
                'crops': self.train_params['crops'][self.img_counter]
            }
            self.img_counter += 1

        path = self.img_paths[index]
        mask_img = io.imread(path)
        target_colour = self._get_target_colour()
        img0 = self._one_train_img_uint8(mask_img, target_colour, self.bg)

        img_out, contrast_target, img_settings = _prepare_stimuli(
            img0, self.colour_space, self.vision_type, self.contrasts, self.mask_image,
            self.pre_transform, self.post_transform, self.same_transforms, self.p,
            self.illuminant, current_param=current_param, sf_filter=self.sf_filter,
            contrast_space=self.contrast_space, grating_detector=self.grating_detector
        )

        if self.grating_detector:
            return img_out, contrast_target, img_settings
        else:
            return img_out[0], img_out[1], contrast_target, img_settings


def _create_samples(samples):
    if 'illuminant' in samples:
        illuminant = samples['illuminant']
        del samples['illuminant']
    else:
        illuminant = 0.0
    settings = samples
    settings['lenghts'] = (
        len(settings['amp']), len(settings['lambda_wave']),
        len(settings['theta']), len(settings['rho']), len(settings['side'])
    )
    num_samples = np.prod(np.array(settings['lenghts']))
    return num_samples, settings, illuminant


def _random_grating(target_size, contrast0):
    rho = random.uniform(0, np.pi)
    sf = random.randint(1, target_size / 2)
    lambda_wave = (target_size * 0.5) / (np.pi * sf)
    theta = random.uniform(0, np.pi)
    omega = [np.cos(theta), np.sin(theta)]
    sinusoid_param = {
        'amp': contrast0, 'omega': omega, 'rho': rho,
        'img_size': [target_size, target_size], 'lambda_wave': lambda_wave
    }
    img0 = stimuli_bank.sinusoid_grating(**sinusoid_param)
    img0 = (img0 + 1) / 2

    # if target size is even, the generated stimuli is 1 pixel larger.
    if np.mod(target_size, 2) == 0:
        img0 = img0[:-1]
    if np.mod(target_size, 2) == 0:
        img0 = img0[:, :-1]
    return img0


def _convert_other_params(img, theta, rho, l_wave):
    shift = 0
    if rho == 180:
        sf = int(np.ceil((0.5 * img.shape[0]) / (np.pi * l_wave)))
        shift = int((90 / (sf * 180)) * img.shape[0])
    if theta == 0:
        img = np.roll(img, shift, axis=1)
    elif theta == 90:
        img = img.transpose()
        img = np.roll(img, shift, axis=0)
    else:
        # theta 45
        vals = img[0].copy()
        if rho == 180:
            vals = np.roll(vals, shift)
        vals = np.repeat(vals, 2)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                img[i, j] = vals[i + j]
        if theta == 135:
            img = img[:, ::-1]
    return img


class GratingImages(AfcDataset, torch_data.Dataset):
    def __init__(self, samples, afc_kwargs, target_size, theta=None, rho=None, lambda_wave=None):
        AfcDataset.__init__(self, **afc_kwargs)
        torch_data.Dataset.__init__(self)
        if type(samples) is dict:
            # under this condition one contrast will be zero while the other
            # takes the arguments of samples.
            self.samples, self.settings, self.illuminant = _create_samples(samples)
        else:
            self.samples = samples
            self.settings = None
        if type(target_size) not in [list, tuple]:
            target_size = (target_size, target_size)
        self.target_size = target_size
        self.theta = theta
        self.rho = rho
        self.lambda_wave = lambda_wave

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (img_l, imgout) where imgout is the same size as
             original image after applied manipulations.
        """
        if self.settings is None:
            if self.contrasts is None:
                contrast0 = random.uniform(0, 1)
                contrast1 = random.uniform(0, 1)
            else:
                contrast0, contrast1 = self.contrasts

            # randomising the parameters
            theta = random.choice([0, 45, 90, 135]) if self.theta is None else self.theta
            rho = random.choice([0, 180]) if self.rho is None else self.rho
            if self.lambda_wave is None:
                lambda_wave = random.uniform(np.pi / 2, np.pi * 10)
            else:
                lambda_wave = self.lambda_wave
        else:
            inds = np.unravel_index(index, self.settings['lenghts'])
            contrast0 = self.settings['amp'][inds[0]]
            lambda_wave = self.settings['lambda_wave'][inds[1]]
            theta = self.settings['theta'][inds[2]]
            rho = self.settings['rho'][inds[3]]
            self.p = self.settings['side'][inds[4]]
            contrast1 = 0

        # always create the 0 one then adjust to the others
        omega = [np.cos(0), np.sin(0)]
        # generating the gratings
        sinusoid_param = {
            'amp': contrast0, 'omega': omega, 'rho': 0,
            'img_size': self.target_size, 'lambda_wave': lambda_wave
        }
        img0 = stimuli_bank.sinusoid_grating(**sinusoid_param)
        sinusoid_param['amp'] = contrast1
        img1 = stimuli_bank.sinusoid_grating(**sinusoid_param)

        # if target size is even, the generated stimuli is 1 pixel larger.
        if np.mod(self.target_size[0], 2) == 0:
            img0 = img0[:-1]
            img1 = img1[:-1]
        if np.mod(self.target_size[1], 2) == 0:
            img0 = img0[:, :-1]
            img1 = img1[:, :-1]

        # if theta and rho are different from 0
        img0 = _convert_other_params(img0, theta, rho, lambda_wave)
        img1 = _convert_other_params(img1, theta, rho, lambda_wave)

        # multiply it by gaussian
        if self.mask_image == 'fixed_size':
            radius = (int(self.target_size[0] / 2.0), int(self.target_size[1] / 2.0))
            [x, y] = np.meshgrid(range(-radius[0], radius[0] + 1), range(-radius[1], radius[1] + 1))
            x1 = +x * np.cos(theta) + y * np.sin(theta)
            y1 = -x * np.sin(theta) + y * np.cos(theta)

            k = 2
            o1 = 8
            o2 = o1 / 2
            omg = (1 / 8) * (np.pi ** 2 / lambda_wave)
            gauss_img = omg ** 2 / (o2 * np.pi * k ** 2) * np.exp(
                -omg ** 2 / (o1 * k ** 2) * (1 * x1 ** 2 + y1 ** 2))

            if np.mod(self.target_size[0], 2) == 0:
                gauss_img = gauss_img[:-1]
            if np.mod(self.target_size[1], 2) == 0:
                gauss_img = gauss_img[:, :-1]
            gauss_img = gauss_img / np.max(gauss_img)
            img0 *= gauss_img
            img1 *= gauss_img
        elif self.mask_image == 'fixed_cycle':
            radius = (int(self.target_size[0] / 2.0), int(self.target_size[1] / 2.0))
            [x, y] = np.meshgrid(range(-radius[0], radius[0] + 1), range(-radius[1], radius[1] + 1))

            sigma = self.target_size[0] / 6
            gauss_img = np.exp(-(np.power(x, 2) + np.power(y, 2)) / (2 * np.power(sigma, 2)))

            if np.mod(self.target_size[0], 2) == 0:
                gauss_img = gauss_img[:-1]
            if np.mod(self.target_size[1], 2) == 0:
                gauss_img = gauss_img[:, :-1]
            gauss_img = gauss_img / np.max(gauss_img)


            img0 *= gauss_img
            img1 *= gauss_img

        img0 = (img0 + 1) / 2
        img1 = (img1 + 1) / 2

        # we simulate the illumination with addition
        if self.contrast_space in ['grey', 'rgb', 'lum', 'lum_ycc']:
            img0 += self.illuminant
            img1 += self.illuminant

        if self.colour_space != 'grey':
            img0 = np.repeat(img0[:, :, np.newaxis], 3, axis=2)
            img1 = np.repeat(img1[:, :, np.newaxis], 3, axis=2)

            # if rg or yb change only the luminance level
            if self.contrast_space in ['rg', 'yb', 'rg_ycc', 'yb_ycc']:
                img0[:, :, 0] = (0.5 + self.illuminant)
                img1[:, :, 0] = (0.5 + self.illuminant)

            if self.contrast_space == 'yb_ycc':
                img0[:, :, 2] = 0.5
                img0 = colour_spaces.ycc012rgb01(img0)
                img1[:, :, 2] = 0.5
                img1 = colour_spaces.ycc012rgb01(img1)
            elif self.contrast_space == 'rg_ycc':
                img0[:, :, 1] = 0.5
                img0 = colour_spaces.ycc012rgb01(img0)
                img1[:, :, 1] = 0.5
                img1 = colour_spaces.ycc012rgb01(img1)
            elif self.contrast_space == 'yb':
                img0[:, :, 1] = 0.5
                img0 = colour_spaces.dkl012rgb01(img0)
                img1[:, :, 1] = 0.5
                img1 = colour_spaces.dkl012rgb01(img1)
            elif self.contrast_space == 'rg':
                img0[:, :, 2] = 0.5
                img0 = colour_spaces.dkl012rgb01(img0)
                img1[:, :, 2] = 0.5
                img1 = colour_spaces.dkl012rgb01(img1)
            elif self.contrast_space == 'lum':
                # this is really not necessary, but just for the sake of floating point
                img0[:, :, [1, 2]] = 0.5
                img0 = colour_spaces.dkl012rgb01(img0)
                img1[:, :, [1, 2]] = 0.5
                img1 = colour_spaces.dkl012rgb01(img1)
            elif self.contrast_space not in ['rgb', 'lum_ycc']:
                sys.exit('Contrast %s not supported' % self.contrast_space)

        if 'grey' not in self.colour_space and self.vision_type != 'trichromat':
            dkl0 = colour_spaces.rgb2dkl(img0)
            dkl1 = colour_spaces.rgb2dkl(img1)
            if self.vision_type == 'dichromat_rg':
                dkl0[:, :, 1] = 0
                dkl1[:, :, 1] = 0
            elif self.vision_type == 'dichromat_yb':
                dkl0[:, :, 2] = 0
                dkl1[:, :, 2] = 0
            elif self.vision_type == 'monochromat':
                dkl0[:, :, [1, 2]] = 0
                dkl1[:, :, [1, 2]] = 0
            else:
                sys.exit('Vision type %s not supported' % self.vision_type)
            img0 = colour_spaces.dkl2rgb01(dkl0)
            img1 = colour_spaces.dkl2rgb01(dkl1)

        if self.post_transform is not None:
            img0, img1 = self.post_transform([img0, img1])
        img_out, contrast_target = _two_pairs_stimuli(img0, img1, contrast0, contrast1, self.p)

        sf_base = (self.target_size[0] * 0.5) / np.pi
        sf = int(np.round(sf_base / lambda_wave))
        item_settings = np.array([contrast0, sf, theta, rho, self.p])

        if self.grating_detector:
            return img_out[contrast_target], contrast_target, item_settings
        else:
            return img_out[0], img_out[1], contrast_target, item_settings

    def __len__(self):
        return self.samples


class GratingImagesOdd4(GratingImages):
    def __getitem__(self, item):
        img0, img1, contrast_target, item_settings = super().__getitem__(item // 4)
        target = np.mod(item, 4)
        imgs = []
        for i in range(4):
            imgs.append(img1.clone() if i == target else img0.clone())
        item_settings[-1] = target
        odd_class = 0
        return *imgs, target, odd_class, item_settings

    def __len__(self):
        return super().__len__() * 4


def train_set(db, target_size, preprocess, extra_transformation=None, **kwargs):
    if extra_transformation is None:
        extra_transformation = []
    if kwargs['train_params'] is None:
        shared_pre_transforms = [
            *extra_transformation,
            cv2_transforms.RandomHorizontalFlip(),
        ]
    else:
        shared_pre_transforms = [*extra_transformation]
    shared_post_transforms = dataset_utils.post_transform(*preprocess)
    if db in NATURAL_DATASETS:
        # if train params are passed don't use any random processes
        if kwargs['train_params'] is None:
            scale = (0.8, 1.0)
            size_transform = cv2_transforms.RandomResizedCrop(target_size, scale=scale)
            pre_transforms = [size_transform, *shared_pre_transforms]
        else:
            pre_transforms = [
                *dataset_utils.pre_transform_eval(target_size),
                *shared_pre_transforms
            ]
        post_transforms = [*shared_post_transforms]
        return _natural_dataset(db, 'train', pre_transforms, post_transforms, **kwargs)
    elif db in ['gratings']:
        return _get_grating_dataset(
            shared_pre_transforms, shared_post_transforms, target_size, **kwargs
        )
    return None


def validation_set(db, target_size, preprocess, extra_transformation=None, **kwargs):
    if extra_transformation is None:
        extra_transformation = []
    shared_pre_transforms = [*extra_transformation]
    shared_post_transforms = dataset_utils.post_transform(*preprocess)
    if db in NATURAL_DATASETS:
        pre_transforms = [
            *dataset_utils.pre_transform_eval(target_size),
            *shared_pre_transforms
        ]
        post_transforms = [*shared_post_transforms]
        return _natural_dataset(db, 'validation', pre_transforms, post_transforms, **kwargs)
    elif db in ['gratings']:
        return _get_grating_dataset(
            shared_pre_transforms, shared_post_transforms, target_size, **kwargs
        )
    return None


def test_set_odd4(target_size, preprocess, test_samples, extra_transformation=None, **kwargs):
    if extra_transformation is None:
        extra_transformation = []
    torch_pre_transforms = torch_transforms.Compose([*extra_transformation])
    torch_post_transforms = torch_transforms.Compose(dataset_utils.post_transform(*preprocess))
    afc_kwargs = {
        'pre_transform': torch_pre_transforms,
        'post_transform': torch_post_transforms,
        **kwargs
    }
    return GratingImagesOdd4(samples=test_samples, afc_kwargs=afc_kwargs, target_size=target_size)


def _natural_dataset(db, which_set, pre_transforms, post_transforms, data_dir, **kwargs):
    torch_pre_transforms = torch_transforms.Compose(pre_transforms)
    torch_post_transforms = torch_transforms.Compose(post_transforms)
    afc_kwargs = {
        'pre_transform': torch_pre_transforms,
        'post_transform': torch_post_transforms,
        **kwargs
    }
    if db == 'imagenet':
        natural_kwargs = {'root': os.path.join(data_dir, which_set)}
        current_db = ImageFolder(afc_kwargs, natural_kwargs)
    elif db == 'land':
        natural_kwargs = {'root': os.path.join(data_dir, 'Images')}
        current_db = ImageFolder(afc_kwargs, natural_kwargs)
    elif db == 'celeba':
        split = 'test' if which_set == 'validation' else 'train'
        natural_kwargs = {'root': data_dir, 'split': split}
        current_db = CelebA(afc_kwargs, natural_kwargs)
    elif db == 'bw':
        shape_kwargs = {'root': data_dir, 'background': 128}
        current_db = BinaryShapes(afc_kwargs, shape_kwargs)
    else:
        sys.exit('Dataset %s is not supported!' % db)
    return current_db


def _get_grating_dataset(pre_transforms, post_transforms, target_size, data_dir, **kwargs):
    torch_pre_transforms = torch_transforms.Compose(pre_transforms)
    torch_post_transforms = torch_transforms.Compose(post_transforms)
    afc_kwargs = {
        'pre_transform': torch_pre_transforms,
        'post_transform': torch_post_transforms,
        **kwargs
    }
    return GratingImages(samples=data_dir, afc_kwargs=afc_kwargs, target_size=target_size)
