"""
Training datasets of odd-one-out task across several visual features.
"""

import sys

import numpy as np
import random

import torch
from torch.utils import data as torch_data
import torchvision.transforms as torch_transforms

from . import dataset_utils, imutils, pattern_bank, polygon_bank


def _rnd_scale(size, scale):
    return int(size * np.random.uniform(*scale))


def draw_polygon_params(img, shape_params, colour, texture):
    draw_params = {'color': colour, 'thickness': -1 if texture['fun'] == 'filled' else 1}
    img = polygon_bank.draw(img, shape_params, **draw_params)
    if texture['fun'] == 'filled':
        return img

    draw_params['color'] = (1, 1, 1)
    draw_params['thickness'] = -1
    shape_mask = polygon_bank.draw(np.zeros(img.shape[:2], np.uint8), shape_params, **draw_params)

    texture_img = pattern_bank.__dict__[texture['fun']](img.shape[:2], **texture['params'])
    shape_mask[shape_mask == 1] = texture_img[shape_mask == 1]
    img[shape_mask == 1] = colour
    return img


def _global_img_processing(img, contrast):
    fun, amount = contrast
    fun = imutils.adjust_gamma if fun == 'gamma' else imutils.adjust_contrast
    return fun(img, amount)


def _make_img_on_bg(stimuli):
    shape_params = polygon_bank.handle_shape(stimuli)
    img_in = _global_img_processing(stimuli.background.copy(), stimuli.contrast)
    srow, scol = dataset_utils.relative_place(stimuli.canvas, img_in.shape, stimuli.position)
    img_out = dataset_utils.crop_fg_from_bg(img_in, stimuli.canvas, srow, scol)
    if stimuli.fg is not None:
        bg_lum, alpha = stimuli.fg
        img_out = (1 - alpha) * _fg_img(bg_lum, img_in, stimuli.canvas) + alpha * img_out
    img_out = draw_polygon_params(img_out, shape_params, stimuli.colour, stimuli.texture)
    return dataset_utils.merge_fg_bg_at_loc(img_in, img_out, srow, scol)


def _make_common_imgs(stimuli, num_imgs):
    imgs = []
    for i in range(num_imgs - 1):
        stimuli.common_settings(i)
        imgs.append(_make_img_on_bg(stimuli))
    return imgs


def _fg_img(fg_type, bg_img, fg_size):
    if fg_type is None:
        fg_img = bg_img.copy()
    elif fg_type in ['rnd_img', 'uniform_achromatic'] or type(fg_type) == int:
        fg_img = dataset_utils.background_img(fg_type, fg_size)
        fg_img = (fg_img * 255).astype('uint8')
    else:
        sys.exit('Unsupported feature type %s' % fg_type)
    return fg_img


def _random_canvas(img_size, fg_paths, fg_scale):
    fg_type = np.random.choice(fg_paths)
    if fg_type == 'uniform_achromatic':
        fg_type = (dataset_utils.randint(0, 256), np.random.uniform(0.5))

    # creating a random size for the canvas image
    canvas_size = (_rnd_scale(img_size[0], fg_scale), _rnd_scale(img_size[1], fg_scale))
    return fg_type, canvas_size


def create_texture(texture=None):
    fun = np.random.choice(pattern_bank.__all__) if texture is None else texture
    params = dict()
    thickness = 1
    if fun in ['wave', 'herringbone', 'diamond']:
        params['height'] = dataset_utils.randint(3, 6)
        params['length'] = thickness
        if fun == 'wave':
            params['gap'] = 0
    elif fun in ['grid', 'brick']:
        params['gaps'] = [dataset_utils.randint(2, 5), dataset_utils.randint(2, 5)]
        params['thicknesses'] = (thickness, thickness)
    elif fun == 'line':
        params['gap'] = dataset_utils.randint(2, 4)
        params['thickness'] = thickness
    return {'fun': fun, 'params': params}


def _rnd_position(stimuli):
    if stimuli.unique_feature in ['rotation', 'size']:
        pos1 = (0.5, 0.5)
        pos2 = pos1
    else:
        pos_pairs = dataset_utils.shuffle([(0, 0), (0, 1), (1, 0), (1, 1)])
        pos1, pos2 = pos_pairs[0], pos_pairs[1]
    return [pos1, pos2]


def _rnd_size(*_args, magnitude_range=(0.6, 0.8)):
    return dataset_utils.shuffle([0, np.random.uniform(*magnitude_range)])


def _rnd_symmetry(*_args):
    symmetrical = np.random.choice(['h', 'v', 'both'])
    non_symmetrical = np.random.choice(['h', 'v', 'none']) if symmetrical == 'both' else 'none'
    return dataset_utils.shuffle([non_symmetrical, symmetrical])


def _rnd_rotation(stimuli):
    angle1 = dataset_utils.randint(3, 12)
    if stimuli.shape['name'] in ['regular6']:
        rot_angles = np.arange(15, 46, 15)
    else:
        rot_angles = np.arange(15, 76, 15)
    angle2 = angle1 + np.random.choice(rot_angles)
    return dataset_utils.shuffle([np.deg2rad(angle1), np.deg2rad(angle2)])


def _rnd_shape(stimuli):
    unique_feature, canvas = stimuli.unique_feature, stimuli.canvas
    if unique_feature == 'symmetry':
        symmetries = stimuli.__getattribute__('rnd_symmetry')
        set1, set2 = [list(polygon_bank.SHAPES_SYMMETRY[sym].keys()) for sym in symmetries]
        poly1 = np.random.choice(set1)
        name1 = np.random.choice(polygon_bank.SHAPES_SYMMETRY[symmetries[0]][poly1])
        if poly1 in set2:  # only one shape from a family to avoid ambiguity of shape-uniqueness
            set2.remove(poly1)
        polys2 = np.random.choice(set2, size=stimuli.num_commons, replace=False)
        names2 = [
            np.random.choice(polygon_bank.SHAPES_SYMMETRY[symmetries[1]][poly]) for poly in polys2]
    else:
        symmetries = (stimuli.symmetry, stimuli.symmetry)
        if stimuli.symmetry == 'n/a':
            if unique_feature == 'rotation':
                polygons = polygon_bank.SHAPES_ORIENTATION
            else:
                polygons = polygon_bank.SHAPES
        else:
            polygons = [
                v for vals in polygon_bank.SHAPES_SYMMETRY[stimuli.symmetry].values() for v in vals
            ]
        polygons = np.array(polygons, dtype=object)
        if len(polygons.shape) > 1:
            s1, s2 = np.random.choice(np.arange(polygons.shape[0]), size=2, replace=False)
            set1, set2 = list(polygons[s1]), list(polygons[s2])
        else:
            set1, set2 = np.random.choice(polygons, size=2, replace=False)
        name1 = np.random.choice(set1) if type(set1) is list else set1
        names2 = [np.random.choice(set2)] if type(set2) is list else [set2]
    shape1 = {'name': name1, 'kwargs': polygon_bank.generate_polygons(name1, canvas, symmetries[0])}
    shape2 = [{'name': name2, 'kwargs': polygon_bank.generate_polygons(
        name2, canvas, symmetries[1])} for name2 in names2]
    return [shape1, *shape2]


def _rnd_texture(*_args):
    textures = pattern_bank.__all__.copy()
    set1, set2 = np.random.choice(textures, size=2, replace=False)
    return [create_texture(set1), create_texture(set2)]


def _rnd_colour(*_args):
    return dataset_utils.unique_colours(2)


def _rnd_contrast(*_args):
    fun = random.choice(['gamma', 'michelson'])
    amount = (0.3, 0.7) if fun == 'michelson' else random.choice([(0.3, 0.7), (1.5, 2.5)])
    return dataset_utils.shuffle([(fun, 1), (fun, np.random.uniform(*amount))])


def _rnd_background(stimuli):
    bg_db, bg_transform, item1 = stimuli.bg_loader
    bg_imgs = [bg_db.__getitem__(item1)]
    if 'background' in [*stimuli.paired_attrs, stimuli.unique_feature]:
        item2 = 0 if (item1 + 1) == bg_db.__len__() else item1 + 1
        bg_imgs.append(bg_db.__getitem__(item2))
    if bg_transform is not None:
        bg_imgs = [bg_transform(bg_img) for bg_img in bg_imgs]
    return bg_imgs


class StimuliSettings:

    def __init__(self, fg, canvas, bg_loader, features=None, **kwargs):
        self.features_pool = {
            'symmetry': {
                'pair': ['position', 'contrast', 'colour', 'texture', 'background'],
            },
            'rotation': {
                'pair': ['contrast', 'colour', 'texture', 'background'],
            },
            'size': {
                'pair': ['contrast', 'colour', 'texture', 'background'],
            },
            'colour': {
                'pair': ['position', 'contrast', 'shape', 'texture', 'background'],
            },
            'shape': {
                'pair': ['position', 'contrast', 'colour', 'texture', 'background'],
            },
            'texture': {
                'pair': ['position', 'contrast', 'colour', 'shape', 'background'],
            },
            'background': {
                'pair': ['position', 'contrast', 'colour', 'shape', 'texture'],
            },
            'contrast': {
                'pair': ['position', 'background', 'colour', 'shape', 'texture'],
            },
            'position': {
                'pair': ['background', 'contrast', 'colour', 'shape', 'texture'],
            }
        }

        self.features_names = list(self.features_pool.keys()) if features is None else features
        self.unique_feature = np.random.choice(self.features_names)
        self.odd_class = self.features_names.index(self.unique_feature)
        self.feature_settings = self.set_settings()
        self.num_commons = 3
        self.paired_attrs = self.feature_settings['pair'][:self.num_commons]
        self.exclusive_attrs = ['shape'] if self.unique_feature == 'symmetry' else []

        self.fg = None if self.unique_feature == 'background' else fg
        self.canvas = canvas
        self.bg_loader = bg_loader

        self.background = kwargs.get("background", None)
        self.contrast = kwargs.get("contrast", None)
        self.colour = kwargs.get("colour", None)
        self.texture = kwargs.get("texture", None)
        self.size = kwargs.get("size", 0)
        self.rotation = kwargs.get("rotation", 0)
        self.position = kwargs.get("position", None)
        # if shape is the unique feature, we should make sure the symmetry is identical in all
        default_symmetry = _rnd_symmetry()[0] if self.unique_feature == 'shape' else "n/a"
        self.symmetry = kwargs.get("symmetry", default_symmetry)
        default_shape = _rnd_shape(self)[0] if self.unique_feature == 'rotation' else None
        self.shape = kwargs.get("shape", default_shape)

        self.fill_in_paired_settings()

    def set_settings(self):
        settings = {'unique': self.unique_feature}
        for key, val in self.features_pool[self.unique_feature].items():
            settings[key] = val.copy()
        random.shuffle(settings['pair'])
        return settings

    def fill_in_paired_settings(self):
        for attr in [self.unique_feature, *self.paired_attrs, *self.exclusive_attrs]:
            self.__setattr__('rnd_%s' % attr, globals()['_rnd_%s' % attr](self))
            self.__setattr__(attr, self.__getattribute__('rnd_%s' % attr)[0])

        for attr in self.features_pool.keys():
            if self.__getattribute__(attr) is None:
                self.__setattr__(attr, globals()['_rnd_%s' % attr](self)[0])

    def common_settings(self, item):
        unique_attr = self.unique_feature
        self.__setattr__(unique_attr, self.__getattribute__('rnd_%s' % unique_attr)[1])
        for attr in self.exclusive_attrs:
            self.__setattr__(attr, self.__getattribute__('rnd_%s' % attr)[item + 1])
        for i, attr in enumerate(self.paired_attrs):
            ind = 0 if (i % self.num_commons) == item else 1
            self.__setattr__(attr, self.__getattribute__('rnd_%s' % attr)[ind])


class OddOneOutTrain(torch_data.Dataset):

    def __init__(self, bg_loader, num_imgs, target_size, transform=None, **kwargs):
        self.bg_loader = bg_loader
        self.target_size = (target_size, target_size) if type(target_size) is int else target_size
        self.num_imgs = num_imgs
        self.transform = transform
        self.single_img = kwargs['single_img'] if 'single_img' in kwargs else None
        self.features = kwargs['features'] if 'features' in kwargs else None
        self.fg_paths = kwargs['fg_paths'] if 'fg_paths' in kwargs else []
        self.fg_paths = [*self.fg_paths, None, 'uniform_achromatic']  # 'rnd_img'
        self.fg_scale = kwargs['fg_scale'] if 'fg_scale' in kwargs else (0.50, 1.00)

    def __getitem__(self, item):
        # drawing the foreground content
        fg, canvas_size = _random_canvas(self.target_size, self.fg_paths, self.fg_scale)
        stimuli = StimuliSettings(fg, canvas_size, (*self.bg_loader, item), self.features)
        odd_img = _make_img_on_bg(stimuli)
        common_imgs = _make_common_imgs(stimuli, self.num_imgs)
        imgs = [odd_img, *common_imgs]

        if self.transform is not None:
            imgs = self.transform(imgs)

        inds = dataset_utils.shuffle(list(np.arange(0, self.num_imgs)))
        # the target is always added the first element in the imgs list
        target = inds.index(0)
        imgs = [imgs[i] for i in inds]
        if self.single_img is not None:
            imgs = [torch.cat(
                [torch.cat([imgs[0], imgs[1]], dim=2), torch.cat([imgs[2], imgs[3]], dim=2)], dim=1
            )]
        return *imgs, target, stimuli.odd_class

    def __len__(self):
        return self.bg_loader[0].__len__()


def oddx_bg_folder(root, num_imgs, target_size, preprocess, scale=(0.5, 1.0), **kwargs):
    single_img = kwargs['single_img'] if 'single_img' in kwargs else None
    if single_img is not None:
        # FIXME: hardcoded here only for 224 224!
        target_size = (target_size // 2, target_size // 2)
    bg_transform = torch_transforms.Compose(dataset_utils.pre_transform_train(target_size, scale))
    transform = torch_transforms.Compose(dataset_utils.post_transform(*preprocess))
    bg_db = dataset_utils.NoTargetFolder(root, loader=dataset_utils.cv2_loader_3chns)
    bg_loader = (bg_db, bg_transform)
    return OddOneOutTrain(bg_loader, num_imgs, target_size, transform, **kwargs)
