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


def _random_length(length, polygon, scale=(0.2, 0.8), min_length=5):
    # ellipse and circle are defined by radius of their axis
    if polygon in polygon_bank.SHAPES_OVAL:
        length = int(length / 2)

    if polygon in ['ellipse']:
        length = (length, max(int(length * np.random.uniform(*scale)), min_length))
    return length


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
    srow, scol = dataset_utils.random_place(stimuli.canvas, img_in.shape)
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


def _choose_rand_remove(elements):
    element = np.random.choice(elements)
    elements.remove(element)
    return element


def _fg_img(fg_type, bg_img, fg_size):
    if fg_type is None:
        fg_img = bg_img.copy()
    elif fg_type in ['rnd_img', 'rnd_uniform'] or type(fg_type) == int:
        fg_img = dataset_utils.background_img(fg_type, fg_size)
        fg_img = (fg_img * 255).astype('uint8')
    else:
        sys.exit('Unsupported feature type %s' % fg_type)
    return fg_img


def _random_canvas(img_size, fg_paths, fg_scale):
    fg_type = np.random.choice(fg_paths)
    if fg_type == 'rnd_uniform':
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


def create_shape(polygon, canvas):
    length = np.minimum(canvas[0], canvas[1]) / 2
    kwargs = dict()
    half_canvas = imutils.centre_pixel(canvas)
    ref_pt = polygon_bank.ref_point(length, polygon, half_canvas)
    if polygon in polygon_bank.SHAPES_OVAL:
        kwargs['ref_pt'] = ref_pt
        kwargs['length'] = _random_length(length, polygon)
    elif polygon in polygon_bank.SHAPES_QUADRILATERAL:
        kwargs = polygon_bank.generate_quadrilaterals(polygon, half_canvas)
        ref_pt = kwargs['ref_pt']
    elif polygon == 'triangle':
        pt1 = ref_pt
        lside = np.random.choice([0, 1])
        sx = length if lside == 0 else dataset_utils.randint(4, 7)
        sy = length if lside == 1 else dataset_utils.randint(4, 7)
        pt2 = (pt1[0] + sx, pt1[1] + sy)
        sx = length if lside == 1 else dataset_utils.randint(4, 7)
        sy = length if lside == 0 else dataset_utils.randint(4, 7)
        pt3 = (pt1[0] + sx, pt1[1] + sy)
        kwargs['pts'] = [np.array([pt1, pt2, pt3]).astype('int')]
    kwargs['ref_pt'] = int(ref_pt[0] + half_canvas[1] / 2), int(ref_pt[1] + half_canvas[0] / 2)
    return kwargs


def _rnd_size(*_args, magnitude_range=(0.6, 0.8)):
    return dataset_utils.shuffle([0, np.random.uniform(*magnitude_range)])


def _rnd_symmetry(*_args):
    symmetrical = np.random.choice(['h', 'v', 'both'])
    non_symmetrical = np.random.choice(['h', 'v', 'none']) if symmetrical == 'both' else 'none'
    return dataset_utils.shuffle([non_symmetrical, symmetrical])


def _rnd_rotation(*_args, rot_angles=None):
    if rot_angles is None:
        rot_angles = [15, 30, 45, 60, 75, 90]
    angle1 = dataset_utils.randint(0, 90)
    angle2 = angle1 + np.random.choice(rot_angles)
    return [np.deg2rad(angle1), np.deg2rad(angle2)]


def _rnd_shape(stimuli):
    polygons = polygon_bank.SHAPES.copy()
    if stimuli.unique_feature == 'rotation':
        polygons.remove('circle')
    shape1_name = _choose_rand_remove(polygons)
    if shape1_name == 'square':
        polygons.remove('rectangle')
    elif shape1_name == 'rectangle':
        polygons.remove('square')
    elif shape1_name == 'circle':
        polygons.remove('ellipse')
    elif shape1_name == 'ellipse' and 'circle' in polygons:
        polygons.remove('circle')
    shape2_name = np.random.choice(polygons)
    shape1 = {'name': shape1_name, 'kwargs': create_shape(shape1_name, stimuli.canvas)}
    shape2 = {'name': shape2_name, 'kwargs': create_shape(shape2_name, stimuli.canvas)}
    return [shape1, shape2]


def _rnd_texture(*_args):
    textures = pattern_bank.__all__.copy()
    texture1 = create_texture(_choose_rand_remove(textures))
    texture2 = create_texture(np.random.choice(textures))
    return [texture1, texture2]


def _rnd_colour(*_args):
    colour1 = [random.randint(0, 255) for _ in range(3)]
    while True:
        colour2 = [random.randint(0, 255) for _ in range(3)]
        if colour1 != colour2:
            return [colour1, colour2]


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
        # 'spatial_pos', 'material'
        self.features_pool = {
            'symmetry': {
                'pair': ['contrast', 'colour', 'texture', 'background'],
            },
            'rotation': {
                'pair': ['contrast', 'colour', 'texture', 'background'],
            },
            'size': {
                'pair': ['contrast', 'colour', 'texture', 'background'],
            },
            'colour': {
                'pair': ['contrast', 'shape', 'texture', 'background'],
            },
            'shape': {
                'pair': ['contrast', 'colour', 'texture', 'background'],
            },
            'texture': {
                'pair': ['contrast', 'colour', 'shape', 'background'],
            },
            'background': {
                'pair': ['contrast', 'colour', 'shape', 'texture'],
            },
            'contrast': {
                'pair': ['background', 'colour', 'shape', 'texture'],
            }
        }

        self.features_names = list(self.features_pool.keys()) if features is None else features
        self.unique_feature = np.random.choice(self.features_names)
        self.odd_class = self.features_names.index(self.unique_feature)
        self.feature_settings = self.set_settings()
        self.num_commons = 3
        self.paired_attrs = self.feature_settings['pair'][:self.num_commons]
        if self.unique_feature == 'symmetry':
            self.paired_attrs = ['shape', *self.paired_attrs]

        self.fg = None if self.unique_feature == 'background' else fg
        self.canvas = canvas
        self.bg_loader = bg_loader

        self.background = kwargs.get("background", None)
        self.contrast = kwargs.get("contrast", None)
        self.shape = kwargs.get("shape", None)
        self.colour = kwargs.get("colour", None)
        self.texture = kwargs.get("texture", None)
        self.size = kwargs.get("size", 0)
        self.rotation = kwargs.get("rotation", 0)
        self.symmetry = kwargs.get("symmetry", "n/a")

        self.fill_in_paired_settings()

    def set_settings(self):
        settings = {'unique': self.unique_feature}
        for key, val in self.features_pool[self.unique_feature].items():
            settings[key] = val.copy()
        random.shuffle(settings['pair'])
        return settings

    def fill_in_paired_settings(self):
        for attr in [*self.paired_attrs, self.unique_feature]:
            self.__setattr__('rnd_%s' % attr, globals()['_rnd_%s' % attr](self))
            self.__setattr__(attr, self.__getattribute__('rnd_%s' % attr)[0])

        for attr in self.features_pool.keys():
            if self.__getattribute__(attr) is None:
                self.__setattr__(attr, globals()['_rnd_%s' % attr](self)[0])

        # if shape is the unique feature, we should make sure the symmetry is identical in all
        if self.unique_feature == 'shape':
            self.symmetry = _rnd_symmetry()[0]
            for shape in self.rnd_shape:
                shape['kwargs'] = polygon_bank.handle_symmetry(
                    self.symmetry, shape['kwargs'], shape['name'], self.canvas
                )

    def common_settings(self, item):
        unique_attr = self.unique_feature
        self.__setattr__(unique_attr, self.__getattribute__('rnd_%s' % unique_attr)[1])
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
        self.fg_paths = [*self.fg_paths, None, 'rnd_uniform']  # 'rnd_img'
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
    bg_db = dataset_utils.NoTargetFolder(root, loader=dataset_utils.cv2_loader)
    bg_loader = (bg_db, bg_transform)
    return OddOneOutTrain(bg_loader, num_imgs, target_size, transform, **kwargs)
