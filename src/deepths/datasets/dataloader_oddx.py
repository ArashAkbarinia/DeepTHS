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


def _randint(low, high):
    low, high = int(low), int(high)
    return low if low >= high else np.random.randint(low, high)


def _rnd_scale(size, scale):
    return int(size * np.random.uniform(*scale))


def _random_length(length, polygon, scale=(0.2, 0.8), min_length=2):
    # ellipse and circle are defined by radius of their axis
    if polygon in polygon_bank.CV2_OVAL_SHAPES:
        length = int(length / 2)

    if polygon in ['ellipse', 'rectangle']:
        length = (length, max(int(length * np.random.uniform(*scale)), min_length))
    return length


def _ref_point(length, polygon, img_size):
    diff = min(img_size[0], img_size[1]) - length - 2
    if polygon in polygon_bank.CV2_OVAL_SHAPES:
        cy, cx = imutils.centre_pixel(img_size)
        if diff <= 0:
            ref_pt = (cx, cy)
        else:
            diff = diff // 2
            ref_pt = (_randint(cx - diff, cx + diff), _randint(cy - diff, cy + diff))
    elif polygon in ['square', 'rectangle']:
        ref_pt = (_randint(0, diff), _randint(0, diff))
    elif polygon in ['triangle']:
        ymax, xmax = img_size[:2]
        ref_pt = (_randint(0, xmax - length), _randint(0, ymax - length))
    else:
        sys.exit('Unsupported polygon to draw: %s' % polygon)
    return ref_pt


def _ref_point_rotation(length, polygon, img_size):
    half_size = imutils.centre_pixel(img_size)
    ref_pt = _ref_point(length, polygon, half_size)
    return int(ref_pt[0] + half_size[1] / 2), int(ref_pt[1] + half_size[0] / 2)


def draw_polygon_params(draw_fun, img, params, colour, texture):
    params['color'] = colour
    params['thickness'] = -1 if texture['fun'] == 'filled' else 1
    img = draw_fun(img, **params)
    if texture['fun'] == 'filled':
        return img

    params['color'] = (1, 1, 1)
    params['thickness'] = -1
    shape_mask = draw_fun(np.zeros(img.shape[:2], np.uint8), **params)

    texture_img = pattern_bank.__dict__[texture['fun']](img.shape[:2], **texture['params'])
    shape_mask[shape_mask == 1] = texture_img[shape_mask == 1]
    img[shape_mask == 1] = colour
    return img


def _enlarge(value, magnitude, ref=None):
    int_in = False
    if not hasattr(value, '__len__'):
        value, int_in = (value,), True
    if ref is None:
        ref = (0,) * len(value)
    value_centered = np.array([v - r for v, r in zip(value, ref)])
    value_magnified = (value_centered + value_centered * magnitude).astype(int)
    return value_magnified[0] if int_in else tuple(value_magnified)


def _enlarge_polygon(magnitude, shape_params, stimuli):
    if magnitude == 0:
        return shape_params
    shape, out_size = stimuli.shape['name'], stimuli.canvas
    length = np.minimum(stimuli.canvas[0], stimuli.canvas[1]) / 2
    ref_pt = _ref_point(_enlarge(length, magnitude), shape, out_size)
    shape_params = shape_params.copy()
    if shape in polygon_bank.CV2_OVAL_SHAPES:
        shape_params['center'] = ref_pt
        oval_length = 'radius' if shape == 'circle' else 'axes'
        shape_params[oval_length] = _enlarge(shape_params[oval_length], magnitude)
    else:
        old_pts = shape_params['pts'][0]
        pt1 = ref_pt
        other_pts = [_enlarge(pt, magnitude, old_pts[0]) for pt in old_pts[1:]]
        other_pts = [(pt[0] + pt1[0], pt[1] + pt1[1]) for pt in other_pts]
        shape_params['pts'] = [np.array([pt1, *other_pts])]
    return shape_params


def _make_img_on_bg(stimuli, shape_draw):
    img_in = _global_img_processing(stimuli.background.copy(), stimuli.contrast)
    srow, scol = dataset_utils.random_place(stimuli.canvas, img_in.shape)
    img_out = dataset_utils.crop_fg_from_bg(img_in, stimuli.canvas, srow, scol)
    if stimuli.fg is not None:
        bg_lum, alpha = stimuli.fg
        img_out = (1 - alpha) * _fg_img(bg_lum, img_in, stimuli.canvas) + alpha * img_out
    draw_fun, shape_params = shape_draw
    img_out = draw_polygon_params(draw_fun, img_out, shape_params, stimuli.colour, stimuli.texture)
    return dataset_utils.merge_fg_bg_at_loc(img_in, img_out, srow, scol)


def _make_img(stimuli):
    shape = stimuli.shape
    shape['kwargs']['rotation'] = stimuli.rotation
    draw_fun, shape_params = polygon_bank.polygon_params(shape['name'], **shape['kwargs'])
    shape_params = _enlarge_polygon(stimuli.size, shape_params, stimuli)
    return _make_img_on_bg(stimuli, [draw_fun, shape_params])


def _make_common_imgs(stimuli, num_imgs):
    imgs = []
    for i in range(num_imgs - 1):
        stimuli.common_settings(i)
        imgs.append(_make_img(stimuli))
    return imgs


def _global_img_processing(img, contrast):
    img = imutils.adjust_contrast(img, contrast)
    return img


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
        fg_type = (_randint(0, 256), np.random.uniform(0.5))

    # creating a random size for the canvas image
    canvas_size = (_rnd_scale(img_size[0], fg_scale), _rnd_scale(img_size[1], fg_scale))
    return fg_type, canvas_size


def create_texture(texture=None):
    fun = np.random.choice(pattern_bank.__all__) if texture is None else texture
    params = dict()
    thickness = 1
    if fun in ['wave', 'herringbone', 'diamond']:
        params['height'] = _randint(3, 6)
        params['length'] = thickness
        if fun == 'wave':
            params['gap'] = 0
    elif fun in ['grid', 'brick']:
        params['gaps'] = [_randint(2, 5), _randint(2, 5)]
        params['thicknesses'] = (thickness, thickness)
    elif fun == 'line':
        params['gap'] = _randint(2, 4)
        params['thickness'] = thickness
    return {'fun': fun, 'params': params}


def create_shape(polygon, stimuli):
    length = np.minimum(stimuli.canvas[0], stimuli.canvas[1]) / 2
    kwargs = dict()
    ref_pt = _ref_point_rotation(length, polygon, stimuli.canvas)
    if polygon in polygon_bank.CV2_OVAL_SHAPES:
        kwargs['ref_pt'] = ref_pt
        kwargs['length'] = _random_length(length, polygon)
    elif polygon in ['square', 'rectangle']:
        rnd_length = _random_length(length, polygon)
        if polygon == 'square':
            rnd_length = (rnd_length, rnd_length)
        pt1 = ref_pt
        pt2 = (pt1[0] + 0, pt1[1] + rnd_length[1])
        pt3 = (pt1[0] + rnd_length[0], pt1[1] + rnd_length[1])
        pt4 = (pt1[0] + rnd_length[0], pt1[1] + 0)
        kwargs['pts'] = [np.array([pt1, pt2, pt3, pt4])]
    elif polygon == 'triangle':
        pt1 = ref_pt
        lside = np.random.choice([0, 1])
        sx = length if lside == 0 else _randint(0, length)
        sy = length if lside == 1 else _randint(0, length)
        pt2 = (pt1[0] + sx, pt1[1] + sy)
        sx = length if lside == 1 else _randint(0, length)
        sy = length if lside == 0 else _randint(0, length)
        pt3 = (pt1[0] + sx, pt1[1] + sy)
        kwargs['pts'] = [np.array([pt1, pt2, pt3])]
    return kwargs


def _rnd_size(*_args, magnitude_range=(0.6, 0.8)):
    size_magnitude = [0, np.random.uniform(*magnitude_range)]
    random.shuffle(size_magnitude)
    return size_magnitude


def _rnd_rotation(*_args, rot_angles=None):
    if rot_angles is None:
        rot_angles = [15, 30, 45, 60, 75, 90]
    angle1 = _randint(0, 90)
    angle2 = angle1 + np.random.choice(rot_angles)
    return [np.deg2rad(angle1), np.deg2rad(angle2)]


def _rnd_shape(stimuli):
    polygons = polygon_bank.SHAPES.copy()
    if stimuli.unique_feature == 'rotation':
        polygons.remove('circle')
    shape1_name = _choose_rand_remove(polygons)
    shape2_name = np.random.choice(polygons)
    shape1 = {'name': shape1_name, 'kwargs': create_shape(shape1_name, stimuli)}
    shape2 = {'name': shape2_name, 'kwargs': create_shape(shape2_name, stimuli)}
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


def _rnd_contrast(*_args, contrast_range=(0.3, 0.7)):
    contrasts = [1, np.random.uniform(*contrast_range)]
    random.shuffle(contrasts)
    return contrasts


class StimuliSettings:

    def __init__(self, fg, canvas, background, all_features, **kwargs):
        self.all_features = list(all_features.keys())
        self.unique_feature = np.random.choice(self.all_features)
        self.odd_class = self.all_features.index(self.unique_feature)
        self.fg = None if self.unique_feature == 'background' else fg
        self.canvas = canvas

        self.rnd_background = background
        self.background = self.rnd_background[0]
        self.contrast = kwargs.get("contrast", None)
        self.shape = kwargs.get("shape", None)
        self.colour = kwargs.get("colour", None)
        self.texture = kwargs.get("texture", None)
        self.size = kwargs.get("size", 0)
        self.rotation = kwargs.get("rotation", 0)

        self.paired_attrs = self.fill_in_paired_settings(all_features)

    def set_settings(self, all_features):
        settings = dict()
        for key, val in all_features[self.unique_feature].items():
            settings[key] = val.copy()
        settings['unique'] = self.unique_feature
        return settings

    def fill_in_paired_settings(self, all_features):
        settings = self.set_settings(all_features)
        for attr in [*settings['pair'], self.unique_feature]:
            if attr == 'background':
                continue
            self.__setattr__('rnd_%s' % attr, globals()['_rnd_%s' % attr](self))
            self.__setattr__(attr, self.__getattribute__('rnd_%s' % attr)[0])

        for attr in [*settings['fixed'], *self.all_features]:
            if self.__getattribute__(attr) is None:
                self.__setattr__(attr, globals()['_rnd_%s' % attr](self)[0])
        return [_choose_rand_remove(settings['pair']) for _ in range(3)]

    def common_settings(self, item):
        unique_attr = self.unique_feature
        self.__setattr__(unique_attr, self.__getattribute__('rnd_%s' % unique_attr)[1])
        for i, attr in enumerate(self.paired_attrs):
            ind = 0 if i == item else 1
            self.__setattr__(attr, self.__getattribute__('rnd_%s' % attr)[ind])


class OddOneOutTrain(torch_data.Dataset):

    def __init__(self, bg_db, num_imgs, transform=None, bg_transform=None, **kwargs):
        # 'spatial_pos', 'symmetry', 'material', 'contrast'
        self.supported_features = {
            'rotation': {
                'fixed': ['shape'],
                'pair': ['contrast', 'colour', 'texture', 'background'],
                'excluded': []
            },
            'size': {
                'fixed': ['shape'],
                'pair': ['contrast', 'colour', 'texture', 'background'],
                'excluded': []
            },
            'colour': {
                'fixed': [],
                'pair': ['contrast', 'shape', 'texture', 'background'],
                'excluded': ['rotation']
            },
            'shape': {
                'fixed': [],
                'pair': ['contrast', 'colour', 'texture', 'background'],
                'excluded': ['rotation']
            },
            'texture': {
                'fixed': [],
                'pair': ['contrast', 'colour', 'shape', 'background'],
                'excluded': ['rotation']
            },
            'background': {
                'fixed': [],
                'pair': ['contrast', 'colour', 'shape', 'texture'],
                'excluded': ['rotation']
            },
        }

        self.bg_db = bg_db
        self.num_imgs = num_imgs
        self.transform = transform
        self.bg_transform = bg_transform
        self.single_img = kwargs['single_img'] if 'single_img' in kwargs else None
        self.features = kwargs['features'] if 'features' in kwargs else self.supported_features
        self.features = [f for f in self.features if f in self.supported_features]
        self.fg_paths = kwargs['fg_paths'] if 'fg_paths' in kwargs else []
        self.fg_paths = [*self.fg_paths, None, 'rnd_uniform']  # 'rnd_img'
        self.fg_scale = kwargs['fg_scale'] if 'fg_scale' in kwargs else (0.50, 1.00)

    def __getitem__(self, item):
        bg_img1 = self.bg_db.__getitem__(item)
        item2 = 0 if (item + 1) == self.__len__() else item + 1
        bg_img2 = self.bg_db.__getitem__(item2)
        bg_imgs = [bg_img1, bg_img2]
        if self.bg_transform is not None:
            bg_imgs = [self.bg_transform(bg_img) for bg_img in bg_imgs]

        # drawing the foreground content
        fg, canvas_size = _random_canvas(bg_imgs[0].shape, self.fg_paths, self.fg_scale)
        stimuli = StimuliSettings(fg, canvas_size, bg_imgs, self.supported_features)
        odd_img = _make_img(stimuli)
        common_imgs = _make_common_imgs(stimuli, self.num_imgs)
        imgs = [odd_img, *common_imgs]

        if self.transform is not None:
            imgs = self.transform(imgs)

        inds = list(np.arange(0, self.num_imgs))
        random.shuffle(inds)
        # the target is always added the first element in the imgs list
        target = inds.index(0)
        imgs = [imgs[i] for i in inds]
        if self.single_img is not None:
            imgs = [torch.cat(
                [torch.cat([imgs[0], imgs[1]], dim=2), torch.cat([imgs[2], imgs[3]], dim=2)], dim=1
            )]
        return *imgs, target, stimuli.odd_class

    def __len__(self):
        return self.bg_db.__len__()


def oddx_bg_folder(root, num_imgs, target_size, preprocess, **kwargs):
    scale = (0.5, 1.0)
    single_img = kwargs['single_img'] if 'single_img' in kwargs else None
    if single_img is not None:
        # FIXME: hardcoded here only for 224 224!
        target_size = (112, 112)
    bg_transform = torch_transforms.Compose(dataset_utils.pre_transform_train(target_size, scale))
    transform = torch_transforms.Compose(dataset_utils.post_transform(*preprocess))
    bg_db = dataset_utils.NoTargetFolder(root, loader=dataset_utils.cv2_loader)
    return OddOneOutTrain(bg_db, num_imgs, transform, bg_transform, **kwargs)
