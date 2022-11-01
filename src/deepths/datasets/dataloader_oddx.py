"""
Training datasets of odd-one-out task across several visual features.
"""

import sys

import numpy as np
import random

import cv2

from torch.utils import data as torch_data
import torchvision.transforms as torch_transforms

from . import dataset_utils, imutils

CV2_BASIC_SHAPES = ['circle', 'ellipse', 'square', 'rectangle']
CV2_CUSTOM_SHAPES = ['triangle']


def _img_centre(img_size):
    return img_size[0] // 2, img_size[1] // 2


def _rnd_colour():
    return [random.randint(0, 255) for _ in range(3)]


def _random_length(length, polygon, scale=(0.2, 0.8), min_length=2):
    # ellipse and circle are defined by radius of their axis
    if polygon in ['circle', 'ellipse']:
        length = int(length / 2)

    if polygon in ['ellipse', 'rectangle']:
        length = (length, max(int(length * np.random.uniform(*scale)), min_length))
    return length


def _ref_point(length, polygon, img_size, thickness):
    if thickness < 0:
        thickness = 0
    min_side = min(img_size[0], img_size[1])
    diff = min_side - length - (thickness * 2)
    if polygon in ['circle', 'ellipse']:
        cy, cx = _img_centre(img_size)
        if diff <= 0:
            ref_pt = (cx, cy)
        else:
            diff = diff // 2
            ref_pt = (_randint(cx - diff, cx + diff), _randint(cy - diff, cy + diff))
    elif polygon in ['square', 'rectangle', 'triangle']:
        ref_pt = (_randint(0, diff), _randint(0, diff))
    else:
        sys.exit('Unsupported polygon to draw: %s' % polygon)
    return ref_pt


def _randint(low, high):
    return low if low >= high else np.random.randint(low, high)


def _polygon_kwargs(polygon, length, img_size, thickness):
    kwargs = dict()
    if polygon in CV2_BASIC_SHAPES:
        kwargs['ref_pt'] = _ref_point(length, polygon, img_size, thickness)
        kwargs['length'] = _random_length(length, polygon)
    elif polygon == 'triangle':
        ymax, xmax = img_size[:2]
        pt1 = (_randint(0, xmax - length), _randint(0, ymax - length))
        lside = np.random.choice([0, 1])
        sx = length if lside == 0 else _randint(0, length)
        sy = length if lside == 1 else _randint(0, length)
        pt2 = (pt1[0] + sx, pt1[1] + sy)
        sx = length if lside == 1 else _randint(0, length)
        sy = length if lside == 0 else _randint(0, length)
        pt3 = (pt1[0] + sx, pt1[1] + sy)
        kwargs['pts'] = [np.array([pt1, pt2, pt3])]
    return kwargs


def cv2_filled_polygons(img, pts, color, thickness):
    img = cv2.polylines(img, pts=pts, color=color, thickness=abs(thickness), isClosed=True)
    if thickness < 0:
        img = cv2.fillPoly(img, pts=pts, color=color)
    return img


def cv2_polygons(pts):
    return cv2_filled_polygons, {'pts': pts}


def cv2_shapes(polygon, length, ref_pt):
    if polygon == 'circle':
        params = {'center': ref_pt, 'radius': length}
        draw_fun = cv2.circle
    elif polygon == 'ellipse':
        params = {'center': ref_pt, 'axes': length, 'angle': 0, 'startAngle': 0, 'endAngle': 360}
        draw_fun = cv2.ellipse
    elif polygon == 'square':
        params = {'pt1': ref_pt, 'pt2': (ref_pt[0] + length, ref_pt[1] + length)}
        draw_fun = cv2.rectangle
    elif polygon == 'rectangle':
        params = {'pt1': ref_pt, 'pt2': (ref_pt[0] + length[0], ref_pt[1] + length[1])}
        draw_fun = cv2.rectangle
    else:
        sys.exit('Unsupported polygon to draw: %s' % polygon)
    return draw_fun, params


def polygon_params(polygon, **kwargs):
    if polygon in CV2_BASIC_SHAPES:
        draw_fun, params = cv2_shapes(polygon, **kwargs)
    elif polygon in ['triangle']:
        draw_fun, params = cv2_polygons(**kwargs)
    else:
        sys.exit('Unsupported polygon to draw: %s' % polygon)
    return draw_fun, params


def draw_polygon_params(draw_fun, img, params, thickness, colour):
    params['color'] = colour
    params['thickness'] = thickness
    return draw_fun(img, **params)


def _change_scale(value, magnitude, ref=0):
    big_or_small, size_change = magnitude
    # centre the value at zero
    value = value - ref
    return value + big_or_small * (value * size_change)


def _enlarge_polygon(magnitude, shape_params, shape, out_size):
    shape_params = shape_params.copy()
    if shape == 'circle':
        shape_params['radius'] = int(_change_scale(shape_params['radius'], magnitude))
        shape_params['center'] = (out_size[0] // 2, out_size[1] // 2)
    elif shape == 'ellipse':
        shape_params['axes'] = (
            int(_change_scale(shape_params['axes'][0], magnitude)),
            int(_change_scale(shape_params['axes'][1], magnitude)),
        )
        shape_params['center'] = (out_size[0] // 2, out_size[1] // 2)
    elif shape in ['square', 'rectangle']:
        shape_params['pt1'] = (0, 0)
        shape_params['pt2'] = (
            int(_change_scale(shape_params['pt2'][0], magnitude)),
            int(_change_scale(shape_params['pt2'][1], magnitude)),
        )
    elif shape in ['triangle']:
        old_pts = shape_params['pts'][0]
        pt1 = (0, 0)
        pt2 = (
            int(_change_scale(old_pts[1][0], magnitude, ref=old_pts[0][0])),
            int(_change_scale(old_pts[1][1], magnitude, ref=old_pts[0][1])),
        )
        pt3 = (
            int(_change_scale(old_pts[2][0], magnitude, ref=old_pts[0][0])),
            int(_change_scale(old_pts[2][1], magnitude, ref=old_pts[0][1])),
        )
        shape_params['pts'] = [np.array([pt1, pt2, pt3])]
    return shape_params


def _make_img_shape(img_in, fg_type, crop_size, contrast, thickness, colour, shape_draw):
    img_in = _global_img_processing(img_in.copy(), contrast)
    if fg_type is None:
        srow, scol = dataset_utils.random_place(crop_size, img_in.shape)
        img_out = dataset_utils.crop_fg_from_bg(img_in, crop_size, srow, scol)
    else:
        img_out = _fg_img(fg_type, img_in, crop_size)
    draw_fun, shape_params = shape_draw
    img_out = draw_polygon_params(draw_fun, img_out, shape_params, thickness, colour)
    if fg_type is None:
        img_out = dataset_utils.merge_fg_bg_at_loc(img_in, img_out, srow, scol)
    else:
        img_out = dataset_utils.merge_fg_bg(img_in, img_out, 'radnom')
    return img_out


def _make_img(img_in, contrast, shape, length, thickness, colour):
    img_out = _global_img_processing(img_in.copy(), contrast)
    img_out = _local_img_drawing(img_out, shape, length, thickness, colour)
    return img_out


def _global_img_processing(img, contrast):
    img = imutils.adjust_contrast(img, contrast)
    return img


def _local_img_drawing(img, shape, length, thickness, colour):
    shape_kwargs = _polygon_kwargs(shape, length, img.shape, thickness)
    draw_fun, params = polygon_params(shape, **shape_kwargs)
    return draw_polygon_params(draw_fun, img, params, colour, thickness)


def _rnd_contrast():
    contrasts = [1, np.random.uniform(0.3, 0.7)]
    random.shuffle(contrasts)
    return contrasts


def _rnd_scale(size, scale):
    rnd_scale = np.random.uniform(*scale)
    return int(size * rnd_scale)


def _rnd_thickness(thickness_range=(4, 7)):
    thicknesses = [-1, _randint(*thickness_range)]
    random.shuffle(thicknesses)
    return thicknesses


def _fg_img(fg_type, bg_img, fg_size):
    if fg_type is None:
        fg_img = bg_img.copy()
    elif fg_type in ['rnd_img', 'rnd_uniform']:
        fg_img = dataset_utils.background_img(fg_type, fg_size)
    else:
        sys.exit('Unsupported feature type %s' % fg_type)
    return fg_img


class OddOneOutTrain(torch_data.Dataset):

    def __init__(self, bg_db, num_imgs, transform=None, bg_transform=None, **kwargs):
        supported_features = [
            'shape', 'size', 'colour', 'texture', 'spatial_pos', 'symmetry', 'rotation',
            'material', 'contrast'
        ]

        self.bg_db = bg_db
        self.num_imgs = num_imgs
        self.transform = transform
        self.bg_transform = bg_transform
        self.features = kwargs['features'] if 'features' in kwargs else supported_features
        self.features = [f for f in self.features if f in supported_features]
        self.fg_paths = kwargs['fg_paths'] if 'fg_paths' in kwargs else []
        self.fg_paths = [*self.fg_paths, None, 'rnd_img', 'rnd_uniform']
        self.fg_scale = kwargs['fg_scale'] if 'fg_scale' in kwargs else (0.30, 0.40)

    def __getitem__(self, item):
        bg_img = self.bg_db.__getitem__(item)
        if self.bg_transform is not None:
            bg_img = self.bg_transform(bg_img)

        # selecting the unique features shared among all except one image
        unique_feature = np.random.choice(self.features)

        # drawing the foreground content
        if unique_feature == 'shape':
            imgs = self.shape_feature(bg_img)
        elif unique_feature == 'colour':
            imgs = self.colour_feature(bg_img)
        elif unique_feature == 'size':
            imgs = self.size_feature(bg_img)

        if self.transform is not None:
            imgs = self.transform(imgs)

        inds = list(np.arange(0, self.num_imgs))
        random.shuffle(inds)
        # the target is always added the first element in the imgs list
        target = inds.index(0)
        imgs = [imgs[i] for i in inds]
        return *imgs, target

    def __len__(self):
        return self.bg_db.__len__()

    def size_feature(self, img_in):
        fg_type = np.random.choice(self.fg_paths)
        # creating a random size for the odd image
        odd_size = (
            _rnd_scale(img_in.shape[0], self.fg_scale),
            _rnd_scale(img_in.shape[1], self.fg_scale)
        )

        polygons = [*CV2_BASIC_SHAPES, *CV2_CUSTOM_SHAPES]

        odd_shape = np.random.choice(polygons)
        odd_colour = _rnd_colour()
        odd_thick = _rnd_thickness()
        contrasts = _rnd_contrast()

        length = np.minimum(odd_size[0], odd_size[1])
        shape_kwargs = _polygon_kwargs(odd_shape, length, odd_size, odd_thick[0])
        draw_fun, shape_params = polygon_params(odd_shape, **shape_kwargs)
        shape_draw = [draw_fun, shape_params]
        odd_img = _make_img_shape(
            img_in, fg_type, odd_size, contrasts[0], odd_thick[0], odd_colour, shape_draw
        )

        # creating a bigger/smaller size for the common images
        magnitude = (np.random.choice([-1, 1]), np.random.uniform(0.5, 0.75))
        com_size = (
            int(_change_scale(odd_size[0], magnitude)),
            int(_change_scale(odd_size[1], magnitude))
        )

        com_imgs = []
        for i in range(self.num_imgs - 1):
            colour = odd_colour if i == 0 else _rnd_colour()
            thick = odd_thick[0] if i == 1 else odd_thick[1]
            contrast = contrasts[0] if i == 2 else contrasts[1]

            com_shape_params = _enlarge_polygon(magnitude, shape_params, odd_shape, com_size)
            shape_draw = [draw_fun, com_shape_params]
            ci_img = _make_img_shape(img_in, fg_type, com_size, contrast, thick, colour, shape_draw)
            com_imgs.append(ci_img)

        return [odd_img, *com_imgs]

    def colour_feature(self, img_in):
        polygons = [*CV2_BASIC_SHAPES, *CV2_CUSTOM_SHAPES]
        length = _rnd_scale(img_in.shape[0], self.fg_scale)

        # this two colours can be identical, the probability is very slim, considered as DB noise
        odd_colour = _rnd_colour()
        com_colour = _rnd_colour()

        odd_shape = np.random.choice(polygons)
        # removing the unique shape from list and choosing the odd shape
        polygons.remove(odd_shape)
        odd_thick = _rnd_thickness()
        contrasts = _rnd_contrast()

        odd_img = _make_img(img_in, contrasts[0], odd_shape, length, odd_thick[0], odd_colour)

        com_imgs = []
        for i in range(self.num_imgs - 1):
            com_shape = odd_shape if i == 0 else np.random.choice(polygons)
            thick = odd_thick[0] if i == 1 else odd_thick[1]
            contrast = contrasts[0] if i == 2 else contrasts[1]

            ci_img = _make_img(img_in, contrast, com_shape, length, thick, com_colour)
            com_imgs.append(ci_img)

        return [odd_img, *com_imgs]

    def shape_feature(self, img_in):
        polygons = [*CV2_BASIC_SHAPES, *CV2_CUSTOM_SHAPES]
        length = _rnd_scale(img_in.shape[0], self.fg_scale)

        com_shape = np.random.choice(polygons)
        # removing the unique shape from list and choosing the odd shape
        polygons.remove(com_shape)
        odd_shape = np.random.choice(polygons)

        contrasts = _rnd_contrast()
        odd_thick = _rnd_thickness()
        odd_colour = _rnd_colour()
        odd_img = _make_img(img_in, contrasts[0], odd_shape, length, odd_thick[0], odd_colour)

        com_imgs = []
        for i in range(self.num_imgs - 1):
            colour = odd_colour if i == 0 else _rnd_colour()
            thick = odd_thick[0] if i == 1 else odd_thick[1]
            contrast = contrasts[0] if i == 2 else contrasts[1]

            ci_img = _make_img(img_in, contrast, com_shape, length, thick, colour)
            com_imgs.append(ci_img)

        return [odd_img, *com_imgs]


def oddx_bg_folder(root, num_imgs, target_size, preprocess, **kwargs):
    scale = (0.08, 1.0)
    bg_transform = torch_transforms.Compose(dataset_utils.pre_transform_train(target_size, scale))
    transform = torch_transforms.Compose(dataset_utils.post_transform(*preprocess))
    bg_db = dataset_utils.NoTargetFolder(root, loader=dataset_utils.cv2_loader)
    return OddOneOutTrain(bg_db, num_imgs, transform, bg_transform, **kwargs)
