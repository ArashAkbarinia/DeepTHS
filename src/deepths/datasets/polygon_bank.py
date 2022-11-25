"""
Generating different polygons and geometrical shapes.
"""

import numpy as np
import random

import cv2

from . import dataset_utils, imutils

SHAPES_OVAL = ['circle', 'ellipse', 'half_circle', 'half_ellipse']
SHAPES_TRIANGLE = ['scalene', 'equilateral', 'isosceles']
SHAPES_QUADRILATERAL = ['square', 'rectangle', 'parallelogram', 'rhombus', 'kite', 'quadri']
SHAPES_HEXAGON = ['regular6']
SHAPES_DODECAGON = ['cross']
SHAPES = [
    SHAPES_OVAL, SHAPES_TRIANGLE, SHAPES_QUADRILATERAL,
    [*SHAPES_HEXAGON, *SHAPES_DODECAGON]
]
SHAPES_ORIENTATION = [SHAPES_OVAL[1:], *SHAPES[1:]]

SHAPES_SYMMETRY_BOTH = {
    'oval': ['circle', 'ellipse'],
    'quadrilateral': ['square', 'rectangle'],
    'hexagon': ['regular6'],
    'dodecagon': ['cross']
}
SHAPES_SYMMETRY_ONE = {
    'oval': ['half_circle', 'half_ellipse'],
    'triangle': ['equilateral', 'isosceles'],
    'quadrilateral': ['kite'],
    'dodecagon': ['cross']
}
SHAPES_SYMMETRY_NONE = {
    'oval': ['half_circle', 'half_ellipse'],
    'triangle': ['scalene', 'equilateral', 'isosceles'],
    'quadrilateral': ['quadri', 'rhombus', 'parallelogram'],
    'hexagon': ['regular6'],
}
SHAPES_SYMMETRY = {
    'both': SHAPES_SYMMETRY_BOTH,
    'h': SHAPES_SYMMETRY_ONE,
    'v': SHAPES_SYMMETRY_ONE,
    'none': SHAPES_SYMMETRY_NONE
}


def draw(img, shape_params, **kwargs):
    pts = shape_params['pts'] if 'pts' in shape_params else cv2.ellipse2Poly(**shape_params)
    return cv2_filled_polygons(img, [pts], **kwargs)


def polygon_params(polygon, **kwargs):
    return ovals(**kwargs) if polygon in SHAPES_OVAL else polygons(**kwargs)


def generate_quadrilaterals(polygon, canvas, symmetry):
    if polygon == 'quadri':
        return [
            (canvas[1] * random.uniform(0.1, 0.3), canvas[0] * random.uniform(0.1, 0.3)),
            (canvas[1] * random.uniform(0.1, 0.3), canvas[0] * random.uniform(0.7, 0.9)),
            (canvas[1] * random.uniform(0.7, 0.9), canvas[0] * random.uniform(0.7, 0.9)),
            (canvas[1] * random.uniform(0.7, 0.9), canvas[0] * random.uniform(0.1, 0.3))
        ]
    if polygon == 'kite':
        rx, ry = canvas[1] * random.uniform(0.1, 0.3), canvas[0] * random.uniform(0.4, 0.6)
        sx, sy = canvas[1] - rx - 2, (min(canvas[0] - ry, ry) - 2) * 2
        alpha = random.uniform(0.6, 0.8)
        alphas = dataset_utils.shuffle([1 - alpha, alpha])
        if symmetry == 'v':
            sy = sy * random.uniform(0.8, 0.9)
            sx = sy * random.uniform(0.3, 0.6)
            pt2, pt4 = (0.5, alphas[0]), (0.5, alphas[1])
        else:
            sx = sx * random.uniform(0.8, 0.9)
            sy = sx * random.uniform(0.3, 0.6)
            pt2, pt4 = (alphas[0], 0.5), (alphas[0], 0.5)
        return [
            (rx, ry), (rx + sx * pt2[0], min(ry + sy * pt2[1], canvas[0])),
            (rx + sx, ry), (rx + sx * pt4[0], max(ry - sy * pt4[1], 0))
        ]

    tilt = 0
    if polygon in ['parallelogram', 'rhombus']:
        tilt = random.uniform(0.05, 0.15) * min(canvas[0], canvas[1])

    if polygon in ['square', 'rhombus']:
        sx = (min(canvas[0], canvas[1]) - tilt) * random.uniform(0.8, 0.9)
        sy = sx
    else:  # rectangle and parallelogram
        sy, sx = [(c - tilt) * random.uniform(0.7, 0.9) for c in canvas]
        if sx == sy:  # in a very unlikely scenario that they are equal
            sx = sy * 0.7
    dx, dy = canvas[1] - sx - 2 - tilt, canvas[0] - sy - 2 - tilt
    if polygon in ['parallelogram', 'rhombus']:
        tilt_dir = np.random.choice([-1, 1])
        if random.random() >= 1.0:  # which side is parallel
            sint, eint = (0, dx) if tilt_dir == 1 else (tilt, tilt + dx)
            rx, ry = dataset_utils.randint(sint, eint), dataset_utils.randint(0, dy)
            t0, t1, t2 = 0, tilt_dir * tilt, 0
        else:
            sint, eint = (0, dy) if tilt_dir == 1 else (tilt, tilt + dy)
            rx, ry = dataset_utils.randint(0, dx), dataset_utils.randint(sint, eint)
            t0, t1 = tilt_dir * tilt, 0
            t2 = -t0
    else:
        rx, ry = dataset_utils.randint(0, dx), dataset_utils.randint(0, dy)
        t0, t1, t2 = 0, 0, 0
    lengths = [(sx, t0), (t1, sy), (-sx, t2)]
    pts = [(rx, ry)]
    for i in range(3):
        pts.append((pts[i][0] + lengths[i][0], pts[i][1] + lengths[i][1]))
    return pts


def generate_triangles(polygon, canvas, symmetry):
    if polygon == 'scalene':
        # TODO: it might become other types of triangle (although very unlikeley)
        probs = dataset_utils.shuffle([[0.1, 0.3], [0.4, 0.6], [0.7, 0.9]])
        return [
            (canvas[1] * random.uniform(*probs[0]), canvas[0] * random.uniform(*probs[1])),
            (canvas[1] * random.uniform(*probs[1]), canvas[0] * random.uniform(*probs[2])),
            (canvas[1] * random.uniform(*probs[2]), canvas[0] * random.uniform(*probs[0])),
        ]
    else:  # 'equilateral' or 'isosceles'
        circum = 'circle' if polygon == 'equilateral' else 'ellipse'
        length = min(canvas[0], canvas[1])
        centre = ref_point(length, canvas)
        radius = length / 2
        if circum == 'ellipse':
            scale, min_length = (0.3, 0.7), 5
            radius = (max(radius * random.uniform(*scale), min_length), radius)
        else:
            radius = (radius, radius)
        half_radius = [r / 2 for r in radius]
        signs = dataset_utils.shuffle([-1, 1])
        # vertically symmetrical
        pt1 = (centre[0], centre[1] + signs[0] * radius[1])
        pt2 = (centre[0] - np.sqrt(3) * half_radius[0], centre[1] + signs[1] * half_radius[1])
        pt3 = (centre[0] + np.sqrt(3) * half_radius[0], centre[1] + signs[1] * half_radius[1])
        pts = np.array([pt1, pt2, pt3])
        if symmetry == 'h':
            pts = rotate2d(pts, centre, np.random.choice([np.pi / 2, -np.pi / 2]))
        elif symmetry == 'none':
            pts = rotate2d(pts, centre, np.deg2rad(np.random.randint(15, 76)))
        return list(pts)


def generate_hexagons(polygon, canvas, symmetry):
    # TODO: only regular6 is supported
    length = min(canvas[0], canvas[1])
    centre = ref_point(length, canvas)
    radius = length / 2
    radius = (radius, radius)
    half_radius = [r / 2 for r in radius]
    signs = dataset_utils.shuffle([-1, 1])
    # both symmetrical
    pt1 = (centre[0], centre[1] + signs[0] * radius[1])
    pt2 = (centre[0] - np.sqrt(3) * half_radius[0], centre[1] - signs[1] * half_radius[1])
    pt3 = (centre[0] - np.sqrt(3) * half_radius[0], centre[1] + signs[1] * half_radius[1])
    pt4 = (centre[0], centre[1] - signs[0] * radius[1])
    pt5 = (centre[0] + np.sqrt(3) * half_radius[0], centre[1] + signs[1] * half_radius[1])
    pt6 = (centre[0] + np.sqrt(3) * half_radius[0], centre[1] - signs[1] * half_radius[1])
    pts = np.array([pt1, pt2, pt3, pt4, pt5, pt6])
    if symmetry == 'none':
        pts = rotate2d(pts, centre, np.deg2rad(np.random.randint(15, 46)))
    return list(pts)


def generate_dodecagons(polygon, canvas, symmetry):
    # TODO: only cross is supported
    if symmetry == 'h':
        sx = canvas[1] * random.uniform(0.8, 0.9)
        sy = min(sx * random.uniform(0.5, 0.6), canvas[0])
        mx = dataset_utils.shuffle([0.2, 0.3, 0.5])
        my = [1 / 3] * 3
    elif symmetry == 'v':
        sy = canvas[0] * random.uniform(0.8, 0.9)
        sx = min(sy * random.uniform(0.5, 0.6), canvas[1])
        mx = [1 / 3] * 3
        my = dataset_utils.shuffle([0.2, 0.3, 0.5])
    else:  # both, seperate none from it
        sx = min(canvas[0], canvas[1]) * random.uniform(0.8, 0.9)
        sy = sx
        mx = [1 / 3] * 3
        my = mx
    dx, dy = canvas[1] - sx - 2, canvas[0] - sy - 2
    rx, ry = dataset_utils.randint(0, dx), dataset_utils.randint(0, dy)
    pts = [(rx + mx[0] * sx, ry)]
    lengths = [
        (mx[1], 0), (0, my[0]), (mx[2], 0), (0, my[1]), (-mx[2], 0), (0, my[2]),
        (-mx[1], 0), (0, -my[2]), (-mx[0], 0), (0, -my[1]), (mx[0], 0)
    ]
    for i in range(11):
        pts.append((pts[i][0] + lengths[i][0] * sx, pts[i][1] + lengths[i][1] * sy))
    return pts


def generate_polygons(polygon, canvas, symmetry):
    half_canvas = imutils.centre_pixel(canvas)
    if polygon in SHAPES_OVAL:
        return generate_ovals(polygon, half_canvas, symmetry)
    elif polygon in SHAPES_TRIANGLE:
        pts = generate_triangles(polygon, half_canvas, symmetry)
    elif polygon in SHAPES_QUADRILATERAL:
        pts = generate_quadrilaterals(polygon, half_canvas, symmetry)
    elif polygon in SHAPES_HEXAGON:
        pts = generate_hexagons(polygon, half_canvas, symmetry)
    else:
        pts = generate_dodecagons(polygon, half_canvas, symmetry)
    return {'pts': np.array(pts)}


def polygons(pts, rotation=0):
    pts = pts if rotation == 0 else rotate2d(pts, np.mean(pts, axis=0), angle=rotation)
    return {'pts': pts.astype(int)}


def cv2_filled_polygons(img, pts, color, thickness):
    img = cv2.polylines(img, pts=pts, color=color, thickness=abs(thickness), isClosed=True)
    if thickness < 0:
        img = cv2.fillPoly(img, pts=pts, color=color)
    return img


def ovals(length, ref_pt, rotation=0, arc_start=0, arc_end=360):
    return {
        'center': ref_pt, 'axes': tuple([int(le) for le in length]), 'delta': 5,
        'angle': int(np.rad2deg(rotation)), 'arcStart': int(arc_start), 'arcEnd': int(arc_end)
    }


def generate_ovals(polygon, canvas, symmetry):
    kwargs = dict()
    length = min(canvas[0], canvas[1])
    kwargs['ref_pt'] = ref_point(length, canvas)
    length = length // 2  # this is the radius
    if 'ellipse' in polygon:
        scale, min_length = (0.2, 0.8), 5
        length = (length, max(int(length * random.uniform(*scale)), min_length))
    else:
        length = (length, length)
    kwargs['length'] = length
    # these are half-circles
    if 'half_' in polygon:
        if symmetry == 'v':
            kwargs[np.random.choice(['arc_start', 'arc_end'])] = 180
        elif symmetry == 'h':
            kwargs['arc_start'], kwargs['arc_end'] = dataset_utils.shuffle([90, 270])
        elif symmetry == 'none':
            kwargs['arc_start'] = np.random.choice([11, 22, 34, 45, 101, 112, 124, 135])
            kwargs['arc_end'] = kwargs['arc_start'] + 180
        else:
            kwargs['arc_start'] = dataset_utils.randint(0, 360)
            kwargs['arc_end'] = kwargs['arc_start'] + 180
    return kwargs


def rotate2d(pts, centre, angle):
    return np.dot(
        pts - centre,
        np.array([
            [np.cos(angle), np.sin(angle)],
            [-np.sin(angle), np.cos(angle)]
        ])
    ) + centre


def _enlarge(value, magnitude, ref=None):
    int_in = False
    if not hasattr(value, '__len__'):
        value, int_in = (value,), True
    if ref is None:
        ref = (0,) * len(value)
    value_centered = np.array([v - r for v, r in zip(value, ref)])
    value_magnified = (value_centered + value_centered * magnitude).astype(int)
    return value_magnified[0] if int_in else tuple(value_magnified)


def enlarge_polygon(magnitude, shape_params, shape, canvas):
    if magnitude == 0:
        return shape_params
    shape_params = shape_params.copy()
    if shape in SHAPES_OVAL:
        length = min(canvas[0], canvas[1]) / 2
        shape_params['center'] = ref_point(_enlarge(length, magnitude), canvas)
        shape_params['axes'] = _enlarge(shape_params['axes'], magnitude)
    else:
        old_pts = shape_params['pts'].copy()
        new_pts = np.array([_enlarge(pt, magnitude, old_pts[0]) for pt in old_pts])
        (min0, min1), (max0, max1) = new_pts.min(axis=0), new_pts.max(axis=0)
        new_pts = [(pt[0] - min0, pt[1] - min1) for pt in new_pts]
        lx, ly = max0 - min0 + 2, max1 - min1 + 2
        dx, dy = dataset_utils.randint(0, canvas[1] - lx), dataset_utils.randint(0, canvas[0] - ly)
        shape_params['pts'] = np.array([(pt[0] + dx, pt[1] + dy) for pt in new_pts])
    return shape_params


def ref_point(length, img_size):
    diff = min(img_size[0], img_size[1]) - length - 2
    cy, cx = imutils.centre_pixel(img_size)
    if diff <= 0:
        ref_pt = (cx, cy)
    else:
        diff = diff // 2
        ref_pt = (
            dataset_utils.randint(cx - diff, cx + diff),
            dataset_utils.randint(cy - diff, cy + diff)
        )
    return ref_pt


def handle_shape(stimuli):
    shape = {'name': stimuli.shape['name'], 'kwargs': stimuli.shape['kwargs'].copy()}
    posy, posx = imutils.centre_pixel(stimuli.canvas)
    posx, posy = posx * stimuli.position[0], posy * stimuli.position[1]
    if 'pts' in shape['kwargs']:
        pts = shape['kwargs']['pts']
        shape['kwargs']['pts'] = np.array([(pt[0] + posx, pt[1] + posy) for pt in pts])
    else:
        ref_pt = shape['kwargs']['ref_pt']
        shape['kwargs']['ref_pt'] = int(ref_pt[0] + posx), int(ref_pt[1] + posy)
    shape['kwargs']['rotation'] = stimuli.rotation
    shape_params = polygon_params(shape['name'], **shape['kwargs'])
    return enlarge_polygon(stimuli.size, shape_params, shape['name'], stimuli.canvas)
