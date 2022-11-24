"""
Generating different polygons and geometrical shapes.
"""

import numpy as np
import random
import sys

import cv2

from . import dataset_utils, imutils

SHAPES_OVAL = ['circle', 'ellipse', 'half_circle', 'half_ellipse']
SHAPES_TRIANGLE = ['scalene', 'equilateral', 'isosceles']
SHAPES_QUADRILATERAL = ['square', 'rectangle', 'parallelogram', 'rhombus', 'kite', 'quadri']
SHAPES = [SHAPES_OVAL, SHAPES_TRIANGLE, SHAPES_QUADRILATERAL]

SHAPES_SYMMETRY_BOTH = ['circle', 'ellipse', 'square', 'rectangle']
SHAPES_SYMMETRY_ONE = ['half_circle', 'half_ellipse', 'equilateral', 'isosceles', 'rhombus', 'kite']
SHAPES_SYMMETRY_NONE = [
    'half_circle', 'half_ellipse', 'scalene', 'equilateral', 'isosceles', 'quadri'
]
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
            (canvas[0] * random.uniform(0.1, 0.3), canvas[1] * random.uniform(0.1, 0.3)),
            (canvas[0] * random.uniform(0.1, 0.3), canvas[1] * random.uniform(0.7, 0.9)),
            (canvas[0] * random.uniform(0.7, 0.9), canvas[1] * random.uniform(0.7, 0.9)),
            (canvas[0] * random.uniform(0.7, 0.9), canvas[1] * random.uniform(0.1, 0.3))
        ]
    if polygon == 'kite':
        r0, r1 = canvas[0] * random.uniform(0.4, 0.6), canvas[1] * random.uniform(0.1, 0.3)
        s0, s1 = min(canvas[0] - r0, r0) * 2, canvas[1] - r1
        alpha = random.uniform(0.6, 0.8)
        alphas = dataset_utils.shuffle([1 - alpha, alpha])
        if symmetry == 'v':
            s1 = s0 * random.uniform(0.3, 0.6)
            pt2, pt4 = (alphas[0], 0.5), (alphas[1], 0.5)
        else:
            s0 = s1 * random.uniform(0.3, 0.6)
            pt2, pt4 = (0.5, alphas[0]), (0.5, alphas[0])
        return [
            (r0, r1), (r0 + s0 * pt2[0], r1 + s1 * pt2[1]),
            (r0, r1 + s1), (r0 - s0 * pt4[0], r1 + s1 * pt4[1])
        ]

    if polygon in ['square', 'rhombus']:
        s0 = int(min(canvas[0], canvas[1]) * random.uniform(0.8, 0.9))
        s1 = s0
    else:  # triangle and parallelogram
        s0, s1 = [int(c * random.uniform(0.7, 0.9)) for c in canvas]
        if s0 == s1:  # in a very unlikely scenario that they are equal
            s0 = s1 * 0.7
    d0, d1 = canvas[0] - s0, canvas[1] - s1
    if polygon in ['parallelogram', 'rhombus']:
        tilts = dataset_utils.shuffle([[0.05, 0.15], [0.85, 0.95]])
        if random.random() >= 0.5:  # which side is parallel
            r0, r1 = dataset_utils.randint(0, d0), d1 * random.uniform(*tilts[0])
            t0, t1, t2 = 0, r1 - d1 * random.uniform(*tilts[1]), 0
        else:
            r0, r1 = d0 * random.uniform(*tilts[0]), dataset_utils.randint(0, d1)
            t0, t1 = r0 - d0 * random.uniform(*tilts[1]), 0
            t2 = -t0
    else:
        r0, r1 = dataset_utils.randint(0, d0), dataset_utils.randint(0, d1)
        t0, t1, t2 = 0, 0, 0
    lengths = [(t0, s0), (s1, 0), (t2, -s0)]
    pts = [(r0, r1)]
    for i in range(3):
        pts.append((pts[i][0] + lengths[i][0], pts[i][1] + lengths[i][1]))
    return pts


def generate_triangles(polygon, canvas, symmetry):
    if polygon == 'scalene':
        # TODO: it might become other types of triangle (although very unlikeley)
        probs = dataset_utils.shuffle([[0.1, 0.3], [0.4, 0.6], [0.7, 0.9]])
        return [
            (canvas[0] * random.uniform(*probs[0]), canvas[1] * random.uniform(*probs[1])),
            (canvas[0] * random.uniform(*probs[1]), canvas[1] * random.uniform(*probs[2])),
            (canvas[0] * random.uniform(*probs[2]), canvas[1] * random.uniform(*probs[0])),
        ]
    else:  # 'equilateral' or 'isosceles'
        circum = 'circle' if polygon == 'equilateral' else 'ellipse'
        length = min(canvas[0], canvas[1])
        centre = ref_point(length, circum, canvas)
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


def generate_polygons(polygon, canvas, symmetry):
    if polygon in SHAPES_QUADRILATERAL:
        pts = generate_quadrilaterals(polygon, canvas, symmetry)
    else:
        pts = generate_triangles(polygon, canvas, symmetry)
    return pts


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
        'center': ref_pt, 'axes': [int(le) for le in length], 'angle': int(np.rad2deg(rotation)),
        'arcStart': int(arc_start), 'arcEnd': int(arc_end), 'delta': 5
    }


def generate_ovals(polygon, canvas, symmetry):
    kwargs = dict()
    length = min(canvas[0], canvas[1])
    kwargs['ref_pt'] = ref_point(length, polygon, canvas)
    length = length // 2  # this is the radius
    if 'ellipse' in polygon:
        scale, min_length = (0.2, 0.8), 5
        length = (length, max(int(length * random.uniform(*scale)), min_length))
    else:
        length = (length, length)
    kwargs['length'] = length
    # these are half-circles
    if symmetry == 'v':
        kwargs[np.random.choice(['arc_start', 'arc_end'])] = 180
    elif symmetry == 'h':
        kwargs['arc_start'], kwargs['arc_end'] = dataset_utils.shuffle([90, 270])
    elif symmetry == 'none':
        kwargs['arc_start'] = np.random.choice([11, 22, 34, 45, 101, 112, 124, 135])
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


def enlarge_polygon(magnitude, shape_params, stimuli):
    if magnitude == 0:
        return shape_params
    shape, out_size = stimuli.shape['name'], stimuli.canvas
    length = np.minimum(stimuli.canvas[0], stimuli.canvas[1]) / 2
    ref_pt = ref_point(_enlarge(length, magnitude), shape, out_size)
    shape_params = shape_params.copy()
    if shape in SHAPES_OVAL:
        shape_params['center'] = ref_pt
        shape_params['axes'] = _enlarge(shape_params['axes'], magnitude)
    else:
        old_pts = shape_params['pts'][0].copy()
        pt1 = ref_pt
        other_pts = [_enlarge(pt, magnitude, old_pts[0]) for pt in old_pts[1:]]
        other_pts = [(pt[0] + pt1[0], pt[1] + pt1[1]) for pt in other_pts]
        shape_params['pts'] = [np.array([pt1, *other_pts])]
    return shape_params


def ref_point(length, polygon, img_size):
    diff = min(img_size[0], img_size[1]) - length - 2
    if polygon in SHAPES_OVAL:
        cy, cx = imutils.centre_pixel(img_size)
        if diff <= 0:
            ref_pt = (cx, cy)
        else:
            diff = diff // 2
            ref_pt = (
                dataset_utils.randint(cx - diff, cx + diff),
                dataset_utils.randint(cy - diff, cy + diff)
            )
    elif polygon in SHAPES_QUADRILATERAL:  # FIXME
        ref_pt = (dataset_utils.randint(0, diff), dataset_utils.randint(0, diff))
    elif polygon in SHAPES_TRIANGLE:
        ymax, xmax = img_size[:2]
        ref_pt = (dataset_utils.randint(0, xmax - length), dataset_utils.randint(0, ymax - length))
    else:
        sys.exit('Unsupported polygon to draw: %s' % polygon)
    return ref_pt


def handle_shape(stimuli):
    shape = stimuli.shape
    shape['kwargs']['rotation'] = stimuli.rotation
    shape_params = polygon_params(shape['name'], **shape['kwargs'])
    return enlarge_polygon(stimuli.size, shape_params, stimuli)
