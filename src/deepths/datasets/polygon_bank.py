"""
Generating different polygons and geometrical shapes.
"""

import numpy as np
import sys

import cv2

from . import dataset_utils, imutils

SHAPES_OVAL = ['circle', 'ellipse']
SHAPES_TRIANGLE = ['triangle', ]  # 'scalene', 'equilateral', 'isosceles']
SHAPES_QUADRILATERAL = ['square', 'rectangle', ]  # 'parallelogram', 'rhombus', 'kite', 'quadri']

SHAPES = [*SHAPES_OVAL, *SHAPES_TRIANGLE, *SHAPES_QUADRILATERAL]


def draw(img, shape_params, **kwargs):
    pts = shape_params['pts'] if 'pts' in shape_params else [cv2.ellipse2Poly(**shape_params)]
    return cv2_filled_polygons(img, pts, **kwargs)


def polygon_params(polygon, **kwargs):
    return ovals(polygon, **kwargs) if polygon in SHAPES_OVAL else polygons(polygon, **kwargs)


def quadrilaterals(polygon, length, ref_pt, angle=0):
    angle_rad = np.deg2rad(abs(angle))
    if polygon in ['square', 'rhombus']:
        length = (length, length)
    if polygon in ['square', 'rectangle']:
        length = ((0, length[0]), (length[1], 0), (0, -length[0]))
    elif polygon in ['parallelogram', 'rhombus', 'kite']:
        angle_pt = (length[1] * np.cos(angle_rad), length[1] * np.sin(angle_rad))
        if polygon in ['parallelogram', 'rhombus']:
            length = ((0, length[0]), angle_pt, (0, -length[0]))
        else:
            length = (angle_pt, (-angle_pt[0], length[0]), (-angle_pt[0], -length[0]))
        if angle < 0:
            length = tuple([(tmp_l[1], tmp_l[0]) for tmp_l in length])
    pts = [ref_pt]
    for i in range(3):
        pts.append((pts[i][0] + length[i][0], pts[i][1] + length[i][1]))
    return np.array(pts), {'length': length, 'angle': angle}


def generate_quadrilaterals(polygon, canvas):
    length = np.minimum(canvas[0], canvas[1]) / 2
    kwargs = dict()
    if polygon in ['parallelogram', 'rhombus', 'kite']:
        kwargs['angle'] = np.random.choice([1, -1]) * np.random.randint(15, 60)
    kwargs['ref_pt'] = ref_point(length, polygon, canvas)
    scale = (0.2, 0.8)
    min_length = 5
    if polygon in ['rectangle', 'parallelogram', 'kite']:
        length = (length, max(int(length * np.random.uniform(*scale)), min_length))
    elif polygon == 'quadri':
        length = [(
            signs[0] * max(int(length * np.random.uniform(*scale)), min_length),
            signs[1] * max(int(length * np.random.uniform(*scale)), min_length)
        ) for signs in [(1, -1), (-1, -1), (1, 1)]]
    kwargs['length'] = length
    return kwargs


def triangles(polygon, pts):
    return pts, {}


def polygons(polygon, rotation=0, **kwargs):
    if polygon in SHAPES_QUADRILATERAL:
        pts, org_def = quadrilaterals(polygon, **kwargs)
    else:
        pts, org_def = triangles(polygon, **kwargs)

    pts = rotated_polygons(pts, rotation)
    return {'pts': [pts.astype(int)], 'def': org_def, 'name': polygon}


def rotated_polygons(pts, rotation=0):
    return pts if rotation == 0 else rotate2d(pts, np.mean(pts, axis=0), angle=rotation)


def cv2_filled_polygons(img, pts, color, thickness):
    img = cv2.polylines(img, pts=pts, color=color, thickness=abs(thickness), isClosed=True)
    if thickness < 0:
        img = cv2.fillPoly(img, pts=pts, color=color)
    return img


def ovals(polygon, length, ref_pt, rotation=0, arc_start=0, arc_end=360):
    if polygon == 'circle':
        length = (length, length)
    return {
        'center': ref_pt, 'axes': length, 'angle': int(np.rad2deg(rotation)),
        'arcStart': int(arc_start), 'arcEnd': int(arc_end), 'delta': 5
    }


def rotate2d(pts, centre, angle):
    return np.dot(
        pts - centre,
        np.array([
            [np.cos(angle), np.sin(angle)],
            [-np.sin(angle), np.cos(angle)]
        ])
    ) + centre


def handle_symmetry(symmetry, org_kwargs, polygon, canvas):
    if symmetry == "n/a":
        return org_kwargs
    shape_kwargs = dict()
    for key, val in org_kwargs.items():
        shape_kwargs[key] = val
    if polygon in SHAPES_OVAL:
        if symmetry == 'v':
            shape_kwargs[np.random.choice(['arc_start', 'arc_end'])] = 180
        elif symmetry == 'h':
            shape_kwargs['arc_start'], shape_kwargs['arc_end'] = dataset_utils.shuffle([90, 270])
        elif symmetry == 'none':
            shape_kwargs['arc_start'] = np.random.choice([11, 22, 34, 45, 101, 112, 124, 135])
            shape_kwargs['arc_end'] = shape_kwargs['arc_start'] + 180
    elif polygon in ['square', 'rectangle']:
        if symmetry != 'both':
            pts = shape_kwargs['pts'][0].copy()
            length = np.minimum(canvas[0], canvas[1]) / 2
            val = dataset_utils.randint(2, int(length * 0.25))
            if symmetry == 'v':
                v0, v1 = np.random.choice([{0, 3}, {1, 2}])
                pts[v0] = (pts[v0][0] + val, pts[v0][1])
                pts[v1] = (pts[v1][0] - val, pts[v1][1])
            elif symmetry == 'h':
                v0, v1 = np.random.choice([{0, 1}, {2, 3}])
                pts[v0] = (pts[v0][0], pts[v0][1] + val)
                pts[v1] = (pts[v1][0], pts[v1][1] - val)
            elif symmetry == 'none':
                v0 = np.random.choice([0, 1, 2, 3])
                val = val if v0 in [0, 1] else -val
                pts[v0] = (pts[v0][0] + val, pts[v0][1])
            shape_kwargs['pts'] = [pts]
    elif polygon == 'triangle':
        if symmetry != ' none':
            pts = shape_kwargs['pts'][0].copy()
            v0, v1, v2 = dataset_utils.shuffle([0, 1, 2])
            if symmetry in ['h', 'both']:
                xind = np.argmax([abs(pts[v2][0] - pts[v0][0]), abs(pts[v2][0] - pts[v1][0])])
                xval = pts[v0][0] if xind == 0 else pts[v1][0]
                pts[v0] = (xval, pts[v0][1])
                pts[v1] = (xval, pts[v1][1])
                pts[v2] = (pts[v2][0], (pts[v0][1] + pts[v1][1]) / 2)
                if symmetry == 'both':
                    pts = np.array([
                        pts[v0], pts[v1], pts[v2], pts[v0],
                        (xval, (pts[v0][1] + pts[v1][1]) / 2),
                        (pts[v2][0], pts[v0][1]),
                        (pts[v2][0], pts[v1][1]),
                        (xval, (pts[v0][1] + pts[v1][1]) / 2)
                    ])
            elif symmetry == 'v':
                yind = np.argmax([abs(pts[v2][1] - pts[v0][1]), abs(pts[v2][1] - pts[v1][1])])
                yval = pts[v0][1] if yind == 0 else pts[v1][1]
                pts[v0] = (pts[v0][1], yval)
                pts[v1] = (pts[v1][1], yval)
                pts[v2] = ((pts[v0][0] + pts[v1][0]) / 2, pts[v2][1])
            shape_kwargs['pts'] = [pts]
    return shape_kwargs


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
    elif polygon in ['triangle']:
        ymax, xmax = img_size[:2]
        ref_pt = (dataset_utils.randint(0, xmax - length), dataset_utils.randint(0, ymax - length))
    else:
        sys.exit('Unsupported polygon to draw: %s' % polygon)
    return ref_pt


def handle_shape(stimuli):
    shape = stimuli.shape
    shape['kwargs']['rotation'] = stimuli.rotation
    shape_kwargs = shape['kwargs'] if stimuli.unique_feature == 'shape' else handle_symmetry(
        stimuli.symmetry, shape['kwargs'], shape['name'], stimuli.canvas)
    shape_params = polygon_params(shape['name'], **shape_kwargs)
    return enlarge_polygon(stimuli.size, shape_params, stimuli)
