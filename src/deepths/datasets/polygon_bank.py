"""
Generating different polygons and geometrical shapes.
"""

import numpy as np
import sys

import cv2

from . import dataset_utils

CV2_OVAL_SHAPES = ['circle', 'ellipse']
CV2_POLYGON_SHAPES = ['square', 'rectangle', 'triangle']

SHAPES = [*CV2_OVAL_SHAPES, *CV2_POLYGON_SHAPES]


def draw(img, shape_params, **kwargs):
    pts = shape_params['pts'] if 'pts' in shape_params else [cv2.ellipse2Poly(**shape_params)]
    return cv2_filled_polygons(img, pts, **kwargs)


def polygon_params(polygon, **kwargs):
    if polygon in CV2_OVAL_SHAPES:
        return cv2_shapes(polygon, **kwargs)
    elif polygon in CV2_POLYGON_SHAPES:
        return cv2_polygons(**kwargs)
    else:
        sys.exit('Unsupported polygon to draw: %s' % polygon)


def cv2_polygons(pts, rotation=0):
    old_pts = pts[0]
    if rotation != 0:
        pts = [rotate2d(old_pts, np.mean(old_pts, axis=0), angle=rotation).astype(int)]
    else:
        pts = [old_pts.astype(int)]
    return {'pts': pts}


def cv2_filled_polygons(img, pts, color, thickness):
    img = cv2.polylines(img, pts=pts, color=color, thickness=abs(thickness), isClosed=True)
    if thickness < 0:
        img = cv2.fillPoly(img, pts=pts, color=color)
    return img


def cv2_shapes(polygon, length, ref_pt, rotation=0, arc_start=0, arc_end=360):
    if polygon in CV2_OVAL_SHAPES:
        if polygon == 'circle':
            length = (length, length)
        return {
            'center': ref_pt, 'axes': length, 'angle': int(np.rad2deg(rotation)),
            'arcStart': int(arc_start), 'arcEnd': int(arc_end), 'delta': 5
        }
    else:
        sys.exit('Unsupported polygon to draw: %s' % polygon)


def rotate2d(pts, centre, angle):
    return np.dot(
        pts - centre,
        np.array([
            [np.cos(angle), np.sin(angle)],
            [-np.sin(angle), np.cos(angle)]
        ])
    ) + centre


def handle_symmetry(symmetry, org_kwargs, stimuli):
    if symmetry == "n/a":
        return org_kwargs
    shape_kwargs = dict()
    for key, val in org_kwargs.items():
        shape_kwargs[key] = val
    polygon = stimuli.shape['name']
    if polygon in CV2_OVAL_SHAPES:
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
            length = np.minimum(stimuli.canvas[0], stimuli.canvas[1]) / 2
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
