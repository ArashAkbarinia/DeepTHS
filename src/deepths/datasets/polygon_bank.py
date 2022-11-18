"""
Generating different polygons and geometrical shapes.
"""

import numpy as np
import sys

import cv2

CV2_OVAL_SHAPES = ['circle', 'ellipse']
CV2_POLYGON_SHAPES = ['square', 'rectangle', 'triangle']

SHAPES = [*CV2_OVAL_SHAPES, *CV2_POLYGON_SHAPES]


def polygon_params(polygon, **kwargs):
    if polygon in CV2_OVAL_SHAPES:
        draw_fun, params = cv2_shapes(polygon, **kwargs)
    elif polygon in CV2_POLYGON_SHAPES:
        draw_fun, params = cv2_polygons(**kwargs)
    else:
        sys.exit('Unsupported polygon to draw: %s' % polygon)
    return draw_fun, params


def cv2_polygons(pts, rotation=0):
    old_pts = pts[0]
    if rotation != 0:
        pts = [rotate2d(old_pts, np.mean(old_pts, axis=0), angle=rotation).astype(int)]
    else:
        pts = [old_pts.astype(int)]
    return cv2_filled_polygons, {'pts': pts}


def cv2_filled_polygons(img, pts, color, thickness):
    img = cv2.polylines(img, pts=pts, color=color, thickness=abs(thickness), isClosed=True)
    if thickness < 0:
        img = cv2.fillPoly(img, pts=pts, color=color)
    return img


def cv2_shapes(polygon, length, ref_pt, rotation=0, start_angle=0, end_angle=360):
    if polygon in CV2_OVAL_SHAPES:
        if polygon == 'circle':
            length = (length, length)
        params = {
            'center': ref_pt, 'axes': length, 'angle': np.rad2deg(rotation),
            'startAngle': start_angle, 'endAngle': end_angle
        }
        return cv2.ellipse, params
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
