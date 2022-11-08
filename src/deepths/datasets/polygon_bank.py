"""
Generating different polygons and geometrical shapes.
"""

import sys

import cv2

CV2_BASIC_SHAPES = ['circle', 'ellipse', 'square', 'rectangle']
CV2_CUSTOM_SHAPES = ['triangle']

SHAPES = [*CV2_BASIC_SHAPES, *CV2_CUSTOM_SHAPES]


def polygon_params(polygon, **kwargs):
    if polygon in CV2_BASIC_SHAPES:
        draw_fun, params = cv2_shapes(polygon, **kwargs)
    elif polygon in CV2_CUSTOM_SHAPES:
        draw_fun, params = cv2_polygons(**kwargs)
    else:
        sys.exit('Unsupported polygon to draw: %s' % polygon)
    return draw_fun, params


def cv2_polygons(pts):
    return cv2_filled_polygons, {'pts': pts}


def cv2_filled_polygons(img, pts, color, thickness):
    img = cv2.polylines(img, pts=pts, color=color, thickness=abs(thickness), isClosed=True)
    if thickness < 0:
        img = cv2.fillPoly(img, pts=pts, color=color)
    return img


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
