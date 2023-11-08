"""
Different colour spaces.
"""

import numpy as np

import cv2

rgb_from_dkl = np.array(
    [[+0.49995000, +0.50001495, +0.49999914],
     [+0.99998394, -0.29898596, +0.01714922],
     [-0.17577361, +0.15319546, -0.99994349]]
)

dkl_from_rgb = np.array(
    [[0.4251999971, +0.8273000025, +0.2267999991],
     [1.4303999955, -0.5912000011, +0.7050999939],
     [0.1444000069, -0.2360000005, -0.9318999983]]
)

# https://en.wikipedia.org/wiki/YCoCg
ycc_from_rgb = np.array(
    [[+0.25, +0.50, +0.25],
     [+0.50, +0.00, -0.50],
     [-0.25, +0.50, -0.25]]
).T

rgb_from_ycc = np.array(
    [[+1.0, +1.0, -1.0],
     [+1.0, +0.0, +1.0],
     [+1.0, -1.0, -1.0]]
).T

# http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
# RGB Working Space: NTSC RGB - WhitePoint: C
rgb_from_xyz = np.array(
    [[1.9099961, -0.9846663, 0.0583056],
     [-0.5324542, 1.999171, -0.1183781],
     [-0.2882091, -0.0283082, 0.8975535]]
)

xyz_from_rgb = np.array(
    [[0.6068909, 0.2989164, 0.],
     [0.1735011, 0.586599, 0.0660957],
     [0.200348, 0.1144845, 1.1162243]]
)


def rgb2xyz(x):
    return np.dot(x, xyz_from_rgb)


def xyz2rgb(x):
    return np.dot(x, rgb_from_xyz)


def xyy2xyz(x):
    xyz = np.zeros(x.shape, x.dtype)
    xyz[..., 0] = (x[..., 0] * x[..., 2]) / x[..., 1]
    xyz[..., 1] = x[..., 2].copy()
    xyz[..., 2] = ((1 - x[..., 0] - x[..., 1]) * x[..., 2]) / x[..., 1]
    return xyz


def xyz2xyy(x):
    xyy = np.zeros(x.shape, x.dtype)
    xyy[..., 0] = x[..., 0] / (x[..., 0] + x[..., 1] + x[..., 2])
    xyy[..., 1] = x[..., 1] / (x[..., 0] + x[..., 1] + x[..., 2])
    xyy[..., 2] = x[..., 1].copy()
    return xyy


def rgb2xyy(x):
    return xyz2xyy(np.dot(x, xyz_from_rgb))


def xyy2rgb(x):
    return np.dot(xyy2xyz(x), rgb_from_xyz)


def rgb012dkl(x):
    return np.dot(x, dkl_from_rgb)


def rgb2dkl(x):
    return rgb012dkl(rgb2double(x))


def rgb2dkl01(x):
    x = rgb2dkl(x)
    x /= 2
    x[..., 1] += 0.5
    x[..., 2] += 0.5
    return x


def dkl2rgb(x):
    return uint8im(dkl2rgb01(x))


def dkl2rgb01(x):
    x = np.dot(x, rgb_from_dkl)
    return clip01(x)


def dkl012rgb01(x):
    x = x.copy()
    x[..., 1] -= 0.5
    x[..., 2] -= 0.5
    x *= 2
    return dkl2rgb01(x)


def dkl012rgb(x):
    return uint8im(dkl012rgb01(x))


def rgb012ycc(x):
    return np.dot(x, ycc_from_rgb)


def rgb2ycc(x):
    return rgb012ycc(rgb2double(x))


def rgb2ycc01(x):
    x = rgb2ycc(x)
    x[..., 1] += 0.5
    x[..., 2] += 0.5
    return x


def ycc2rgb(x):
    return uint8im(ycc2rgb01(x))


def ycc2rgb01(x):
    x = np.dot(x, rgb_from_ycc)
    return clip01(x)


def ycc012rgb(x):
    return uint8im(ycc012rgb01(x))


def ycc012rgb01(x):
    x = x.copy()
    x[..., 1] -= 0.5
    x[..., 2] -= 0.5
    return ycc2rgb01(x)


def rgb2double(x):
    if x.dtype == 'uint8':
        x = np.float32(x) / 255
    else:
        assert x.max() <= 1, 'rgb must be either uint8 or in the range of [0 1]'
    return x


def rgb2hsv01(x):
    x = x.copy()
    x = rgb2double(x)
    x = np.float32(cv2.cvtColor(x.astype('float32'), cv2.COLOR_RGB2HSV))
    x[..., 0] /= 360
    return x


def hsv012rgb(x):
    return uint8im(hsv012rgb01(x))


def hsv012rgb01(x):
    x = x.copy()
    x[..., 0] *= 360
    x = cv2.cvtColor(x.astype('float32'), cv2.COLOR_HSV2RGB)
    return clip01(x)


def identity(x):
    return x


def clip01(x):
    x = np.maximum(x, 0)
    x = np.minimum(x, 1)
    return x


def uint8im(image):
    image = clip01(image)
    image *= 255
    return np.uint8(image)
