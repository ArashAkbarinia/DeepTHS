"""
Generating different patterns/textures.
"""

import numpy as np


def repeat_to_img_size(pattern_img, img_size):
    if img_size is None:
        return pattern_img
    repeats = ((img_size[0] // pattern_img.shape[0]) + 1, (img_size[1] // pattern_img.shape[1]) + 1)
    return np.tile(pattern_img, repeats)[:img_size[0], :img_size[1]]


def wave(img_size, height, gap, peak=1, length=1):
    cols = (height - 1) * 2 * length + peak
    wave_img = np.zeros((height + gap, cols))
    peack_scol = (height - 1) * length
    peack_ecol = peack_scol + peak
    wave_img[0, peack_scol:peack_ecol] = 1
    for h in range(1, height):
        sind = (h - 1) * length + 1
        cols_ind = [
            *[peack_scol - i for i in range(sind, sind + length)],
            *[peack_ecol + i - 1 for i in range(sind, sind + length)]
        ]
        wave_img[h, cols_ind] = 1
    return repeat_to_img_size(wave_img, img_size)


def herringbone(img_size, height, peak=1, length=1):
    peack_scol = (height - 1) * length
    peack_ecol = peack_scol + peak
    herringbone_img = wave(None, height, 0, peak, length)
    herringbone_img[:, peack_scol:peack_ecol] = 1
    herringbone_img[:, 0] = 1
    return repeat_to_img_size(herringbone_img, img_size)


def line(img_size, gap, thickness=1):
    line_img = np.zeros((thickness + gap, img_size[1]))
    line_img[0:thickness] = 1
    return repeat_to_img_size(line_img, img_size)


def grid(img_size, gaps, thicknesses=(1, 1)):
    grid_img = np.zeros((thicknesses[0] + gaps[0], thicknesses[1] + gaps[1]))
    grid_img[0:thicknesses[0]] = 1
    grid_img[:, 0:thicknesses[0]] = 1
    return repeat_to_img_size(grid_img, img_size)


def brick(img_size, gaps, thicknesses=(1, 1)):
    rows = thicknesses[0] + gaps[0]
    half_gap = gaps[1] // 2
    brick_img = np.zeros((rows * 2, thicknesses[1] + gaps[1]))
    # setting the rows
    brick_img[0:thicknesses[0]] = 1
    srow = gaps[0] + thicknesses[0]
    brick_img[srow:srow + thicknesses[0]] = 1
    # setting columns
    brick_img[:rows, :thicknesses[1]] = 1
    brick_img[srow:, half_gap:half_gap + thicknesses[1]] = 1
    return repeat_to_img_size(brick_img, img_size)
