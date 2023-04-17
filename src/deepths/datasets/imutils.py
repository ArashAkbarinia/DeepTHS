"""
Image manipulation functions.
"""

import numpy as np
import math
import cv2

from . import filters


def _img_max_val(image):
    max_val = np.maximum(np.max(image), 1)
    for bits in [8, 16, 32, 64]:
        if image.dtype == 'uint%d' % bits:
            max_val = (2 ** bits) - 1
            break
    return max_val


def im2double(image):
    return np.float32(image) / _img_max_val(image)


def double2im(image, org_img):
    return (image * _img_max_val(org_img)).astype(org_img.dtype)


def process_img(fun, in_image, *args, **kwargs):
    image = im2double(in_image.copy())
    image = fun(image, *args, **kwargs)
    return double2im(fun(image, *args, **kwargs), in_image)


def adjust_contrast(in_image, amount):
    assert np.all(amount >= 0.0), 'contrast_level too low.'
    assert np.all(amount <= 1.0), 'contrast_level too high.'
    return in_image if amount == 1 else process_img(_adjust_contrast, in_image, amount)


def _adjust_contrast(image, amount):
    return (1 - amount) / 2.0 + np.multiply(image, amount)


def adjust_gamma(in_image, amount):
    return in_image if amount == 1 else process_img(_adjust_gamma, in_image, amount)


def _adjust_gamma(image, amount):
    return image ** amount


def gaussian_blur(in_image, **kwargs):
    return process_img(_gaussian_blur, in_image, **kwargs)


def _gaussian_blur(image, **kwargs):
    return cv2.filter2D(image, -1, filters.gaussian_kernel2(**kwargs))


def filter_img_sf(img, **kwargs):
    img_norm = (img.copy() - 0.5) / 0.5
    if len(img_norm.shape) > 2:
        img_back = np.zeros(img_norm.shape)
        for i in range(img_norm.shape[2]):
            img_back[:, :, i] = _filter_chn_sf(img_norm[:, :, i], **kwargs)
    else:
        img_back = _filter_chn_sf(img_norm, **kwargs)
    img_back = (img_back * 0.5) + 0.5
    return img_back


def _filter_chn_sf(img, **kwargs):
    img_freq = np.fft.fft2(img)
    img_freq_cent = np.fft.fftshift(img_freq)
    img_sf_filtered = _cutoff_chn_fourier(img_freq_cent, **kwargs)

    img_back = np.real(np.fft.ifft2(np.fft.ifftshift(img_sf_filtered)))
    img_back[img_back < -1] = -1
    img_back[img_back > 1] = 1
    return img_back


def _cutoff_chn_fourier(img, hsf_cut, lsf_cut):
    inverse = False
    if lsf_cut < 0:
        lsf_cut = abs(lsf_cut)
        inverse = True
    mask_img = ring_mask(img, inner=lsf_cut, outer=hsf_cut)
    if inverse:
        mask_img = ~mask_img
    img_sf_filtered = np.multiply(img, mask_img)
    return img_sf_filtered


def ring_mask(img, inner, outer):
    rows = img.shape[0]
    cols = img.shape[1]
    smaller_side = np.minimum(rows, cols)
    centre = (int(math.floor(cols / 2)), int(math.floor(rows / 2)))

    if inner == 0:
        mask_in = np.ones(img.shape, np.uint8)
        in_length = 0
    else:
        if type(inner) is float and inner < 1.0:
            in_length = int(math.floor(inner * smaller_side * 0.5))
        else:
            in_length = int(inner - 1)
        mask_in = np.zeros(img.shape, np.uint8)
        mask_in = 1 - cv2.circle(mask_in, centre, in_length, (1, 1, 1), -1)

    if outer == 0:
        mask_out = np.ones(img.shape, np.uint8)
    else:
        if type(outer) is float and abs(outer) < 1.0:
            outer = 1 - outer
            out_length = int(math.floor(outer * smaller_side * 0.5))
        elif outer < 0:
            out_length = centre[0] - int(abs(outer))
        else:
            out_length = in_length + int(outer)
        mask_out = np.zeros(img.shape, np.uint8)
        mask_out = cv2.circle(mask_out, centre, out_length, (1, 1, 1), -1)

    mask_img = np.logical_and(mask_in, mask_out)
    return mask_img


def centre_pixel(img_size):
    return img_size[0] // 2, img_size[1] // 2
