"""
Testing the CLIP network with different text prompts.
"""

import numpy as np
import glob
import argparse
import ntpath
import sys
import os

from skimage import io
from PIL import Image

import torch
from torch.utils.tensorboard import SummaryWriter
import clip


def main(argv):
    parser = argparse.ArgumentParser(description='Testing CLIP.')
    parser.add_argument('--val_dir', required=True, type=str)
    parser.add_argument('--test_file', required=True, type=str)
    parser.add_argument('--text_path', required=True, type=str)
    parser.add_argument('--out_dir', default='outputs', type=str)
    parser.add_argument('--clip_arch', default='ViT-B/32', type=str)
    parser.add_argument('--bg', default='128', type=int)

    args = parser.parse_args(argv)

    os.makedirs(os.path.join(args.out_dir, 'eval'), exist_ok=True)

    _main_worker(args)


def _main_worker(args):
    model, preprocess = clip.load(args.clip_arch)
    model.cuda().eval()

    # reading the test file
    test_colours = np.loadtxt(args.test_file, delimiter=',')
    if test_colours.shape[0] == 3:
        test_colours = test_colours.T
    if test_colours.shape[1] != 3:
        sys.exit('Unsupported test file %s with size %s' % (args.test_file, test_colours.shape))

    labels = np.loadtxt(args.text_path, dtype=str)

    out_file = '%s/eval/text_probs_%.3d.npy' % (args.out_dir, args.bg)
    old_results = np.load(out_file, allow_pickle=True)[0] if os.path.exists(out_file) else dict()

    tb_writer = SummaryWriter(os.path.join(args.out_dir, 'test'))

    all_img_paths = [*glob.glob(args.val_dir + '/*.png'), *glob.glob(args.val_dir + '/*.gif')]
    all_text_probls = dict()
    for img_ind, img_path in enumerate(sorted(all_img_paths)):
        image_name = ntpath.basename(img_path)[:-4]
        if image_name in old_results.keys():
            all_text_probls[image_name] = old_results[image_name]
            continue
        else:
            print(img_path)
            text_probs = _one_image(img_path, test_colours, labels, model, preprocess, args.bg,
                                    tb_writer if img_ind < 10 else None)
            all_text_probls[image_name] = text_probs

        np.save(out_file, [all_text_probls])


def _one_image(img_path, test_colours, labels, model, preprocess, bg_lum, tb_writer):
    image = io.imread(img_path)
    image_mask = image == 255

    bg_img = np.zeros((*image.shape, 3), dtype='uint8')
    bg_img[:, :] = bg_lum

    images = []
    for test_colour in test_colours:
        image_vis = bg_img.copy()
        image_vis[image_mask] = test_colour
        images.append(preprocess(Image.fromarray(image_vis)))

    text_descriptions = [f"This is a {label} object" for label in labels]
    text_tokens = clip.tokenize(text_descriptions).cuda()
    image_input = torch.tensor(np.stack(images)).cuda()
    if tb_writer is not None:
        tb_writer.add_images('{}'.format(ntpath.basename(img_path)[:-4]), image_input, 0)

    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
        text_features = model.encode_text(text_tokens).float()

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs_raw = image_features @ text_features.T

    return text_probs_raw.cpu().numpy()


if __name__ == '__main__':
    main(sys.argv[1:])
