"""
Dataloader for the orientation discrimination task.
"""

import numpy as np

from skimage import io

from .binary_shapes import ShapeTrain
from . import dataset_utils


class ShapeOddOneOut(ShapeTrain):

    def __init__(self, root, transform=None, **kwargs):
        ShapeTrain.__init__(self, root, transform=transform, **kwargs)
        self.num_stimuli = 4

    def __getitem__(self, item):
        target_path = self.img_paths[item]
        masks = []
        for i in range(self.num_stimuli):
            mask_img = io.imread(target_path)
            rows, cols = mask_img.shape[:2]
            mask = np.zeros((rows, cols), dtype=mask_img.dtype)
            mask = dataset_utils.merge_fg_bg(mask, mask_img, 'centre')
            if i == 0:
                mask = dataset_utils.rotate_img(mask, np.random.randint(1, 180))
            masks.append(mask)

        # set the colours
        colour = self._get_target_colour()
        bg = self._unique_bg([colour])
        imgs = self._mul_train_imgs(masks, colour, colour, bg)

        inds = dataset_utils.shuffle(list(np.arange(0, self.num_stimuli)))
        # the target is always added the first element in the imgs list
        target = inds.index(0)
        imgs = [imgs[i] for i in inds]
        return *imgs, target


def train_val_set(root, target_size, preprocess, **kwargs):
    transform = dataset_utils.eval_preprocess(target_size, preprocess)
    return ShapeOddOneOut(root, transform, **kwargs)
