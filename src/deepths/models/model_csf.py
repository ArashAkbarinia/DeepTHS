"""
A collection of architectures to do the contrast discrimination task.
"""

import torch

from . import readout


def _load_csf_model(net_type, weights, target_size):
    net_class = ContrastDiscrimination if net_type == 'ContrastDiscrimination' else GratingDetector
    return readout.load_model(net_class, weights, target_size)


def load_contrast_discrimination(weights, target_size):
    return _load_csf_model('ContrastDiscrimination', weights, target_size)


def load_grating_detector(weights, target_size):
    return _load_csf_model('GratingDetector', weights, target_size)


class ContrastDiscrimination(readout.ClassifierNet):
    def __init__(self, classifier_kwargs, readout_kwargs):
        super(ContrastDiscrimination, self).__init__(2, 2, **classifier_kwargs, **readout_kwargs)

    def forward(self, x0, x1):
        x0 = self.do_features(x0)
        x1 = self.do_features(x1)
        x = torch.cat([x0, x1], dim=1)
        return self.do_classifier(x)


class GratingDetector(readout.ClassifierNet):
    def __init__(self, classifier_kwargs, readout_kwargs):
        super(GratingDetector, self).__init__(1, 2, **classifier_kwargs, **readout_kwargs)

    def forward(self, x):
        x = self.do_features(x)
        return self.do_classifier(x)
