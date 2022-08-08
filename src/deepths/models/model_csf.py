"""
A collection of architectures to do the contrast discrimination task.
"""

import torch

from . import readout


def _load_csf_model(weights, target_size, net_type, classifier):
    net_class = ContrastDiscrimination if net_type == 'ContrastDiscrimination' else GratingDetector
    return readout.load_model(weights, target_size, net_class, classifier)


def load_contrast_discrimination(weights, target_size, classifier):
    return _load_csf_model(weights, target_size, 'ContrastDiscrimination', classifier)


def load_grating_detector(weights, target_size, classifier):
    return _load_csf_model(weights, target_size, 'GratingDetector', classifier)


class ContrastDiscrimination(readout.ClassifierNet):
    def __init__(self, architecture, target_size, transfer_weights, classifier):
        super(ContrastDiscrimination, self).__init__(
            architecture, target_size, transfer_weights, 2, classifier, 2
        )

    def forward(self, x0, x1):
        x0 = self.extract_features_flatten(x0)
        x1 = self.extract_features_flatten(x1)
        x = torch.cat([x0, x1], dim=1)
        return self.do_classifier(x)


class GratingDetector(readout.ClassifierNet):
    def __init__(self, architecture, target_size, transfer_weights, classifier):
        super(GratingDetector, self).__init__(
            architecture, target_size, transfer_weights, 1, classifier, 2
        )

    def forward(self, x):
        x = self.extract_features_flatten(x)
        return self.do_classifier(x)
