"""
A collection of architectures to do the contrast discrimination task.
"""

import torch
from .readout import ReadOutNetwork


def _load_csf_model(weights, target_size, net_type, classifier):
    print('Loading CSF test model from %s!' % weights)
    checkpoint = torch.load(weights, map_location='cpu')
    architecture = checkpoint['arch']
    transfer_weights = checkpoint['transfer_weights']

    net_class = ContrastDiscrimination if net_type == 'ContrastDiscrimination' else GratingDetector
    model = net_class(architecture, target_size, transfer_weights, classifier)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    return model


def load_contrast_discrimination(weights, target_size, classifier):
    return _load_csf_model(weights, target_size, 'ContrastDiscrimination', classifier)


def load_grating_detector(weights, target_size, classifier):
    return _load_csf_model(weights, target_size, 'GratingDetector', classifier)


class ContrastDiscrimination(ReadOutNetwork):
    def __init__(self, architecture, target_size, transfer_weights, classifier):
        super(ContrastDiscrimination, self).__init__(
            architecture, target_size, transfer_weights, 2, classifier, 2
        )

    def forward(self, x0, x1):
        x0 = self.extract_features(x0)
        x0 = x0.view(x0.size(0), -1).float()
        x1 = self.extract_features(x1)
        x1 = x1.view(x1.size(0), -1).float()
        x = torch.cat([x0, x1], dim=1)
        return self.do_classifier(x)


class GratingDetector(ReadOutNetwork):
    def __init__(self, architecture, target_size, transfer_weights, classifier):
        super(GratingDetector, self).__init__(
            architecture, target_size, transfer_weights, 1, classifier, 2
        )

    def forward(self, x):
        x = self.extract_features(x)
        x = x.view(x.size(0), -1).float()
        return self.do_classifier(x)
