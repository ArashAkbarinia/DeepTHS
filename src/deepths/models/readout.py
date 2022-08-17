"""

"""

import numpy as np

import torch
import torch.nn as nn

from . import pretrained_models as pretraineds


class BackboneNet(nn.Module):
    def __init__(self, architecture, weights):
        super(BackboneNet, self).__init__()

        model = pretraineds.get_pretrained_model(architecture, weights)
        if '_scratch' in architecture:
            architecture = architecture.replace('_scratch', '')
        self.architecture = architecture
        self.backbone = pretraineds.get_backbone(architecture, model)
        self.in_type = self.set_img_type(self.backbone)

    def set_img_type(self, model):
        return model.conv1.weight.dtype if 'clip' in self.architecture else torch.float32

    def check_img_type(self, x):
        return x.type(self.in_type) if 'clip' in self.architecture else x

    def extract_features(self, x):
        x = x.to(next(self.parameters()).device)
        return self.backbone(self.check_img_type(x)).float()

    def extract_features_flatten(self, x):
        x = self.extract_features(x)
        x = x.view(x.size(0), -1)
        return x


class FeatureExtractor(BackboneNet):
    def forward(self, x):
        return self.extract_features(x)


class ReadOutNet(BackboneNet):
    def __init__(self, architecture, target_size, transfer_weights):
        super(ReadOutNet, self).__init__(architecture, transfer_weights[0])

        self.backbone, self.out_dim = pretraineds.model_features(
            self.backbone, architecture, transfer_weights[1], target_size
        )


class ClassifierNet(ReadOutNet):
    def __init__(self, architecture, target_size, transfer_weights, input_nodes, classifier,
                 num_classes):
        super(ClassifierNet, self).__init__(architecture, target_size, transfer_weights)

        self.input_nodes = input_nodes

        if classifier == 'nn':
            org_classes = np.prod(self.out_dim)
            self.fc = nn.Linear(int(org_classes * self.input_nodes), num_classes)
        else:
            self.fc = None  # e.g. for SVM

    def do_classifier(self, x):
        return x if self.fc is None else self.fc(x)


def load_model(weights, target_size, net_class, classifier):
    print('Loading test model from %s!' % weights)
    checkpoint = torch.load(weights, map_location='cpu')
    architecture = checkpoint['arch']
    transfer_weights = checkpoint['transfer_weights']

    model = net_class(architecture, target_size, transfer_weights, classifier)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    return model
