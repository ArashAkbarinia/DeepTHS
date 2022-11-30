"""

"""

import sys
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
        x = torch.flatten(x, start_dim=1)
        return x


class ActivationLoader(BackboneNet):
    def forward(self, x):
        return self.extract_features(x)


class ReadOutNet(BackboneNet):
    def __init__(self, architecture, target_size, transfer_weights):
        super(ReadOutNet, self).__init__(architecture, transfer_weights[0])

        self.backbone, self.out_dim = pretraineds.model_features(
            self.backbone, architecture, transfer_weights[1], target_size
        )


class FeatureExtractor(ReadOutNet):
    def forward(self, x, pooling=None, flatten=True):
        x = self.extract_features(x)
        if pooling is not None:
            x = pooling(x)
        if flatten:
            x = torch.flatten(x, start_dim=1)
        return x


class ClassifierNet(ReadOutNet):
    def __init__(self, input_nodes, num_classes, classifier, pooling=None, **kwargs):
        super(ClassifierNet, self).__init__(**kwargs)

        self.input_nodes = input_nodes
        if pooling is not None and len(self.out_dim) == 3:
            pool_size = pooling.split('_')[1:]
            pool_size = (int(pool_size[0]), int(pool_size[1]))
            self.out_dim = (self.out_dim[0], *pool_size)
            if 'avg' in pooling:
                self.pool = nn.AdaptiveAvgPool2d(pool_size)
            elif 'max' in pooling:
                self.pool = nn.AdaptiveMaxPool2d(pool_size)
            else:
                sys.exit('Pooling %s not supported!' % pooling)
        else:
            self.pool = None

        self.feature_units = np.prod(self.out_dim)
        if classifier == 'nn':
            self.fc = nn.Linear(int(self.feature_units * self.input_nodes), num_classes)
        else:
            self.fc = None  # e.g. for SVM

    def do_features(self, x):
        x = self.extract_features(x)
        x = x if self.pool is None else self.pool(x)
        return torch.flatten(x, start_dim=1)

    def do_classifier(self, x):
        return x if self.fc is None else self.fc(x)


def load_model(net_class, weights, target_size):
    print('Loading test model from %s!' % weights)
    checkpoint = torch.load(weights, map_location='cpu')
    architecture = checkpoint['arch']
    transfer_weights = checkpoint['transfer_weights']
    classifier = checkpoint['net']['classifier'] if hasattr(checkpoint, 'net') else 'nn'
    pooling = checkpoint['net']['pooling'] if hasattr(checkpoint, 'net') else None
    extra_params = checkpoint['net']['extra'] if hasattr(checkpoint, 'net') else []

    readout_kwargs = _readout_kwargs(architecture, target_size, transfer_weights)
    classifier_kwargs = {
        'classifier': classifier,
        'pooling': pooling
    }
    model = net_class(*extra_params, classifier_kwargs, readout_kwargs)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    return model


def make_model(net_class, args, *extra_params):
    if args.test_net:
        return load_model(net_class, args.test_net, args.target_size)
    else:
        readout_kwargs = _readout_kwargs(args.architecture, args.target_size, args.transfer_weights)
        classifier_kwargs = {
            'classifier': args.classifier,
            'pooling': args.pooling
        }
        return net_class(*extra_params, classifier_kwargs, readout_kwargs)


def _readout_kwargs(architecture, target_size, transfer_weights):
    return {
        'architecture': architecture,
        'target_size': target_size,
        'transfer_weights': transfer_weights
    }
