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
        return torch.flatten(x, start_dim=1)


class ActivationLoader(BackboneNet):
    def forward(self, x):
        return self.extract_features(x)


class ReadOutNet(BackboneNet):
    def __init__(self, architecture, target_size, transfer_weights, pooling=None):
        super(ReadOutNet, self).__init__(architecture, transfer_weights[0])

        if len(transfer_weights) == 2:
            self.backbone, self.out_dim = pretraineds.model_features(
                self.backbone, architecture, transfer_weights[1], target_size
            )
        else:
            self.act_dict, self.out_dim = pretraineds.mix_features(
                self.backbone, architecture, transfer_weights[1:], target_size
            )

        if pooling is None:
            if hasattr(self, 'act_dict'):
                sys.exit('With mix features, pooling must be set!')
            self.pool = None
        else:
            pool_size = pooling.split('_')[1:]
            pool_size = (int(pool_size[0]), int(pool_size[1]))

            if 'avg' in pooling:
                self.pool = nn.AdaptiveAvgPool2d(pool_size)
            elif 'max' in pooling:
                self.pool = nn.AdaptiveMaxPool2d(pool_size)
            else:
                sys.exit('Pooling %s not supported!' % pooling)

            if hasattr(self, 'act_dict'):  # assuming there is always pooling when mix features
                total_dim = 0
                for odim in self.out_dim:
                    if type(odim) is int:
                        total_dim += odim
                    else:
                        tmp_size = 1 if len(odim) < 3 else pool_size[0] * pool_size[1]
                        total_dim += (odim[0] * tmp_size)
                self.out_dim = (total_dim, 1)
            else:
                self.out_dim = (self.out_dim[0], *pool_size)

    def _do_pool(self, x):
        if self.pool is None or len(x.shape) < 3:
            return x
        if len(x.shape) == 3:
            x = x.unsqueeze(dim=-1)
        return self.pool(x)

    def extract_features(self, x):
        x = super(ReadOutNet, self).extract_features(x)
        if hasattr(self, 'act_dict'):
            xs = []
            for val in self.act_dict.values():
                if len(val.shape) >= 3:
                    val = self._do_pool(val)
                xs.append(torch.flatten(val, start_dim=1))
            x = torch.cat(xs, dim=1)
        else:
            x = self._do_pool(x)
        return x


class FeatureExtractor(ReadOutNet):
    def forward(self, x, pooling=None, flatten=True):
        x = self.extract_features(x)
        if pooling is not None:
            x = pooling(x)
        if flatten:
            x = torch.flatten(x, start_dim=1)
        return x


class ClassifierNet(ReadOutNet):
    def __init__(self, input_nodes, num_classes, classifier, **kwargs):
        super(ClassifierNet, self).__init__(**kwargs)

        self.input_nodes = input_nodes
        self.feature_units = np.prod(self.out_dim)
        if classifier == 'nn':
            self.fc = nn.Linear(int(self.feature_units * self.input_nodes), num_classes)
        else:
            self.fc = None  # e.g. for SVM

    def do_features(self, x):
        x = self.extract_features(x)
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

    readout_kwargs = _readout_kwargs(architecture, target_size, transfer_weights, pooling)
    classifier_kwargs = {'classifier': classifier}
    model = net_class(*extra_params, classifier_kwargs, readout_kwargs)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    return model


def make_model(net_class, args, *extra_params):
    if args.test_net:
        return load_model(net_class, args.test_net, args.target_size)
    else:
        readout_kwargs = _readout_kwargs(
            args.architecture, args.target_size, args.transfer_weights, args.pooling
        )
        classifier_kwargs = {'classifier': args.classifier}
        return net_class(*extra_params, classifier_kwargs, readout_kwargs)


def _readout_kwargs(architecture, target_size, transfer_weights, pooling):
    return {
        'architecture': architecture,
        'target_size': target_size,
        'transfer_weights': transfer_weights,
        'pooling': pooling
    }
