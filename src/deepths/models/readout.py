"""

"""

import sys
import numpy as np

import torch
import torch.nn as nn

from . import pretrained_models as pretraineds


class ReadOutNet(nn.Module):
    def __init__(self, architecture, target_size, transfer_weights):
        super(ReadOutNet, self).__init__()

        self.architecture = architecture

        model = pretraineds.get_pretrained_model(architecture, transfer_weights[0])
        if '_scratch' in architecture:
            architecture = architecture.replace('_scratch', '')
        model = pretraineds.get_backbone(architecture, model)
        self.in_type = self.set_img_type(model)

        # TODO better handing the layer
        layer = transfer_weights[1]

        if layer == 'fc':
            features = model
            if hasattr(model, 'num_classes'):
                out_dim = model.num_classes
            else:
                last_layer = list(model.children())[-1]
                if type(last_layer) is torch.nn.modules.container.Sequential:
                    out_dim = last_layer[-1].out_features
                else:
                    out_dim = last_layer.out_features
        elif (
                'fcn_' in architecture or 'deeplab' in architecture
                or 'resnet' in architecture or 'resnext' in architecture
                or 'taskonomy_' in architecture
        ):
            features, out_dim = pretraineds.resnet_features(model, architecture, layer, target_size)
        elif 'regnet' in architecture:
            features, out_dim = pretraineds.regnet_features(model, layer, target_size)
        elif 'vgg' in architecture:
            features, out_dim = pretraineds.vgg_features(model, layer, target_size)
        elif 'vit_' in architecture:
            features, out_dim = pretraineds.vit_features(model, layer, target_size)
        elif 'clip' in architecture:
            features, out_dim = pretraineds.clip_features(model, architecture, layer, target_size)
        else:
            sys.exit('Unsupported network %s' % architecture)
        self.out_dim = out_dim
        self.features = features

    def set_img_type(self, model):
        return model.conv1.weight.dtype if 'clip' in self.architecture else torch.float32

    def check_img_type(self, x):
        return x.type(self.in_type) if 'clip' in self.architecture else x

    def extract_features(self, x):
        x = x.to(next(self.parameters()).device)
        return self.features(self.check_img_type(x)).float()

    def extract_features_flatten(self, x):
        x = self.extract_features(x)
        x = x.view(x.size(0), -1)
        return x


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
