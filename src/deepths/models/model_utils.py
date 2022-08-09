"""
Utility functions for networks.
"""

import sys
import os

import torch
import torch.nn as nn
import torchvision.models as pmodels
import torchvision.models.segmentation as seg_models

from . import resnet
from . import pretrained_features


def get_mean_std(colour_space, colour_vision=None):
    if colour_space in ['imagenet_rgb']:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif colour_space in ['clip_rgb']:
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
    elif colour_space in ['taskonomy_rgb']:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif colour_space in ['rgb', 'grey3'] or colour_vision in ['trichromat']:
        mean = [0.5, 0.5, 0.5]
        std = [0.25, 0.25, 0.25]
    elif colour_space in ['grey'] or colour_vision in ['monochromat']:
        mean = [0.5]
        std = [0.25]
    elif 'dichromat' in colour_vision:
        mean = [0.5, 0.5]
        std = [0.25, 0.25]
    else:
        # just create mean and std based on number of channels
        mean = []
        std = []
        for i in range(colour_space):
            mean.append(0.5)
            std.append(0.25)
    return mean, std


def resnet_conv_ind(module, layer_num, conv_num):
    if (isinstance(module[layer_num], pmodels.resnet.BasicBlock) or
            isinstance(module[layer_num], resnet.BasicBlock)):
        conv_ind = (conv_num - 1) * 3 + 1
    else:
        conv_ind = (conv_num - 1) * 2 + 1
    return conv_ind


def resnet_conv_layer(module, layer_num, conv_num):
    sub_layer = [*module[:layer_num]]
    sub_layer = nn.Sequential(*sub_layer)

    module_list = list(module[layer_num].children())
    conv_ind = resnet_conv_ind(module, layer_num, conv_num)
    sub_conv = nn.Sequential(*module_list[:conv_ind])
    return sub_layer, sub_conv


def resnet_bn_layer(module, layer_num, conv_num):
    sub_layer = [*module[:layer_num]]
    sub_layer = nn.Sequential(*sub_layer)

    module_list = list(module[layer_num].children())
    bn_ind = resnet_conv_ind(module, layer_num, conv_num) + 1
    sub_bn = nn.Sequential(*module_list[:bn_ind])
    return sub_layer, sub_bn


def which_architecture(network_name, customs=None):
    if customs is None:
        model = pmodels.__dict__[network_name](pretrained=False)
    else:
        pooling_type = customs['pooling_type']
        num_classes = customs['num_classes']
        if 'in_chns' in customs:
            in_chns = customs['in_chns']
        else:
            # assuming if it doesn't exist, it's 3
            in_chns = 3
        if 'kernel_size' in customs:
            kernel_size = customs['kernel_size']
        else:
            # assuming if it doesn't exist, it's 3
            kernel_size = 7
        if 'stride' in customs:
            stride = customs['stride']
        else:
            # assuming if it doesn't exist, it's 3
            stride = 2
        # differentiating between custom models and nominal one
        if 'blocks' in customs and customs['blocks'] is not None:
            num_kernels = customs['num_kernels']
            model = resnet.__dict__[network_name](
                customs['blocks'], pretrained=False, pooling_type=pooling_type,
                in_chns=in_chns, num_classes=num_classes, inplanes=num_kernels,
                kernel_size=kernel_size, stride=stride
            )
        else:
            model = resnet.__dict__[network_name](
                pretrained=False, pooling_type=pooling_type,
                in_chns=in_chns, num_classes=num_classes
            )
    return model


def which_network(network_name, task_type, **kwargs):
    if task_type == 'classification':
        model = which_network_classification(network_name, **kwargs)
    elif task_type == 'segmentation':
        model = which_network_segmentation(network_name, **kwargs)
    else:
        sys.exit('Task type %s is not supported.' % task_type)
    return model


def which_network_classification(network_name, num_classes):
    if os.path.isfile(network_name):
        print('Loading %s' % network_name)
        checkpoint = torch.load(network_name, map_location='cpu')
        customs = None
        if 'customs' in checkpoint:
            customs = checkpoint['customs']
            # TODO: num_classes is just for backward compatibility
            if 'num_classes' not in customs:
                customs['num_classes'] = num_classes
        model = which_architecture(checkpoint['arch'], customs=customs)

        model.load_state_dict(checkpoint['state_dict'])
    elif 'inception' in network_name:
        model = pmodels.__dict__[network_name](pretrained=True, aux_logits=False)
    else:
        model = pmodels.__dict__[network_name](pretrained=True)

    return model


def which_network_segmentation(network_name, num_classes):
    if os.path.isfile(network_name):
        checkpoint = torch.load(network_name, map_location='cpu')
        customs = None
        aux_loss = None
        if 'customs' in checkpoint:
            customs = checkpoint['customs']
            # TODO: num_classes is just for backward compatibility
            if 'num_classes' not in customs:
                customs['num_classes'] = num_classes
            if 'aux_loss' in customs:
                aux_loss = customs['aux_loss']
            backbone = customs['backbone']
        # TODO: for now only predefined models
        # model = which_architecture(checkpoint['arch'], customs=customs)
        model = resnet.__dict__[checkpoint['arch']](
            backbone, num_classes=num_classes, pretrained=False, aux_loss=aux_loss
        )

        model.load_state_dict(checkpoint['state_dict'])
    else:
        model = seg_models.__dict__[network_name](
            num_classes=num_classes, pretrained=True, aux_loss=True
        )

    return model


def out_hook(name, out_dict):
    def hook(model, input_x, output_y):
        out_dict[name] = output_y.detach()

    return hook


def resnet_hooks(model, layers, is_clip=False):
    act_dict = dict()
    rf_hooks = dict()
    model_layers = list(model.children())
    for layer in layers:
        l_ind = pretrained_features.resnet_layer(layer, is_clip=is_clip)
        rf_hooks[layer] = model_layers[l_ind].register_forward_hook(out_hook(layer, act_dict))
    return act_dict, rf_hooks


def clip_hooks(model, layers, architecture):
    if architecture.replace('clip_', '') in ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64']:
        act_dict, rf_hooks = resnet_hooks(model, layers, is_clip=True)
    return act_dict, rf_hooks


def register_model_hooks(model, architecture, layers):
    if pretrained_features.is_resnet_backbone(architecture):
        act_dict, rf_hooks = resnet_hooks(model, layers)
    elif 'clip' in architecture:
        act_dict, rf_hooks = clip_hooks(model, layers, architecture)
    else:
        sys.exit('Unsupported network %s' % architecture)
    return act_dict, rf_hooks
