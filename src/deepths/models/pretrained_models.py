"""

"""

import os
import sys
import numpy as np

import torch
import torch.nn as nn

from torchvision.models import segmentation
import torchvision.transforms.functional as torchvis_fun
import clip

from . import model_utils, pretrained_features
from . import vqvae
from .taskonomy import taskonomy_network


class ViTLayers(nn.Module):
    def __init__(self, parent_model, encoder_layer):
        super().__init__()
        self.parent_model = parent_model
        encoder_layer = encoder_layer + 1
        self.parent_model.encoder.layers = self.parent_model.encoder.layers[:encoder_layer]
        del self.parent_model.heads

    def forward(self, x):
        # Reshape and permute the input tensor
        x = self.parent_model._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.parent_model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.parent_model.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        return x


class ViTClipLayers(nn.Module):
    def __init__(self, parent_model, encoder_layer):
        super().__init__()
        self.parent_model = parent_model
        block = encoder_layer + 1
        self.parent_model.transformer.resblocks = self.parent_model.transformer.resblocks[:block]
        del self.parent_model.proj
        del self.parent_model.ln_post

    def forward(self, x):
        x = self.parent_model.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.parent_model.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.parent_model.positional_embedding.to(x.dtype)
        x = self.parent_model.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.parent_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = x[:, 0, :]

        return x


def vit_features(model, layer, target_size):
    encoder_layer = int(layer.replace('encoder', ''))
    features = ViTLayers(model, encoder_layer)
    out_dim = generic_features_size(features, target_size)
    return features, out_dim


def vgg_features(model, layer, target_size):
    if 'feature' in layer:
        layer = int(layer.replace('feature', '')) + 1
        features = nn.Sequential(*list(model.features.children())[:layer])
    elif 'classifier' in layer:
        layer = int(layer.replace('classifier', '')) + 1
        features = nn.Sequential(
            model.features, model.avgpool, nn.Flatten(1), *list(model.classifier.children())[:layer]
        )
    else:
        sys.exit('Unsupported layer %s' % layer)
    out_dim = generic_features_size(features, target_size)
    return features, out_dim


def generic_features_size(model, target_size, dtype=None):
    img = np.random.randint(0, 256, (target_size, target_size, 3)).astype('float32') / 255
    img = torchvis_fun.to_tensor(img).unsqueeze(0)
    if dtype is not None:
        img = img.cuda()
        img = img.type(dtype)
    model.eval()
    with torch.no_grad():
        out = model(img)
    return out[0].shape


def clip_features(model, network_name, layer, target_size):
    if layer == 'encoder':
        features = model
        if 'B32' in network_name or 'B16' in network_name or 'RN101' in network_name:
            out_dim = 512
        elif 'L14' in network_name or 'RN50x16' in network_name:
            out_dim = 768
        elif 'RN50x4' in network_name:
            out_dim = 640
        else:
            out_dim = 1024
    else:
        if network_name.replace('clip_', '') in ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64']:
            l_ind = pretrained_features.resnet_slice(layer, is_clip=True)
            features = nn.Sequential(*list(model.children())[:l_ind])
        else:
            block_layer = int(layer.replace('block', ''))
            features = ViTClipLayers(model, block_layer)
        out_dim = generic_features_size(features, target_size, model.conv1.weight.dtype)
    return features, out_dim


def regnet_features(model, layer, target_size):
    if 'stem' in layer:
        features = model.stem
    elif 'block' in layer:
        if layer == 'block1':
            layer = 1
        elif layer == 'block2':
            layer = 2
        elif layer == 'block3':
            layer = 3
        elif layer == 'block4':
            layer = 4
        features = nn.Sequential(model.stem, *list(model.trunk_output.children())[:layer])
    else:
        sys.exit('Unsupported layer %s' % layer)
    out_dim = generic_features_size(features, target_size)
    return features, out_dim


def resnet_features(model, layer, target_size):
    l_ind = pretrained_features.resnet_slice(layer)
    features = nn.Sequential(*list(model.children())[:l_ind])
    out_dim = generic_features_size(features, target_size)
    return features, out_dim


def get_pretrained_model(network_name, weights):
    if 'clip' in network_name:
        if 'B32' in network_name:
            clip_version = 'ViT-B/32'
        elif 'B16' in network_name:
            clip_version = 'ViT-B/16'
        elif 'L14' in network_name:
            clip_version = 'ViT-L/14'
        else:
            clip_version = network_name.replace('clip_', '')
        model, _ = clip.load(clip_version)
    elif 'taskonomy_' in network_name:
        # NOTE: always assumed pretrained
        feature_task = network_name.replace('taskonomy_', '')
        model = taskonomy_network.TaskonomyEncoder()
        feature_type_url = taskonomy_network.TASKONOMY_PRETRAINED_URLS[feature_task + '_encoder']
        checkpoint = torch.utils.model_zoo.load_url(feature_type_url, model_dir=None, progress=True)
        model.load_state_dict(checkpoint['state_dict'])
    elif os.path.isfile(weights):
        # FIXME: cheap hack!
        if 'vqvae' in network_name or 'vqvae' in weights:
            vqvae_info = torch.load(weights, map_location='cpu')

            backbone = {
                'arch_name': vqvae_info['backbone']['arach'],
                'layer_name': vqvae_info['backbone']['area'],
            }
            # hardcoded to test one type
            hidden = vqvae_info['backbone']['hidden']
            k = vqvae_info['backbone']['k']
            kl = vqvae_info['backbone']['kl']
            model = vqvae.Backbone_VQ_VAE(
                hidden, k=k, kl=kl, num_channels=3, colour_space='rgb2rgb',
                task=None, out_chns=3, cos_distance=False, use_decor_loss=False, backbone=backbone
            )
            model.load_state_dict(vqvae_info['state_dict'])
            print('Loaded the VQVAE model!')
        else:
            task = 'classification'
            num_classes = 1000
            if 'fcn_' in network_name or 'deeplab' in network_name:
                task = 'segmentation'
                num_classes = 21
            model = model_utils.which_network(weights, task, num_classes=num_classes)
    elif '_scratch' in network_name:
        model = model_utils.which_architecture(network_name.replace('_scratch', ''))
    elif 'fcn_' in network_name or 'deeplab' in network_name:
        model = segmentation.__dict__[network_name](pretrained=True)
    else:
        model = model_utils.which_network(weights, 'classification', num_classes=1000)
    return model


def get_backbone(network_name, model):
    if 'clip' in network_name:
        return model.visual
    elif 'vqvae' in network_name:
        return model.backbone_encoder.features
    elif 'fcn_' in network_name or 'deeplab' in network_name:
        return model.backbone
    return model


def model_features(model, architecture, layer, target_size):
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
    elif pretrained_features.is_resnet_backbone(architecture):
        features, out_dim = resnet_features(model, layer, target_size)
    elif 'regnet' in architecture:
        features, out_dim = regnet_features(model, layer, target_size)
    elif 'vgg' in architecture:
        features, out_dim = vgg_features(model, layer, target_size)
    elif 'vit_' in architecture:
        features, out_dim = vit_features(model, layer, target_size)
    elif 'clip' in architecture:
        features, out_dim = clip_features(model, architecture, layer, target_size)
    else:
        sys.exit('Unsupported network %s' % architecture)
    return features, out_dim
