"""
Optimisation of uniform colour space
"""

import numpy as np
import pandas as pd
import argparse
import os
import sys
import json

from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import colour as colour_science

import torch
import torch.nn as nn

from ..utils import colour_spaces

arch_areas = {
    'clip_RN50': [*['area%d' % i for i in range(0, 5)], 'encoder'],
    'clip_B32': [*['block%d' % i for i in [1, 4, 7, 10, 11]], 'encoder'],
    'resnet50': [*['area%d' % i for i in range(0, 5)], 'fc'],
    'vit_b_32': [*['block%d' % i for i in [1, 4, 7, 10, 11]], 'fc'],
    'taskonomy': [*['area%d' % i for i in range(0, 5)], 'encoder']
}


def load_model(path, return_info=False):
    model_info = torch.load(path, map_location='cpu')
    model = ColourSpaceNet(
        model_info['units'],
        model_info['nonlinearities'],
        model_info['mean_std'],
    )
    model.load_state_dict(model_info['state_dict'])
    if return_info:
        return model, model_info['units'], model_info['nonlinearities']
    return model


def load_human_data(path):
    data = read_test_pts(path)
    ref_pts = np.expand_dims(np.array([val['ref'] for val in data.values()]), axis=1)
    hot_cen, hot_bor = [], []
    for key, val in data.items():
        for pt in val['ext']:
            hot_cen.append(val['ref'])
            hot_bor.append(pt)
    hot_cen, hot_bor = np.array(hot_cen), np.array(hot_bor)
    return {'data': data, 'ref_pts': ref_pts,
            'hot_cen': hot_cen, 'hot_bor': hot_bor}


def compare_colour_discrimination(test_file, method, is_onehot_vector=False):
    if is_onehot_vector:
        df = pd.read_csv(test_file)
        human_data = {'hot_cen': df.iloc[:, :3].to_numpy(), 'hot_bor': df.iloc[:, 3:].to_numpy()}
    else:
        human_data = load_human_data(test_file)
    cen_pts, bor_pts = human_data['hot_cen'], human_data['hot_bor']
    illuminant = np.array([0.31271, 0.32902])
    cen_lab = colour_science.XYZ_to_Lab(colour_science.sRGB_to_XYZ(cen_pts), illuminant)
    bor_lab = colour_science.XYZ_to_Lab(colour_science.sRGB_to_XYZ(bor_pts), illuminant)
    pred = method_out(method, cen_pts, bor_pts, cen_lab, bor_lab)
    return pred


def method_out(method, cen_pts, bor_pts, cen_pts_lab, bor_pts_lab):
    checkpoint_path = '%s/model.pth' % method
    if not isinstance(method, str) or os.path.exists(checkpoint_path):
        network = load_model(checkpoint_path) if isinstance(method, str) else method
        cen_pts = pred_model(network, cen_pts)
        bor_pts = pred_model(network, bor_pts)
        pred = euc_distance(cen_pts, bor_pts)
    elif 'euc' in method:
        space = method.split('_')[1]
        if space == 'dkl':
            cen_pts = colour_spaces.rgb2dkl01(cen_pts)
            bor_pts = colour_spaces.rgb2dkl01(bor_pts)
        elif space == 'ycc':
            cen_pts = colour_spaces.rgb2ycc01(cen_pts)
            bor_pts = colour_spaces.rgb2ycc01(bor_pts)
        elif space == 'lab':
            cen_pts, bor_pts = cen_pts_lab, bor_pts_lab
        pred = euc_distance(cen_pts, bor_pts)
    else:
        pred = colour_science.delta_E(cen_pts_lab, bor_pts_lab, method=method)
    return pred


def compare_colour_difference(path, method):
    human_data = np.loadtxt(path, delimiter=',')
    gt, cen_pts, bor_pts = human_data[:, 0], human_data[:, 1:4], human_data[:, 4:7]
    pred = method_out(method, cen_pts, bor_pts, human_data[:, 7:10], human_data[:, 10:13])
    if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
        return 0, 0, 0
    pearsonr_corr, _ = stats.pearsonr(pred, gt)
    spearmanr_corr, _ = stats.spearmanr(pred, gt)
    return pearsonr_corr, spearmanr_corr, stress(pred, gt)


def compare_human_data(method, test_dir):
    # MacAdam 1942
    macadam_res = compare_colour_discrimination('%s/macadam_rgb_srgb.csv' % test_dir, method)
    # Luo-Rigg 1986
    luorigg_res = compare_colour_discrimination('%s/luorigg_rgb_srgb.csv' % test_dir, method)
    # Melgosa
    melgosa97_res = compare_colour_discrimination('%s/melgosa1997_rgb_srgb.csv' % test_dir, method)
    # Huang
    huang2012_res = compare_colour_discrimination('%s/huang2012_rgb_srgb.csv' % test_dir, method)
    # KTeam
    kteam_res = compare_colour_discrimination('%s/kteam_thresholds.csv' % test_dir, method, True)
    # MacAdam 1974
    macadam1974_res = compare_colour_difference('%s/macadam1974_srgb.csv' % test_dir, method)
    return {
        'colour_discrimination': {
            'MacAdam': macadam_res,
            'Luo-Rigg': luorigg_res,
            'Melgosa1997': melgosa97_res,
            'Huang2012': huang2012_res,
            'TeamK': kteam_res
        },
        'colour_difference': {
            'MacAdam1974': macadam1974_res
        }
    }


def stress(de, dv=None):
    if dv is None:
        dv = np.ones(len(de))
        # dv = np.ones(len(de)) * np.random.normal(size=len(de), loc=1.0, scale=0.1)
    # return np.sqrt(1 - (np.sum(de * dv) ** 2) / (np.sum(de ** 2) * np.sum(dv ** 2)))
    return colour_science.index_stress(de, dv)


def stress_torch(de, dv=None):
    if dv is None:
        dv = torch.ones(len(de))
        # dv = np.ones(len(de)) * np.random.normal(size=len(de), loc=1.0, scale=0.1)
    return torch.sqrt(1 - (torch.sum(de * dv) ** 2) / (torch.sum(de ** 2) * torch.sum(dv ** 2)))


def predict_human_data(methods, test_dir, discrimination, difference):
    predictions = {key: compare_human_data(method, test_dir) for key, method in methods.items()}
    # colour discrimination
    for method in predictions.keys():
        for db in predictions[method]['colour_discrimination'].keys():
            x = predictions[method]['colour_discrimination'][db]
            if discrimination == 'cv':  # coefficient of variation (CV)
                predictions[method]['colour_discrimination'][db] = np.std(x) / np.mean(x)
            elif discrimination == 'stress':
                predictions[method]['colour_discrimination'][db] = stress(x)
            elif discrimination == 'entropy':
                predictions[method]['colour_discrimination'][db] = stats.entropy(x)
            elif discrimination == 'mad':
                predictions[method]['colour_discrimination'][db] = stats.median_abs_deviation(x)
            elif discrimination == 'interquartile':
                predictions[method]['colour_discrimination'][db] = stats.iqr(x)
            else:
                predictions[method]['colour_discrimination'][db] = np.std(x) / (x.max() - x.min())
    # colour difference
    for method in predictions.keys():
        for db in predictions[method]['colour_difference'].keys():
            x = predictions[method]['colour_difference'][db]
            if difference == 'pearson':
                predictions[method]['colour_difference'][db] = x[0]
            elif difference == 'spearman':
                predictions[method]['colour_difference'][db] = x[1]
            else:
                predictions[method]['colour_difference'][db] = x[2]
    return predictions


def plot_predictions(predictions, ylabel_discrimination, ylabel_difference):
    fig = plt.figure(figsize=(20, 8))
    fontsize = 18

    datasets = list(predictions['RGB']['colour_discrimination'].keys())
    ax = fig.add_subplot(1, 2, 1, polar=True)
    df = pd.DataFrame({
        'Dataset': datasets,
        **{key: val['colour_discrimination'].values() for key, val in predictions.items()}
    })
    values = df.iloc[:, 1:].to_numpy()
    angles = np.linspace(0, 2 * np.pi, len(datasets), endpoint=False).tolist()
    values = np.concatenate([values, values[:1]], axis=0)
    angles += angles[:1]
    datasets += datasets[:1]

    ax.plot(angles, values, linewidth=3, label=list(predictions.keys()))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles), datasets, fontsize=fontsize)
    if ylabel_discrimination != 'STRESS':
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
    ax.set_title(ylabel_discrimination, fontsize=fontsize, fontweight='bold')
    ax.set_xlabel('Dataset', fontsize=fontsize)
    ax.set_ylabel('', fontsize=fontsize)
    ax.legend(fontsize=13, loc='upper right', bbox_to_anchor=(1.2, 1.2))

    datasets = list(predictions['RGB']['colour_difference'].keys())
    ax = fig.add_subplot(1, 2, 2)
    df = pd.DataFrame({
        'Dataset': datasets,
        **{key: val['colour_difference'].values() for key, val in predictions.items()}
    })
    tidy = df.melt(id_vars='Dataset', var_name='Method').rename(columns=str.title)
    sns.barplot(x='Dataset', y='Value', hue='Method', data=tidy, ax=ax)
    ax.set_title('MacAdam 1974', fontsize=fontsize, fontweight='bold')
    ax.set_xlabel('Dataset', fontsize=fontsize)
    ax.set_ylabel(ylabel_difference, fontsize=fontsize)
    ax.legend(fontsize=13, ncol=4, loc='lower center')

    return fig


def clip_01(x):
    return np.maximum(np.minimum(x, 1), 0)


def euc_distance(a, b):
    return np.sum((a.astype('float32') - b.astype('float32')) ** 2, axis=-1) ** 0.5


def estimate_max_distance(method, nrands=10000, rgb_type='srgb'):
    if not isinstance(rgb_type, str):
        min_rgb, max_rgb = rgb_type.min(), rgb_type.max()
    else:
        min_rgb, max_rgb = (0, 1) if rgb_type == 'srgb' else (0, 8.125)
    rand_rgbs = np.random.uniform(min_rgb, max_rgb, (nrands, 3))
    if not isinstance(method, str):
        netspace = pred_model(method, rand_rgbs)
        pred = euc_distance(netspace[:nrands // 2], netspace[nrands // 2:])
    elif method == 'euc':
        pred = euc_distance(rand_rgbs[:nrands // 2], rand_rgbs[nrands // 2:])
    else:
        a_lab = colour_science.XYZ_to_Lab(colour_science.sRGB_to_XYZ(rand_rgbs[:nrands // 2]))
        b_lab = colour_science.XYZ_to_Lab(colour_science.sRGB_to_XYZ(rand_rgbs[:nrands // 2:]))
        pred = colour_science.delta_E(a_lab, b_lab, method=method)
    max_dis = np.quantile(pred, 0.9)
    return max_dis


def identity(x):
    return x


def read_test_pts(pts_path):
    test_file = np.loadtxt(pts_path, delimiter=',', dtype=str)
    test_pts = dict()
    for test_pt in test_file:
        pt_val = test_pt[:3].astype('float')
        test_pt_name = test_pt[-2]
        if 'ref_' == test_pt_name[:4]:
            test_pt_name = test_pt_name[4:]
            if test_pt[-1] == 'rgb':
                ffun = identity
                bfun = identity
                chns_name = ['X', 'Y', 'Y']
            else:
                sys.exit('Unsupported colour space %s' % test_pt[-1])
            test_pts[test_pt_name] = {
                'ref': pt_val, 'space': chns_name, 'ext': [],
                'ffun': ffun, 'bfun': bfun
            }
        else:
            test_pts[test_pt_name]['ext'].append(pt_val)
    return test_pts


def read_network_results(res_dir, arch, test_data, exclude_list=None):
    if exclude_list is None:
        exclude_list = []
    net_result = dict()
    for area in arch_areas[arch]:
        area_result = dict()
        for ps in test_data.keys():
            if ps in exclude_list:
                continue
            area_result[ps] = []
            for pind in range(len(test_data[ps]['ext'])):
                res_path = '%s/%s/evolution_%s_%d.csv' % (res_dir, area, ps, pind)
                if not os.path.exists(res_path):
                    continue
                current_result = np.loadtxt(res_path, delimiter=',')
                sens_th = current_result if len(current_result.shape) == 1 else current_result[-1]
                area_result[ps].append(sens_th)
            area_result[ps] = np.array(area_result[ps])
        net_result[area] = area_result
    return net_result


def centre_threshold_arrays(test_data, area_res):
    centre_pts = []
    border_pts = []
    for focal_name in area_res.keys():
        test_pts = test_data[focal_name]
        org_cen = test_pts['ref']
        org_pts = np.expand_dims(org_cen, axis=(0, 1))
        rgb_pts = test_pts['ffun'](org_pts.astype('float32'))
        centre_pts.append(rgb_pts.squeeze())

        sen_res = area_res[focal_name]
        sense_pts = np.array(sen_res)[:, 1:4]
        org_pts = np.expand_dims(sense_pts, axis=1)
        rgb_pts = test_pts['ffun'](org_pts.astype('float32'))
        bor_rgb = rgb_pts.squeeze()
        border_pts.append(bor_rgb)
    return np.array(centre_pts, dtype=object), np.array(border_pts, dtype=object)


def onehot_centre_threshold_arrays(centre_pts, border_pts):
    onehot_centre = []
    onehot_border = []
    for bind, all_borders in enumerate(border_pts):
        for border in all_borders:
            onehot_centre.append(centre_pts[bind])
            onehot_border.append(border)
    return np.array(onehot_centre).astype('float32'), np.array(onehot_border).astype('float32')


def parse_network_results(net_res_dir, arch, test_data, exclude_list=None):
    if exclude_list is None:
        exclude_list = []
    network_thresholds = read_network_results(net_res_dir, arch, test_data, exclude_list)
    network_result_summary = dict()
    for area_name, area_val in network_thresholds.items():
        centre_pts, border_pts = centre_threshold_arrays(test_data, area_val)
        onehot_cen, onehot_bor = onehot_centre_threshold_arrays(centre_pts, border_pts)
        network_result_summary[area_name] = {
            'cat_cen': centre_pts, 'cat_bor': border_pts,
            'hot_cen': onehot_cen, 'hot_bor': onehot_bor
        }
    return network_result_summary


def plot_colour_pts(points, colours, title=None, axis_names=None, whichd='all',
                    projections=None, axs_range=None, fontsize=10):
    if whichd == '2d':
        naxis = 3
    elif whichd == '3d':
        naxis = 1
    else:
        naxis = 4
    fig = plt.figure(figsize=(naxis * 5 + 3, 5))

    axis_names = ['Ax=0', 'Ax=1', 'Ax=2'] if axis_names is None else axis_names
    if axs_range == 'auto':
        min_pts = points.min(axis=(1, 0))
        max_pts = points.max(axis=(1, 0))
        axs_len = max_pts - min_pts
        axs_range = list(zip(-0.05 * abs(axs_len) + min_pts, 0.05 * abs(axs_len) + max_pts))
    if whichd != '2d':
        ax_3d = fig.add_subplot(1, naxis, 1, projection='3d')
        scatter3d(points, colours, ax_3d, axis_names, fontsize, axs_range)
    if whichd != '3d':
        if projections is None:
            projections = [None] * 3
        axs_2d = [fig.add_subplot(
            1, naxis, chn, projection=projections[chn - 2]
        ) for chn in range(naxis - 2, naxis + 1)]
        axs_2d = scatter2d(points, colours, axs_2d, axis_names, fontsize, axs_range)
    if title is not None:
        fig.suptitle(title, fontsize=int(fontsize * 1.5))
    return fig


def scatter3d(points, colours, ax, axis_names, fontsize=14, axs_range=None):
    """Plotting the points in a 3D space."""
    s_size = 8 ** 2

    if axs_range is None:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    else:
        ax.set_xlim(*axs_range[0])
        ax.set_ylim(*axs_range[1])
        ax.set_zlim(*axs_range[2])

    if axis_names[0] == 'H':
        org_points = points.copy()
        points = points.copy()
        points[..., 0] = org_points[..., 1] * np.cos(org_points[..., 0])
        points[..., 1] = org_points[..., 1] * np.sin(org_points[..., 0])

    ax.scatter(points[..., 0], points[..., 1], points[..., 2],
               c=colours, marker='o', edgecolors='gray', s=s_size)
    ax.set_xlabel(axis_names[0], fontsize=fontsize, rotation=-15, labelpad=0)
    ax.set_ylabel(axis_names[1], fontsize=fontsize, rotation=45, labelpad=0)
    ax.set_zlabel(axis_names[2], fontsize=fontsize, rotation=90, labelpad=0)
    return ax


def scatter2d(points, colours, axs, axis_names, fontsize=14, axs_range=None):
    """Plotting three planes of a 3D space."""
    s_size = 10 ** 2

    p1s = [0, 0, 1]
    p2s = [1, 2, 2]
    for ax_ind, ax in enumerate(axs):
        if ax is None:
            continue
        ax.scatter(points[..., p1s[ax_ind]], points[..., p2s[ax_ind]], s=s_size,
                   marker='o', color=colours, edgecolors='gray')
        ax.set_xlabel(axis_names[p1s[ax_ind]], fontsize=fontsize, loc='left')
        ax.set_ylabel(axis_names[p2s[ax_ind]], fontsize=fontsize, loc='bottom')

        if axs_range is None:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.set_xlim(*axs_range[p1s[ax_ind]])
            ax.set_ylim(*axs_range[p2s[ax_ind]])
    return axs


def sample_rgb(cube_samples=1000):
    num_samples = round(cube_samples ** (1 / 3))
    linspace_vals = np.linspace(0, 1, num_samples)
    r_pts = np.tile(linspace_vals, (num_samples ** 2, 1)).T.reshape(-1, 1)
    g_pts = np.tile(linspace_vals, (num_samples, num_samples)).T.reshape(-1, 1)
    b_pts = np.tile(linspace_vals, (1, num_samples ** 2)).T.reshape(-1, 1)
    return np.stack((r_pts, g_pts, b_pts), axis=2)


def main(argv):
    parser = argparse.ArgumentParser(description='Optimising Colour Spaces!')
    parser.add_argument('--in_dir', required=True, type=str)
    parser.add_argument('--test_file', required=True, type=str)
    parser.add_argument('--human_data_dir', required=True, type=str)
    parser.add_argument('--out_dir', default='outputs', type=str)
    parser.add_argument('--epochs', default='5000', type=int)

    args = parser.parse_args(argv)
    _main_worker(args)


def _main_worker(args):
    rgb_test_data = read_test_pts(args.test_file)
    arch = None
    for key in arch_areas.keys():
        if key in args.in_dir:
            arch = key
            break
    if arch is None:
        sys.exit('Unsupported architecture %s' % arch)
    pretrained_db = 'clip' if 'clip' in arch else 'ImageNet'

    network_result_summary = parse_network_results(args.in_dir, arch, rgb_test_data)
    for layer in ['block7']:  # fixme arch_areas[arch]
        optimise_layer(args, network_result_summary, (pretrained_db, arch), layer)


def train_test_splits(layer_results, test_perc=0.1):
    num_centres = layer_results['cat_cen'].shape[0]
    num_tests = int(num_centres * test_perc)
    data_inds = np.arange(num_centres)
    np.random.shuffle(data_inds)
    train_inds = data_inds[:num_centres - num_tests]
    test_inds = data_inds[num_centres - num_tests:]
    train_split = {
        'cat_cen': layer_results['cat_cen'][train_inds],
        'cat_bor': layer_results['cat_bor'][train_inds]
    }
    test_split = {
        'cat_cen': layer_results['cat_cen'][test_inds],
        'cat_bor': layer_results['cat_bor'][test_inds]
    }
    return train_split, test_split


def train_val_sets(layer_results, val_perc=0.1):
    num_centres = layer_results['cat_cen'].shape[0]
    num_vals = int(num_centres * val_perc)
    data_inds = np.arange(num_centres)
    np.random.shuffle(data_inds)
    val_inds = data_inds[num_centres - num_vals:]

    centre_data = layer_results['cat_cen'].copy()
    border_data = layer_results['cat_bor'].copy()
    train_pts, train_map_inds = [], []
    val_pts, val_map_inds = [], []
    for centre_ind, centre_pt in enumerate(centre_data):
        all_pts = val_pts if centre_ind in val_inds else train_pts
        map_inds = val_map_inds if centre_ind in val_inds else train_map_inds
        all_pts.append(centre_pt)
        cen_in_ind = len(all_pts) - 1
        for border_pt in border_data[centre_ind]:
            all_pts.append(border_pt)
            bor_in_ind = len(all_pts) - 1
            map_inds.append([cen_in_ind, bor_in_ind])
    train_pts = np.array(train_pts, dtype='float32')
    train_map_inds = np.array(train_map_inds)
    val_pts = np.array(val_pts, dtype='float32')
    val_map_inds = np.array(val_map_inds)
    return (train_pts, train_map_inds), (val_pts, val_map_inds)


class ColourSpaceNet(nn.Module):
    def __init__(self, units=None, nonlinearities='GELU', mean_std=None):
        super().__init__()
        self.mean_std = (0, 1) if mean_std is None else mean_std
        if units is None:
            units = [7, 15, 7]
        num_units = [int(unit) for unit in units]
        in_units = [3, *num_units]
        out_units = [*num_units, 3]
        if type(nonlinearities) is not list:
            nonlinearities = [nonlinearities] * (len(num_units) + 1)
        nonlinear_units = [non_linear_funs[nonlinearity] for nonlinearity in nonlinearities]

        layers = []
        for i in range(len(num_units) + 1):
            layers.append(nn.Linear(in_units[i], out_units[i]))
            layers.append(nonlinear_units[i])
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


def pred_model(model, rgbs):
    model = model.eval()
    with torch.set_grad_enabled(False):
        input_space = torch.tensor(rgbs.copy()).float()
        out_space = model(input_space)
    return out_space.numpy()


non_linear_funs = {
    'GELU': nn.GELU(),
    'ReLU': nn.ReLU(),
    'SELU': nn.SELU(),
    'SiLU': nn.SiLU(),
    'Tanh': nn.Tanh(),
    'Sigmoid': nn.Sigmoid(),
    'identity': nn.Identity()
}

optimisers = {
    'Adam': torch.optim.Adam,
    'Adamax': torch.optim.Adamax,
    'Adagrad': torch.optim.Adagrad,
    'SGD': torch.optim.SGD,
}


def optimise_layer(args, network_result_summary, pretrained, layer):
    pretrained_db, pretrained_arch = pretrained
    layer_out_dir = '%s/%s/%s/%s/' % (args.out_dir, pretrained_db, pretrained_arch, layer)

    layer_results = network_result_summary[layer]

    intermediate_nonlinears = ['GELU', 'ReLU', 'SELU', 'SiLU', 'Tanh']
    args.loss = 'nada!'
    losses = ['range', 'mean_distance']
    for opt_method in ['Adamax', 'Adam']:
        for i in range(10):
            num_units = np.random.randint(7, 15, size=np.random.randint(2, 5)).tolist()
            for non_lin_ind in range(5):
                nonlinearity = [
                    *list(np.random.choice(intermediate_nonlinears, len(num_units))),
                    np.random.choice(['Tanh', 'Sigmoid', 'identity'], 1)[0]
                ]
                exname = '%s_%.2d_%s' % (
                    opt_method, non_lin_ind, '_'.join(str(i) for i in num_units))
                for instance in range(3):
                    out_dir = '%s/%s/i%.3d/' % (layer_out_dir, exname, instance)
                    os.makedirs(out_dir, exist_ok=True)

                    orig_stdout = sys.stdout
                    f = open('%s/log.txt' % out_dir, 'w')
                    sys.stdout = f

                    args.num_units = num_units
                    args.nonlinearities = nonlinearity
                    args.opt_method = opt_method
                    args.lr = np.random.choice([0.1, 0.01])
                    json_file_name = os.path.join(out_dir, 'args.json')
                    with open(json_file_name, 'w') as fp:
                        json.dump(dict(args._get_kwargs()), fp, sort_keys=True, indent=4)
                    optimise_instance(args, layer_results, out_dir)

                    sys.stdout = orig_stdout
                    f.close()


def optimise_instance(args, layer_results, out_dir):
    mean_std = (0.5, 0.5)
    # model
    model = ColourSpaceNet(args.num_units, args.nonlinearities, mean_std)
    print(model)

    # optimisation
    optimiser = optimisers[args.opt_method](params=model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=args.epochs // 3)

    # epoch loop
    print_freq = args.epochs // 10
    losses = []
    train_db, _ = train_val_sets(layer_results, 0)
    for epoch in range(args.epochs):
        model = model.train()
        # train_db, _ = train_val_sets(layer_results, 0.1)
        with torch.set_grad_enabled(True):
            input_space = torch.tensor(train_db[0].copy()).float()
            out_space = model(input_space)
            euc_dis = torch.sum(
                (out_space[train_db[1][:, 0]] - out_space[train_db[1][:, 1]]) ** 2, axis=-1
            ) ** 0.5
            min_vals, _ = out_space.min(axis=0)
            max_vals, _ = out_space.max(axis=0)
            deltad = max_vals - min_vals
            uniformity_loss = torch.std(euc_dis)  # / torch.mean(euc_dis)
            # uniformity_loss = stress_torch(euc_dis)
            if args.loss == 'range':
                range_loss = 0.5 * (abs(1 - deltad[0]) + abs(1 - deltad[1]) + abs(1 - deltad[2]))
            elif args.loss == 'mean_distance':
                range_loss = 0.5 * abs(0.1 - torch.mean(euc_dis))
            else:
                range_loss = 0
            loss = uniformity_loss + range_loss

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            scheduler.step()

        if torch.isnan(loss):
            print('NaN!', epoch)
            return

        human_cv = predict_human_data({'Network': model}, args.human_data_dir, 'cv', 'pearson')
        human_stress = predict_human_data({'Network': model}, args.human_data_dir, 'stress',
                                          'pearson')
        cv_str = '[ '
        for key, val in human_cv['Network']['colour_discrimination'].items():
            if key in ['Huang2012', 'TeamK']:
                cv_str += ('%s: %.2f ' % (key[:2], val))
        cv_str += ']'
        stress_str = '[ '
        for key, val in human_stress['Network']['colour_discrimination'].items():
            if key in ['Huang2012', 'TeamK']:
                stress_str += ('%s: %.2f ' % (key[:2], val))
        stress_str += ']'
        if np.mod(epoch, print_freq) == 0 or epoch == (args.epochs - 1):
            print(
                '[%.5d] loss=%.4f [%.2f %.2f %.2f] CV=%s STRESS=%s CORR=%.2f' % (
                    epoch, uniformity_loss, *deltad, cv_str, stress_str,
                    human_cv['Network']['colour_difference']['MacAdam1974']
                )
            )
        header = 'loss,'
        epoch_loss = [uniformity_loss.item()]
        for dbname, db_res in human_cv['Network']['colour_discrimination'].items():
            epoch_loss.append(db_res)
            header += 'CV_%s,' % dbname
        for dbname, db_res in human_stress['Network']['colour_discrimination'].items():
            epoch_loss.append(db_res)
            header += 'STRESS_%s,' % dbname
        header += 'CORR_MacAdam1974'
        epoch_loss.append(human_cv['Network']['colour_difference']['MacAdam1974'])
        losses.append(epoch_loss)

    rgb_pts = sample_rgb()
    rgb_squeezed = rgb_pts.copy().squeeze()
    rgb_pts_pred = pred_model(model, rgb_squeezed)
    rgb_pts_pred = np.expand_dims(rgb_pts_pred, axis=1)
    space_range = list(rgb_pts_pred.max(axis=(0, 1)) - rgb_pts_pred.min(axis=(0, 1)))
    print('Network-space range:\t%s (%.3f, %.3f %.3f)' % ('', *space_range))
    fig = plot_colour_pts(
        rgb_pts_pred, rgb_pts,
        'loss=%.4f   CV=%s   STRESS=%s   r=%.2f' % (
            uniformity_loss, cv_str, stress_str,
            human_cv['Network']['colour_difference']['MacAdam1974']
        ),
        axs_range='auto'
    )

    fig.savefig('%s/rgb_pred.svg' % out_dir)
    plt.close('all')
    np.savetxt('%s/losses.txt' % out_dir, losses, delimiter=',', header=header)

    torch.save({
        'state_dict': model.state_dict(),
        'units': args.num_units,
        'nonlinearities': args.nonlinearities,
        'mean_std': mean_std
    }, '%s/model.pth' % out_dir)


if __name__ == '__main__':
    main(sys.argv[1:])
