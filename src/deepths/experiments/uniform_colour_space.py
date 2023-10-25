"""
Optimisation of uniform colour space
"""

import numpy as np
import argparse
import os
import sys
import json

from matplotlib import pyplot as plt
from scipy import stats
from skimage import color as skicolour
import colour as colour_science

import torch
import torch.nn as nn

arch_areas = {
    'clip_RN50': [*['area%d' % i for i in range(0, 5)], 'encoder'],
    'clip_B32': [*['block%d' % i for i in [1, 4, 7, 10, 11]], 'encoder'],
    'resnet50': [*['area%d' % i for i in range(0, 5)], 'fc'],
    'vit_b_32': [*['block%d' % i for i in [1, 4, 7, 10, 11]], 'fc'],
    'taskonomy': [*['area%d' % i for i in range(0, 5)], 'encoder']
}


def clip_01(x):
    return np.maximum(np.minimum(x, 1), 0)


def load_human_data(path):
    human_data = read_test_pts(path)
    human_data_ref_pts = np.expand_dims(np.array([val['ref'] for val in human_data.values()]), axis=1)
    human_hot_cen, human_hot_bor = [], []
    for key, val in human_data.items():
        for pt in val['ext']:
            human_hot_cen.append(val['ref'])
            human_hot_bor.append(pt)
    human_hot_cen = np.array(human_hot_cen)
    human_hot_bor = np.array(human_hot_bor)
    return {'data': human_data, 'ref_pts': human_data_ref_pts,
            'hot_cen': human_hot_cen, 'hot_bor': human_hot_bor}


def prophoto_rgb_colour_diff(a, b, diff_fun='de2000'):
    illuminant = np.array([0.31271, 0.32902])
    a_lab = colour_science.XYZ_to_Lab(
        colour_science.RGB_to_XYZ(a, 'ProPhoto RGB', illuminant, chromatic_adaptation_transform=None),
        illuminant
    )
    b_lab = colour_science.XYZ_to_Lab(
        colour_science.RGB_to_XYZ(b, 'ProPhoto RGB', illuminant, chromatic_adaptation_transform=None),
        illuminant
    )
    return colour_diff_lab(a_lab, b_lab, diff_fun)


def pred_human_data(path, model, model_max=1, de_max=1, print_val=None):
    human_data = load_human_data(path)
    de2000 = prophoto_rgb_colour_diff(human_data['hot_cen'], human_data['hot_bor'], diff_fun='de2000')
    cen_pred = pred_model(model, clip_01(human_data['hot_cen']))
    bor_pred = pred_model(model, clip_01(human_data['hot_bor']))
    pred_euc = euc_distance(cen_pred, bor_pred)
    if print_val is not None:
        print('%sDE-2000 %.4f [%.4f] CV [%.4f]' % (
            print_val, np.std(de2000), np.std(de2000 / de_max), np.std(de2000) / np.mean(de2000)))
        print('%sNetwork %.4f [%.4f] CV [%.4f]' % (
            print_val, np.std(pred_euc), np.std(pred_euc / model_max), np.std(pred_euc) / np.mean(pred_euc)))
    return {
        'de2000': [np.std(de2000), np.std(de2000 / de_max)],
        'model': [np.std(pred_euc), np.std(pred_euc / model_max)]
    }


def pred_macadam1972(path, model, print_val='\t'):
    macadam1972_data = np.loadtxt(path, delimiter=',')
    tile1 = pred_model(model, clip_01(macadam1972_data[:, 1:4]))
    tile2 = pred_model(model, clip_01(macadam1972_data[:, 4:7]))
    pred_euc = euc_distance(tile1, tile2)
    pred_euc[np.isnan(pred_euc)] = 0
    pcorr, _ = stats.pearsonr(pred_euc, macadam1972_data[:, 0])
    scorr, _ = stats.spearmanr(pred_euc, macadam1972_data[:, 0])
    de2000 = colour_diff_lab(macadam1972_data[:, 7:10], macadam1972_data[:, 10:13])
    pcorr_de, _ = stats.pearsonr(de2000, macadam1972_data[:, 0])
    scorr_de, _ = stats.spearmanr(de2000, macadam1972_data[:, 0])
    if print_val is not None:
        print('%sDE-2000 Pearson %.2f \t Spearman %.2f' % (print_val, pcorr_de, scorr_de))
        print('%sNetwork Pearson %.2f \t Spearman %.2f' % (print_val, pcorr, scorr))
    return {
        'de2000': [pcorr_de, scorr_de],
        'model': [pcorr, scorr]
    }


def test_human_data(model, human_data_dir, do_print=True):
    print_val = '\t' if do_print else None
    model_max, de_max = estimate_max_distance(model, 10000)
    if do_print:
        print('* MacAdam 1942')
    macadam_res = pred_human_data(
        human_data_dir + '/macadam_rgb_org.csv', model, model_max=model_max, de_max=de_max,
        print_val=print_val
    )
    if do_print:
        print('* Luo-Rigg 1986')
    luorigg_res = pred_human_data(
        human_data_dir + 'luorigg_rgb_org.csv', model, model_max=model_max, de_max=de_max,
        print_val=print_val
    )
    if do_print:
        print('* MacAdam 1972')
    macadam1972_res = pred_macadam1972(
        human_data_dir + '/macadam1972.csv', model, print_val=print_val
    )
    return {
        'MacAdam': macadam_res,
        'Luo-Rigg': luorigg_res,
        'MacAdam1972': macadam1972_res,
    }


def euc_distance(a, b):
    return np.sum((a.astype('float32') - b.astype('float32')) ** 2, axis=-1) ** 0.5


def colour_diff_lab(a_lab, b_lab, diff_fun='de2000'):
    if diff_fun == 'de2000':
        diff_fun = skicolour.deltaE_ciede2000
    elif diff_fun == 'de1994':
        diff_fun = skicolour.deltaE_ciede94
    else:
        diff_fun = skicolour.deltaE_cie76
    return diff_fun(a_lab, b_lab)


def colour_diff(a, b, diff_fun='euc'):
    a = a.copy().astype('float32')
    b = b.copy().astype('float32')
    if diff_fun == 'euc':
        return euc_distance(a, b)
    else:
        return colour_diff_lab(skicolour.rgb2lab(a), skicolour.rgb2lab(b), diff_fun)


def estimate_max_distance(model, nrands=10000):
    rand_rgbs = np.random.uniform(0, 1, (nrands, 3))
    rand_rgbs_pred = pred_model(model, rand_rgbs)
    model_max = np.mean(euc_distance(rand_rgbs_pred[:nrands // 2], rand_rgbs_pred[nrands // 2:]))
    de_max = np.mean(colour_diff(rand_rgbs[:nrands // 2], rand_rgbs[nrands // 2:], diff_fun='de2000'))
    return model_max, de_max


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
                    projections=None, axs_range=None):
    if whichd == '2d':
        naxis = 3
    elif whichd == '3d':
        naxis = 1
    else:
        naxis = 4
    fig = plt.figure(figsize=(naxis * 5 + 3, 5))

    fontsize = 18
    axis_names = ['Ax=0', 'Ax=1', 'Ax=2'] if axis_names is None else axis_names
    if axs_range == 'auto':
        min_pts = points.min(axis=(1, 0))
        max_pts = points.max(axis=(1, 0))
        axs_len = max_pts - min_pts
        axs_range = list(zip(-0.05 * abs(axs_len) + min_pts, 0.05 * abs(axs_len) + max_pts))
    if whichd != '2d':
        ax_3d = fig.add_subplot(1, naxis, 1, projection='3d')
        ax_3d = scatter_3D(points, colours, ax_3d, axis_names, fontsize, axs_range)
    if whichd != '3d':
        if projections is None:
            projections = [None] * 3
        axs_2d = [fig.add_subplot(
            1, naxis, chn, projection=projections[chn - 2]
        ) for chn in range(naxis - 2, naxis + 1)]
        axs_2d = scatter_2D(points, colours, axs_2d, axis_names, fontsize, axs_range)
    if title is not None:
        fig.suptitle(title, fontsize=int(fontsize * 1.5))
    return fig


def scatter_3D(points, colours, ax, axis_names, fontsize=14, axs_range=None):
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


def scatter_2D(points, colours, axs, axis_names, fontsize=14, axs_range=None):
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
    for layer in ['block4', 'block7', 'block10']:  # fixme arch_areas[arch]
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

    for loss in ['range', 'mean_distance']:
        args.loss = loss
        for i in range(10):
            num_units = np.random.randint(7, 15, size=np.random.randint(2, 5)).tolist()
            for non_lin_ind in range(10):
                nonlinearity = [
                    *list(np.random.choice(['GELU', 'ReLU', 'SELU', 'SiLU', 'Tanh'], len(num_units))),
                    np.random.choice(['Tanh', 'Sigmoid', 'identity'], 1)[0]
                ]
                opt_method = 'Adamax'
                exname = '%s_%.2d_%s' % (opt_method, non_lin_ind, '_'.join(str(i) for i in num_units))
                for instance in range(3):
                    out_dir = '%s/%s/i%.3d/' % (layer_out_dir, exname, instance)
                    os.makedirs(out_dir, exist_ok=True)

                    orig_stdout = sys.stdout
                    f = open('%s/log.txt' % out_dir, 'w')
                    sys.stdout = f

                    args.num_units = num_units
                    args.nonlinearities = nonlinearity
                    args.opt_method = opt_method
                    args.lr = 0.1
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
    for epoch in range(args.epochs):
        model = model.train()
        train_db, _ = train_val_sets(layer_results, 0.1)
        with torch.set_grad_enabled(True):
            input_space = torch.tensor(train_db[0].copy()).float()
            out_space = model(input_space)
            euc_dis = torch.sum((out_space[train_db[1][:, 0]] - out_space[train_db[1][:, 1]]) ** 2, axis=-1) ** 0.5
            min_vals, _ = out_space.min(axis=0)
            max_vals, _ = out_space.max(axis=0)
            range_dis = max_vals - min_vals
            uniformity_euc_dis = torch.std(euc_dis)
            if args.loss == 'range':
                loss = uniformity_euc_dis + 0.5 * (
                        abs(1 - range_dis[0]) + abs(1 - range_dis[1]) + abs(1 - range_dis[2])
                )
            elif args.loss == 'mean_distance':
                loss = uniformity_euc_dis + 0.5 * abs(0.1 - torch.mean(euc_dis))

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            scheduler.step()

        if torch.isnan(loss):
            print('NaN!', epoch)
            return

        if np.mod(epoch, print_freq) == 0:
            human_tests = test_human_data(model, args.human_data_dir, False)
            print(
                '[%.5d] loss=%.4f [%.2f %.2f %.2f] MacAdam=[%.4f|%.4f]vs[%.4f] Luo-Rigg=[%.4f|%.4f]vs[%.4f] r=[%.2f]vs[%.2f]' % (
                    epoch, uniformity_euc_dis, *range_dis,
                    human_tests['MacAdam']['model'][0], human_tests['MacAdam']['model'][1],
                    human_tests['MacAdam']['de2000'][1],
                    human_tests['Luo-Rigg']['model'][0], human_tests['Luo-Rigg']['model'][1],
                    human_tests['Luo-Rigg']['de2000'][1],
                    human_tests['MacAdam1972']['model'][0], human_tests['MacAdam1972']['de2000'][0]
                )
            )
        losses.append([
            uniformity_euc_dis.item(),
            human_tests['MacAdam']['model'][0], human_tests['MacAdam']['model'][1],
            human_tests['Luo-Rigg']['model'][0], human_tests['Luo-Rigg']['model'][1],
            human_tests['MacAdam1972']['model'][0]
        ])

    rgb_pts = sample_rgb()
    rgb_squeezed = rgb_pts.copy().squeeze()
    rgb_pts_pred = pred_model(model, rgb_squeezed)
    rgb_pts_pred = np.expand_dims(rgb_pts_pred, axis=1)
    print('Range:\t', rgb_pts_pred.min(axis=(0, 1)), rgb_pts_pred.max(axis=(0, 1)))
    fig = plot_colour_pts(
        rgb_pts_pred, rgb_pts,
        'loss=%.4f   MacAdam=%.4f|%.4f   Luo-Rigg=%.4f|%.4f   r=%.2f' % (
            losses[-1][0], losses[-1][1], losses[-1][2], losses[-1][3], losses[-1][4], losses[-1][5]
        ),
        axs_range='auto'
    )

    fig.savefig('%s/rgb_pred.svg' % out_dir)
    plt.close('all')
    header = 'loss,MacAdam_raw,MacAdam_norm,LuoRigg_raw,LuoRigg_norm,Corr'
    np.savetxt('%s/losses.txt' % out_dir, losses, delimiter=',', header=header)

    torch.save({
        'state_dict': model.state_dict(),
        'units': args.num_units,
        'nonlinearities': args.nonlinearities,
        'mean_std': mean_std
    }, '%s/model.pth' % out_dir)


if __name__ == '__main__':
    main(sys.argv[1:])
