"""
Optimisation of uniform colour space
"""

import numpy as np
import argparse
import os
import sys
import json

from matplotlib import pyplot as plt

import torch
import torch.nn as nn

arch_areas = {
    'clip_RN50': [*['area%d' % i for i in range(0, 5)], 'encoder'],
    'clip_B32': [*['block%d' % i for i in [1, 4, 7, 10, 11]], 'encoder'],
    'resnet50': [*['area%d' % i for i in range(0, 5)], 'fc'],
    'vit_b_32': [*['block%d' % i for i in [1, 4, 7, 10, 11]], 'fc'],
    'taskonomy': [*['area%d' % i for i in range(0, 5)], 'encoder']
}


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
        org_pts = np.expand_dims(sense_pts, axis=(1))
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
    parser.add_argument('--out_dir', default='outputs', type=str)
    parser.add_argument('--epochs', default='50000', type=int)

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
    for layer in arch_areas[arch]:
        optimise_layer(args, network_result_summary, (pretrained_db, arch), layer)


def get_db(network_result_summary, layer):
    centre_data = network_result_summary[layer]['cat_cen'].copy()
    border_data = network_result_summary[layer]['cat_bor'].copy()
    all_pts = []
    centre_border_inds = []
    for centre_ind, centre_pt in enumerate(centre_data):
        all_pts.append(centre_pt)
        cen_in_ind = len(all_pts) - 1
        for border_pt in border_data[centre_ind]:
            all_pts.append(border_pt)
            bor_in_ind = len(all_pts) - 1
            centre_border_inds.append([cen_in_ind, bor_in_ind])
    all_pts = np.array(all_pts, dtype='float32')
    centre_border_inds = np.array(centre_border_inds)
    db = (all_pts, centre_border_inds)
    return db


class ColourSpaceNet(nn.Module):
    def __init__(self, num_units=None, nonlin_fun=nn.ReLU()):
        super().__init__()
        if num_units is None:
            num_units = [5]
        self.linear_0 = nn.Linear(in_features=3, out_features=num_units[0])
        self.nonlin_0 = nonlin_fun
        intermediates = []
        for i in range(1, len(num_units)):
            intermediates.append(nn.Linear(in_features=num_units[i - 1], out_features=num_units[i]))
            intermediates.append(nonlin_fun)
        self.other_layers = nn.Sequential(*intermediates)
        self.linear_n = nn.Linear(in_features=num_units[-1], out_features=3)

    def forward(self, x):
        x = self.linear_0(x)
        x = self.nonlin_0(x)
        x = self.other_layers(x)
        x = self.linear_n(x)
        x = torch.tanh(x)
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
    'SELU': nn.ReLU(),
    'SiLU': nn.ReLU(),
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

    db = get_db(network_result_summary, layer)

    for num_units in [[25], [49], [5, 5], [7, 7]]:
        for nonlinearity in non_linear_funs.keys():
            for opt_method in optimisers:
                exname = '%s_%s_%s' % (opt_method, nonlinearity, '_'.join(str(i) for i in num_units))
                for instance in range(3):
                    out_dir = '%s/%s/i%.3d/' % (layer_out_dir, exname, instance)
                    os.makedirs(out_dir, exist_ok=True)

                    orig_stdout = sys.stdout
                    f = open('%s/log.txt' % out_dir, 'w')
                    sys.stdout = f

                    args.num_units = num_units
                    args.nonlin_fun = nonlinearity
                    args.opt_method = opt_method
                    args.lr = 0.1
                    json_file_name = os.path.join(out_dir, 'args.json')
                    with open(json_file_name, 'w') as fp:
                        json.dump(dict(args._get_kwargs()), fp, sort_keys=True, indent=4)
                    optimise_instance(args, db, out_dir)

                    sys.stdout = orig_stdout
                    f.close()


def optimise_instance(args, db, out_dir):
    # model
    model = ColourSpaceNet(args.num_units, non_linear_funs[args.nonlin_fun])
    model = model.train()
    print(model)

    # optimisation
    optimiser = optimisers[args.opt_method](params=model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=args.epochs // 3)

    # epoch loop
    euc_dis = np.sum((db[0][db[1][:, 0]] - db[0][db[1][:, 1]]) ** 2, axis=-1) ** 0.5
    print('RGB error: %.4f' % np.std(euc_dis))
    print_freq = args.epochs // 10
    losses = []
    for epoch in range(args.epochs):
        with torch.set_grad_enabled(True):
            input_space = torch.tensor(db[0].copy()).float()
            out_space = model(input_space)
            euc_dis = torch.sum((out_space[db[1][:, 0]] - out_space[db[1][:, 1]]) ** 2, axis=-1) ** 0.5
            loss = torch.std(euc_dis)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            scheduler.step()

        if np.mod(epoch, print_freq) == 0:
            print('[%.2d] loss=%.4f' % (epoch, loss.item()))
        if torch.isnan(loss):
            print('NaN!', epoch)
            return
        losses.append(loss.item())

    rgb_pts = sample_rgb()
    rgb_squeezed = rgb_pts.copy().squeeze()
    area_space_no = pred_model(model, rgb_squeezed)
    area_space_no = np.expand_dims(area_space_no, axis=1)
    print('Range:\t', area_space_no.min(axis=(0, 1)), area_space_no.max(axis=(0, 1)))
    fig = plot_colour_pts(area_space_no, rgb_pts, 'loss=%.4f' % losses[-1], axs_range='auto')

    fig.savefig('%s/rgb_pred.svg' % out_dir)
    plt.close('all')
    np.savetxt('%s/losses.txt' % out_dir, losses)

    torch.save({
        'state_dict': model.state_dict(),
        'num_units': args.num_units,
        'nonlin_fun': args.nonlin_fun
    }, '%s/model.pth' % out_dir)


if __name__ == '__main__':
    main(sys.argv[1:])
