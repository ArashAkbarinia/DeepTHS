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
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats
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


def method_out(method, data):
    ref_rgb = data.loc[:, ['Ref-R', 'Ref-G', 'Ref-B']].to_numpy().astype('float32')
    test_rgb = data.loc[:, ['Test-R', 'Test-G', 'Test-B']].to_numpy().astype('float32')
    ref_lab = data.loc[:, ['Ref-L', 'Ref-a', 'Ref-b']].to_numpy().astype('float32')
    test_lab = data.loc[:, ['Test-L', 'Test-a', 'Test-b']].to_numpy().astype('float32')
    checkpoint_path = '%s/model.pth' % method
    if not isinstance(method, str) or os.path.exists(checkpoint_path):
        network = load_model(checkpoint_path) if isinstance(method, str) else method
        ref_rgb = pred_model(network, ref_rgb)
        test_rgb = pred_model(network, test_rgb)
        pred = euc_distance(ref_rgb, test_rgb)
    elif 'euc' in method:
        space = method.split('_')[1]
        if space == 'rgb':
            ref_val, test_val = ref_rgb, test_rgb
        elif space == 'lab':
            ref_val, test_val = ref_lab, test_lab
        else:
            sys.exit(f"Colour space {space} is not supported.")
        pred = euc_distance(ref_val, test_val)
    else:
        pred = colour_science.delta_E(ref_lab, test_lab, method=method)
    return pred


def db_pred(path, method):
    human_data = pd.read_csv(path)
    pred = method_out(method, human_data)
    return pred


def compare_human_data(method, test_dir, rgb_type):
    rgbs = {
        'srgb': 'sRGB',
        'prophoto': 'prophoto',
        'AdobeRgb1998': 'Adobe RGB (1998)'
    }
    rgb = rgbs[rgb_type]
    # MacAdam 1942
    macadam1942_res = db_pred('%s/macadam1942_%s_d65.csv' % (test_dir, rgb_type), method)
    # RIT-DuPont
    ritdupont_res = db_pred('%s/rit-dupont_%s.csv' % (test_dir, rgb_type), method)
    # Karl
    karl_res = db_pred('%s/karl_%s.csv' % (test_dir, rgb_type), method)

    # MacAdam 1974
    macadam1974_res = db_pred('%s/macadam1974_%s_d65.csv' % (test_dir, rgb_type), method)
    # Witt
    witt_res = db_pred('%s/witt_%s.csv' % (test_dir, rgb_type), method)
    # Leeds
    leeds_res = db_pred('%s/leeds_%s.csv' % (test_dir, rgb_type), method)
    # BFD
    bfd_res = db_pred('%s/bfd_%s.csv' % (test_dir, rgb_type), method)
    # TeamK
    teamk_res = db_pred('%s/teamk_%s_m2j2.csv' % (test_dir, rgb_type), method)

    return {
        'colour_discrimination': {
            'MacAdam1942': macadam1942_res,
            'RIT-DuPont1991': ritdupont_res,
            'Gegenfurtner': karl_res,
        },
        'colour_difference': {
            'Leeds1997': leeds_res,
            'Witt1999': witt_res,
            'MacAdam1974': macadam1974_res,
            'BFD1986': bfd_res,
            'TeamK': teamk_res
        }
    }


def stress(de, dv=None):
    if dv is None:
        dv = np.ones(len(de))
    return colour_science.index_stress(de, dv)


def stress_torch(de, dv=None):
    if dv is None:
        dv = torch.ones(len(de))
    f1 = torch.sum(de ** 2) / torch.sum(de * dv)
    return torch.sqrt(torch.sum((de - f1 * dv) ** 2) / torch.sum(f1 ** 2 * dv ** 2))


def predict_human_data(methods, test_dir, rgb_type='srgb'):
    predictions = {
        key: compare_human_data(method, test_dir, rgb_type) for key, method in methods.items()
    }
    return predictions


def evaluate_discrimination(predictions, discrimination):
    eval_pred = dict()
    # colour discrimination
    for method in predictions.keys():
        eval_pred[method] = {'colour_discrimination': dict()}
        for db in predictions[method]['colour_discrimination'].keys():
            x = predictions[method]['colour_discrimination'][db]
            if discrimination == 'cv':  # coefficient of variation (CV)
                eval_pred[method]['colour_discrimination'][db] = np.std(x) / np.mean(x)
            elif discrimination == 'stress':
                eval_pred[method]['colour_discrimination'][db] = stress(x)
            elif discrimination == 'entropy':
                eval_pred[method]['colour_discrimination'][db] = stats.entropy(x)
            elif discrimination == 'mad':
                eval_pred[method]['colour_discrimination'][db] = stats.median_abs_deviation(x)
            elif discrimination == 'interquartile':
                eval_pred[method]['colour_discrimination'][db] = stats.iqr(x)
            elif discrimination == 'normalised-STD':
                eval_pred[method]['colour_discrimination'][db] = np.std(x / x.max())
            else:
                eval_pred[method]['colour_discrimination'][db] = np.std(x) / (x.max() - x.min())
    return eval_pred


def evaluate_difference(predictions, difference, gts):
    eval_pred = dict()
    # colour difference
    for method in predictions.keys():
        eval_pred[method] = {'colour_difference': dict()}
        for db in predictions[method]['colour_difference'].keys():
            x = predictions[method]['colour_difference'][db]
            if np.any(np.isnan(x)) or np.any(np.isinf(x)):
                eval_pred[method]['colour_difference'][db] = 0
            else:
                if difference == 'pearson':
                    pearsonr_corr, _ = stats.pearsonr(x, gts[db])
                    eval_pred[method]['colour_difference'][db] = pearsonr_corr
                elif difference == 'spearman':
                    spearmanr_corr, _ = stats.spearmanr(x, gts[db])
                    eval_pred[method]['colour_difference'][db] = spearmanr_corr
                else:
                    eval_pred[method]['colour_difference'][db] = stress(x, gts[db])
    return eval_pred


def data_vs_network(preds, test_dir, **kwargs):
    method = 'Network'
    preds_mega_ordered = np.concatenate([
        preds[method]['colour_difference']['BFD1986'],
        preds[method]['colour_difference']['Leeds1997'],
        preds[method]['colour_discrimination']['RIT-DuPont1991'],
        preds[method]['colour_difference']['Witt1999'],
        preds[method]['colour_discrimination']['MacAdam1942'],
        preds[method]['colour_difference']['MacAdam1974'],
        preds[method]['colour_discrimination']['Gegenfurtner'],
        preds[method]['colour_difference']['TeamK'],
    ])

    mega_db = pd.read_csv(f"{test_dir}/meta_dbs_srgb.csv")

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    metrics = plot_human_vs_method(preds_mega_ordered, mega_db, return_metrics=True,
                                   ax=ax, xlabel='Network Euclidean Distance', **kwargs)
    return fig, metrics


def plot_predictions(preds, discriminations, differences, test_dir, return_metrics=False, **kargs):
    for _cspace in preds.keys():
        preds[_cspace]['colour_discrimination']['All'] = np.array(
            [x for db in preds[_cspace]['colour_discrimination'].values() for x in db]
        )
        preds[_cspace]['colour_difference']['All'] = np.array(
            [x for db in preds[_cspace]['colour_difference'].values() for x in db]
        )

    mega_db = pd.read_csv(f"{test_dir}/meta_dbs_srgb.csv")
    gts = {
        'Leeds1997': mega_db.loc[mega_db['Dataset'] == 'LEEDS', 'DV'],
        'Witt1999': mega_db.loc[mega_db['Dataset'] == 'WITT', 'DV'],
        'MacAdam1974': mega_db.loc[mega_db['Dataset'] == 'MacAdam1974', 'DV'],
        'BFD1986': mega_db.loc[
            mega_db['Dataset'].isin(['BFD-P(D65)', 'BFD-P( C )', 'BFD-P(M)']), 'DV'],
        'TeamK': mega_db.loc[mega_db['Dataset'] == 'Laysa2024', 'DV'],
    }
    gts['All'] = np.concatenate([_gt for _gt in gts.values()])

    eval_discrimination = [evaluate_discrimination(preds, metric) for metric in discriminations]
    eval_differences = [evaluate_difference(preds, metric, gts) for metric in differences]
    fig = plt.figure(figsize=(20, 12), layout="constrained")
    gs = GridSpec(2, max(len(eval_discrimination), len(eval_differences)), figure=fig)
    fontsize = 18

    for dis_ind, dis_res in enumerate(eval_discrimination):
        datasets = list(dis_res['RGB']['colour_discrimination'].keys())
        ax = fig.add_subplot(gs[0, dis_ind], polar=True)
        df = pd.DataFrame({
            'Dataset': datasets,
            **{key: val['colour_discrimination'].values() for key, val in dis_res.items()}
        })
        values = df.iloc[:, 1:].to_numpy()
        angles = np.linspace(0, 2 * np.pi, len(datasets), endpoint=False).tolist()
        values = np.concatenate([values, values[:1]], axis=0)
        angles += angles[:1]
        datasets += datasets[:1]

        ax.plot(angles, values, linewidth=3, label=list(dis_res.keys()))
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles), datasets, fontsize=fontsize)
        # ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
        ax.set_title(discriminations[dis_ind], fontsize=fontsize, fontweight='bold')
        ax.set_xlabel('', fontsize=fontsize)
        ax.set_ylabel('', fontsize=fontsize)
        if dis_ind == 0:
            ax.legend(fontsize=15, loc='upper right', bbox_to_anchor=(1.2, 1.2))

    for diff_ind, diff_res in enumerate(eval_differences):
        datasets = list(diff_res['RGB']['colour_difference'].keys())
        ax = fig.add_subplot(gs[1, diff_ind])
        df = pd.DataFrame({
            'Dataset': datasets,
            **{key: val['colour_difference'].values() for key, val in diff_res.items()}
        })
        tidy = df.melt(id_vars='Dataset', var_name='Method').rename(columns=str.title)
        sns.barplot(x='Dataset', y='Value', hue='Method', data=tidy, ax=ax)
        # ax.set_title('MacAdam 1974', fontsize=fontsize, fontweight='bold')
        ax.set_xlabel('', fontsize=fontsize)
        ax.set_ylabel(differences[diff_ind], fontsize=fontsize)
        ax.legend(fontsize=13, ncol=4, loc='lower center')

    method = 'Network'
    preds_mega_ordered = np.concatenate([
        preds[method]['colour_difference']['BFD1986'],
        preds[method]['colour_difference']['Leeds1997'],
        preds[method]['colour_discrimination']['RIT-DuPont1991'],
        preds[method]['colour_difference']['Witt1999'],
        preds[method]['colour_discrimination']['MacAdam1942'],
        preds[method]['colour_difference']['MacAdam1974'],
        preds[method]['colour_discrimination']['Gegenfurtner'],
        preds[method]['colour_difference']['TeamK'],
    ])

    ax = fig.add_subplot(gs[0, 2])
    metrics = plot_human_vs_method(preds_mega_ordered, mega_db, return_metrics=return_metrics,
                                   ax=ax, xlabel='Network Euclidean Distance', **kargs)
    if return_metrics:
        return metrics
    return fig


def euc_distance(a, b):
    return np.sum((a.astype('float32') - b.astype('float32')) ** 2, axis=-1) ** 0.5


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


def read_network_results_all_pts(res_dir, arch, test_data, exclude_list=None):
    if exclude_list is None:
        exclude_list = []
    net_result = dict()
    for area in arch_areas[arch]:
        area_result = []
        for ps in test_data.keys():
            if ps in exclude_list:
                continue
            for pind in range(len(test_data[ps]['ext'])):
                res_path = '%s/%s/evolution_%s_%d.csv' % (res_dir, area, ps, pind)
                if not os.path.exists(res_path):
                    continue
                current_result = np.loadtxt(res_path, delimiter=',')
                if len(current_result.shape) == 1:
                    current_result = [current_result]

                for test_pt in current_result:
                    area_result.append([test_pt[0], *test_data[ps]['ref'], *test_pt[1:4]])
        area_result = pd.DataFrame(
            area_result,
            columns=['DV', 'Ref-R', 'Ref-G', 'Ref-B', 'Test-R', 'Test-G', 'Test-B']
        )
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

    axis_names = ['Ax0', 'Ax1', 'Ax2'] if axis_names is None else axis_names
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


def summary_plot_all_nets_layers(results_dir, test_dir, ins_name='bg128_i0'):
    networks = ['clip_B32', 'clip_RN50', 'vit_b_32', 'resnet50']

    mega_db = pd.read_csv(f"{test_dir}/meta_dbs_srgb.csv")

    fig = plt.figure(figsize=(36, 24))
    ax_ind = 1

    dbs_dict = {
        'bfd': 'bfd_srgb',
        'leeds': 'leeds_srgb',
        'rit-duppont': 'rit-dupont_srgb',
        'witt': 'witt_srgb',
        'macadam1942': 'macadam1942_srgb_d65',
        'macadam1974': 'macadam1974_srgb_d65',
        'karl': 'karl_srgb',
        'teamk_m2j2': 'teamk_srgb_m2j2'
    }

    metrics_networks = {'pearson': dict(), 'spearman': dict(), 'stress': dict()}
    network_accuracies = []
    for net_name in networks:
        predb = 'clip' if 'clip' in net_name else 'imagenet'
        for _key in metrics_networks.keys():
            metrics_networks[_key][net_name] = []
        for block in arch_areas[net_name]:
            preds_mega_ordered = []
            for db_res_name, db_test_name in dbs_dict.items():
                net_dir = f"{results_dir}/bw_4afc_{db_res_name}/bg_128/{predb}/{net_name}/{ins_name}"

                if not os.path.exists(f"{net_dir}/{block}.csv"):
                    print(f"{net_dir}/{block}.csv")
                    continue
                block_res = pd.read_csv(f"{net_dir}/{block}.csv")
                network_accuracies.append(block_res.loc[:, 'Accuracy'].mean())
                network_prediction = block_res.loc[:, 'Accuracy'].to_numpy()
                # network_prediction = network_prediction ** (14)
                preds_mega_ordered.append(network_prediction)

            ax = fig.add_subplot(4, 6, ax_ind)
            ax_ind += 1
            preds_mega_ordered = np.concatenate(preds_mega_ordered)

            if net_name == 'clip_B32':
                net_label = 'CLIP ViT-B32'
            elif net_name == 'clip_RN50':
                net_label = 'CLIP ResNet50'
            elif net_name == 'vit_b_32':
                net_label = 'ImageNet ViT-B32'
            else:
                net_label = 'ImageNet ResNet50'
            xlabel = f"{net_label} {block} Accuracy"
            r_p, r_s, stress_val = plot_human_vs_method(
                preds_mega_ordered, mega_db, ax=ax, xlabel=xlabel, return_metrics=True
            )
            metrics_networks['pearson'][net_name].append(r_p)
            metrics_networks['spearman'][net_name].append(r_s)
            metrics_networks['stress'][net_name].append(stress_val)
    return fig, metrics_networks


def data_vs_metrics(data, **kwargs):
    methods = [
        'RGB',
        'CIE 1976', 'CIE 1994', 'CIE 2000',
        # 'CMC', 'ITP', 'CAM02-LCD', 'CAM02-SCD',
        # 'CAM02-UCS', 'CAM16-LCD', 'CAM16-SCD', 'CAM16-UCS',
        # 'DIN99',
    ]
    fig = plt.figure(figsize=(24, 6))
    for ax_ind, method_name in enumerate(methods):
        if method_name == 'RGB':
            ref_pt = data.loc[:, ['Ref-R', 'Ref-G', 'Ref-B']].to_numpy().astype('float32')
            test_pt = data.loc[:, ['Test-R', 'Test-G', 'Test-B']].to_numpy().astype('float32')
            method_prediction = np.linalg.norm(ref_pt - test_pt, axis=1)
        else:
            ref_lab = data.loc[:, ['Ref-L', 'Ref-a', 'Ref-b']]
            test_lab = data.loc[:, ['Test-L', 'Test-a', 'Test-b']]
            method_prediction = colour_science.delta_E(ref_lab, test_lab, method=method_name)
        method_prediction = np.array(method_prediction)

        if method_name == 'CIE 1976':
            xlabel = 'Lab Euclidean Distance'
        elif method_name == 'CIE 1994':
            xlabel = "$\Delta E_{1994}$"
        elif method_name == 'CIE 2000':
            xlabel = "$\Delta E_{2000}$"
        else:
            xlabel = 'RGB Euclidean Distance'

        ax = fig.add_subplot(1, 4, ax_ind + 1)
        plot_human_vs_method(method_prediction, data, xlabel=xlabel, ax=ax, **kwargs)
        if ax_ind != 0:
            ax.set_ylabel('')
            ax.set_yticklabels([])
    return fig


def plot_human_vs_method(method_prediction, data, ylabel=None, docorr=True,
                         xlabel=None, ax=None, return_metrics=False,
                         include_db=None, exclude_db=None):
    if ylabel is None:
        ylabel = 'Human Observed Difference'
    if ax is None:
        fig = plt.figure(figsize=(4, 6))
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig = None

    method_prediction = np.array(method_prediction)
    comparison_data = data.loc[:, 'DV']
    comparison_data = comparison_data.to_numpy()

    if include_db is not None:
        which_rows = data['Dataset'].isin(include_db)
    elif exclude_db is not None:
        which_rows = ~(data['Dataset'].isin(exclude_db))
    else:
        which_rows = np.arange(data.shape[0])

    if docorr:
        r_p, p = stats.pearsonr(method_prediction[which_rows], comparison_data[which_rows])
        r_s, p = stats.spearmanr(method_prediction[which_rows], comparison_data[which_rows])
        title_str = "$r_p$=%.2f    $r_s$=%.2f" % (r_p, r_s)
    else:
        cv = np.std(method_prediction) / np.mean(method_prediction)
        title_str = f"cv={cv:.02f}"
    stress_val = stress(method_prediction[which_rows], comparison_data[which_rows])

    fsize = 18
    ax.set_title("%s    $S$=%.2f" % (title_str, stress_val), fontsize=fsize)
    if 'Dataset' in data.columns:
        unique_dbs = np.unique(data['Dataset'])
        db_num_elements = [comparison_data[data['Dataset'] == db].shape[0] for db in unique_dbs]
        order_dbs = np.argsort(db_num_elements)[::-1]
        if 'Gegenfurtner' in unique_dbs:
            order_dbs = [5, 0, 1, 2, 7, 9, 4, 6, 8, 3]
        else:
            order_dbs = [0, 1, 2, 6, 8, 3, 4, 5, 7]

        legend_elements = []
        for db in unique_dbs[order_dbs]:
            if exclude_db is not None and db in exclude_db:
                continue
            if include_db is not None and db not in include_db:
                continue
            alpha = 0.5
            vis_w = 1
            db_label = db
            if 'BFD' in db:
                colour = 'blue'
                marker = 'D'
                db_label = 'BFD-1986' if 'D65' in db else '_'
                alpha = 0.25
            elif 'WITT' in db:
                colour = 'green'
                marker = 's'
                alpha = 1
                db_label = 'Witt-1999'
            elif 'RIT-DuPont' in db:
                colour = 'orange'
                marker = 'o'
                # alpha=1
                db_label = 'RIT-DuPont-1991'
            elif 'LEEDS' in db:
                colour = 'red'
                marker = 'H'
                # alpha=0.3
                db_label = 'Leeds-1997'
            elif 'Laysa' in db:
                colour = 'gray'
                marker = 'v'
                db_label = 'Hedjar-2024'
            elif 'Gegenfurtner' in db:
                colour = 'brown'
                marker = '^'
                db_label = 'Gegenfurtner-1992'
                vis_w = 1.2
            elif 'MacAdam1974' in db:
                colour = 'magenta'
                marker = 'p'
                # alpha = 1
                db_label = 'MacAdam-1974'
            elif 'MacAdam1942' in db:
                colour = 'lime'
                marker = '>'
                # alpha=0.3
                db_label = 'MacAdam-1942'
                vis_w = 0.8
            else:
                print('UPS!', db)
            ax.plot(method_prediction[data['Dataset'] == db],
                    comparison_data[data['Dataset'] == db] * vis_w, marker,
                    markeredgecolor=colour, alpha=alpha,  # label=db_label,
                    fillstyle='none')
            legend_elements.append(Line2D(
                [0], [0], marker=marker, color='w', label=db_label,
                markerfacecolor=colour, markersize=15))
        # legend_elements = [legend_elements[i] for i in [2, 1, 4, 3, 0, 5, 6, 7]]
        ax.legend(prop={'size': fsize - 5}, handles=legend_elements)
    else:
        ax.plot(method_prediction, comparison_data, 'x')
    ax.set_ylabel(ylabel, fontsize=fsize)
    xlabel = 'Distance' if xlabel is None else xlabel
    ax.set_xlabel(xlabel, fontsize=fsize)
    if return_metrics:
        return r_p, r_s, stress_val
    return ax if fig is None else fig


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


def sample_rgb(cube_samples=1000, achromatic=False):
    if achromatic:
        return np.expand_dims(
            np.stack([np.linspace(0, 1, 101) for _ in range(3)]), axis=1
        ).transpose(2, 1, 0)
    num_samples = round(cube_samples ** (1 / 3))
    linspace_vals = np.linspace(0, 1, num_samples)
    r_pts = np.tile(linspace_vals, (num_samples ** 2, 1)).T.reshape(-1, 1)
    g_pts = np.tile(linspace_vals, (num_samples, num_samples)).T.reshape(-1, 1)
    b_pts = np.tile(linspace_vals, (1, num_samples ** 2)).T.reshape(-1, 1)
    return np.stack((r_pts, g_pts, b_pts), axis=2)


def main(argv):
    parser = argparse.ArgumentParser(description='Optimising Colour Spaces!')
    parser.add_argument('--in_dir', required=True, type=str)
    parser.add_argument('--in_dir_fine_grain', default=None, type=str)
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

    if args.in_dir_fine_grain is not None:
        for layer in arch_areas[arch]:
            layer_data = pd.read_csv(f"{args.in_dir_fine_grain}/{layer}.csv")
            points = {
                'DV': layer_data['DV'],
                'Ref-RGB': layer_data.loc[:, ['Ref-R', 'Ref-G', 'Ref-B']].to_numpy(),
                'Test-RGB': layer_data.loc[:, ['Test-R', 'Test-G', 'Test-B']].to_numpy(),
            }
            optimise_layer(args, points, (pretrained_db, arch), layer)
    else:
        network_result_summary = parse_network_results(args.in_dir, arch, rgb_test_data)
        for layer in arch_areas[arch]:
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
        which_pts = val_pts if centre_ind in val_inds else train_pts
        which_map_inds = val_map_inds if centre_ind in val_inds else train_map_inds
        which_pts.append(centre_pt)
        cen_in_ind = len(which_pts) - 1
        for border_pt in border_data[centre_ind]:
            which_pts.append(border_pt)
            bor_in_ind = len(which_pts) - 1
            which_map_inds.append([cen_in_ind, bor_in_ind])
    train_pts = np.array(train_pts, dtype='float32')
    train_map_inds = np.array(train_map_inds)
    val_pts = np.array(val_pts, dtype='float32')
    val_map_inds = np.array(val_map_inds)

    train_data = {
        'DV': np.ones(len(train_map_inds)),
        'Ref-RGB': train_pts[train_map_inds[:, 0]],
        'Test-RGB': train_pts[train_map_inds[:, 1]]
    }
    if val_perc > 0:
        val_data = {
            'DV': np.ones(len(val_map_inds)),
            'Ref-RGB': val_pts[val_map_inds[:, 0]],
            'Test-RGB': val_pts[val_map_inds[:, 1]]
        }
    else:
        val_data = None

    return train_data, val_data


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
    'SparseAdam': torch.optim.SparseAdam,
    'AdamW': torch.optim.AdamW
}


def optimise_layer(args, network_result_summary, pretrained, layer):
    pretrained_db, pretrained_arch = pretrained
    layer_out_dir = '%s/%s/%s/%s/' % (args.out_dir, pretrained_db, pretrained_arch, layer)

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
                    if args.in_dir_fine_grain is not None:
                        optimise_points(args, network_result_summary, out_dir)
                    else:
                        optimise_instance(args, network_result_summary[layer], out_dir)

                    sys.stdout = orig_stdout
                    f.close()


def optimise_instance(args, layer_results, out_dir):
    train_points, _ = train_val_sets(layer_results, 0)
    return optimise_points(args, train_points, out_dir)


def optimise_all_layers(args, network_results, out_dir):
    train_points = {
        'DV': [],
        'Ref-RGB': [],
        'Test-RGB': [],
    }
    for layer_results in network_results.values():
        layer_points, _ = train_val_sets(layer_results, 0)
        for key, val in layer_points.items():
            train_points[key].extend(val)
    for key, val in train_points.items():
        train_points[key] = np.array(val)
    return optimise_points(args, train_points, out_dir)


def optimise_points(args, points, out_dir):
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

    mega_db = pd.read_csv(f"{args.human_data_dir}/meta_dbs_srgb.csv")
    human_gts = {
        'Leeds1997': mega_db.loc[mega_db['Dataset'] == 'LEEDS', 'DV'],
        'Witt1999': mega_db.loc[mega_db['Dataset'] == 'WITT', 'DV'],
        'MacAdam1974': mega_db.loc[mega_db['Dataset'] == 'MacAdam1974', 'DV'],
        'BFD1986': mega_db.loc[
            mega_db['Dataset'].isin(['BFD-P(D65)', 'BFD-P( C )', 'BFD-P(M)']), 'DV'],
        'TeamK': mega_db.loc[mega_db['Dataset'] == 'Laysa2024', 'DV'],
    }
    comparison_data = mega_db.loc[:, 'DV']
    comparison_data = comparison_data.to_numpy()

    dv = torch.tensor(points['DV']).float()
    print(dv.shape)
    best_stress = np.inf
    for epoch in range(args.epochs):
        model = model.train()
        with torch.set_grad_enabled(True):
            ref_pts_in = torch.tensor(points['Ref-RGB'].copy()).float()
            ref_pts_out = model(ref_pts_in)
            test_pts_in = torch.tensor(points['Test-RGB'].copy()).float()
            test_pts_out = model(test_pts_in)
            de = torch.linalg.norm(ref_pts_out - test_pts_out, ord=2, dim=1)
            min_vals, _ = ref_pts_out.min(axis=0)
            max_vals, _ = ref_pts_out.max(axis=0)
            deltad = max_vals - min_vals
            # uniformity_loss = torch.std(euc_dis)  # / torch.mean(euc_dis)
            uniformity_loss = stress_torch(de, dv)
            if args.loss == 'range':
                range_loss = 0.5 * (abs(1 - deltad[0]) + abs(1 - deltad[1]) + abs(1 - deltad[2]))
            elif args.loss == 'mean_distance':
                range_loss = 0.5 * abs(0.1 - torch.mean(de))
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

        human_pred = predict_human_data({'Network': model}, args.human_data_dir)
        dis_metrics = ['stress', 'cv']
        eval_discrimination = [evaluate_discrimination(human_pred, _m) for _m in dis_metrics]
        diff_metrics = ['stress', 'pearson']
        eval_differences = [evaluate_difference(human_pred, _m, human_gts) for _m in diff_metrics]

        epoch_loss = [uniformity_loss.item()]
        headers = ['loss']

        tmp_strs_i = []
        for m_ind, _eval in enumerate(eval_discrimination):
            tmp_strs_j = []
            for key, val in _eval['Network']['colour_discrimination'].items():
                tmp_strs_j.append('%s=%.2f' % (key[:2], val))
                epoch_loss.append(val)
                headers.append(f"{dis_metrics[m_ind]}_{key}")
            tmp_strs_i.append(
                f"{dis_metrics[m_ind]} [" + ' '.join(_tj for _tj in tmp_strs_j) + ']'
            )
        disc_str = ' '.join(_ti for _ti in tmp_strs_i)

        tmp_strs_i = []
        for m_ind, _eval in enumerate(eval_differences):
            tmp_strs_j = []
            for key, val in _eval['Network']['colour_difference'].items():
                tmp_strs_j.append('%s=%.2f' % (key[:2], val))
                epoch_loss.append(val)
                headers.append(f"{diff_metrics[m_ind]}_{key}")
            tmp_strs_i.append(
                f"{diff_metrics[m_ind]} [" + ' '.join(_tj for _tj in tmp_strs_j) + ']'
            )
        diff_str = ' '.join(_ti for _ti in tmp_strs_i)

        method = 'Network'
        preds_mega_ordered = np.concatenate([
            human_pred[method]['colour_difference']['BFD1986'],
            human_pred[method]['colour_difference']['Leeds1997'],
            human_pred[method]['colour_discrimination']['RIT-DuPont1991'],
            human_pred[method]['colour_difference']['Witt1999'],
            human_pred[method]['colour_discrimination']['MacAdam1942'],
            human_pred[method]['colour_difference']['MacAdam1974'],
            human_pred[method]['colour_discrimination']['Gegenfurtner'],
            human_pred[method]['colour_difference']['TeamK'],
        ])

        r_p, p = stats.pearsonr(preds_mega_ordered, comparison_data)
        r_s, p = stats.spearmanr(preds_mega_ordered, comparison_data)
        stress_val = stress(preds_mega_ordered, comparison_data)
        epoch_loss.append(r_p)
        epoch_loss.append(r_s)
        epoch_loss.append(stress_val)
        headers.append('pearson_mega')
        headers.append('spearman_mega')
        headers.append('stress_mega')

        losses.append(epoch_loss)

        if stress_val < best_stress:
            best_stress = stress_val
            torch.save({
                'state_dict': model.state_dict(),
                'units': args.num_units,
                'nonlinearities': args.nonlinearities,
                'mean_std': mean_std
            }, '%s/model_best_stress.pth' % out_dir)

        if np.mod(epoch, print_freq) == 0 or epoch == (args.epochs - 1):
            print(
                '[%.5d] loss=%.4f \n\t%s \n\t%s' % (epoch, uniformity_loss, disc_str, diff_str)
            )

    rgb_pts, rgb_pts_pred = rgb_mapping(model)
    space_range = list(rgb_pts_pred.max(axis=(0, 1)) - rgb_pts_pred.min(axis=(0, 1)))
    print('Network-space range:\t%s (%.3f, %.3f %.3f)' % ('', *space_range))
    fig = plot_colour_pts(
        rgb_pts_pred, rgb_pts,
        'loss=%.4f   %s %s' % (uniformity_loss, disc_str, diff_str),
        axs_range='auto'
    )

    fig.savefig('%s/rgb_pred.svg' % out_dir)
    plt.close('all')

    df = pd.DataFrame(columns=headers, data=losses)
    df.to_csv('%s/losses.csv' % out_dir)

    torch.save({
        'state_dict': model.state_dict(),
        'units': args.num_units,
        'nonlinearities': args.nonlinearities,
        'mean_std': mean_std
    }, '%s/model.pth' % out_dir)


def rgb_mapping(model, cube_samples=1000, achromatic=False):
    if isinstance(model, str):
        model = load_model(model)
    rgb_pts = sample_rgb(cube_samples, achromatic)
    rgb_squeezed = rgb_pts.copy().squeeze()
    rgb_pts_pred = pred_model(model, rgb_squeezed)
    rgb_pts_pred = np.expand_dims(rgb_pts_pred, axis=1)
    return rgb_pts, rgb_pts_pred


if __name__ == '__main__':
    main(sys.argv[1:])
