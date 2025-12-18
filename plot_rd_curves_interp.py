import argparse
import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from svgpath2mpl import parse_path

from matplotlib import font_manager
font_path = 'D:\\data\\fonts\\lm\\fonts\\opentype\\public\\lm\\lmroman12-regular.otf'  # Your font path goes here
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()
plt.rcParams['font.size'] = 16


# plt.style.use('ggplot')


def get_anchor():
    anchor = parse_path(
        "M 12.971,160.00042 H 45.365 C 67.172,57.265419 181.944,4.1938112e-4 288,4.1938112e-4 394.229,4.1938112e-4 508.853,57.380419 530.635,160.00042 h 32.394 c 10.691,0 16.045,12.926 8.485,20.485 l -67.029,67.029 c -4.686,4.686 -12.284,4.686 -16.971,0 l -67.029,-67.029 c -7.56,-7.56 -2.206,-20.485 8.485,-20.485 h 35.146 C 443.826,105.68342 379.153,73.412419 319.999,65.985419 V 256.00042 h 52 c 6.627,0 12,5.373 12,12 v 40 c 0,6.627 -5.373,12 -12,12 h -52 v 5.47 c 37.281,13.178 63.995,48.725 64,90.518 0.006,52.24 -42.394,95.274 -94.629,96.002 -53.647,0.749 -97.37,-42.515 -97.37,-95.99 0,-41.798 26.716,-77.35 64,-90.53 v -5.47 h -52 c -6.627,0 -12,-5.373 -12,-12 v -40 c 0,-6.627 5.373,-12 12,-12 h 52 V 65.985419 c -58.936,7.399 -123.82,39.679001 -144.117,94.015001 h 35.146 c 10.691,0 16.045,12.926 8.485,20.485 l -67.029,67.029 c -4.686,4.686 -12.284,4.686 -16.971,0 l -67.029,-67.029 c -7.559,-7.559 -2.205,-20.485 8.486,-20.485 z m 275.029,288 c 17.645,0 32,-14.355 32,-32 0,-17.645 -14.355,-32 -32,-32 -17.645,0 -32,14.355 -32,32 0,17.645 14.355,32 32,32 z")
    anchor.vertices -= anchor.vertices.mean(axis=0)
    return anchor


def get_magnifier():
    magnifier = parse_path(
        "m 505,69.35 -99.7,99.7 c -4.5,4.5 -10.6,7 -17,7 H 372 c 27.6,35.3 44,79.7 44,128 0,114.9 -93.1,208 -208,208 -114.9,0 -208,-93.1 -208,-208 0,-114.9 93.1,-208 208,-208 48.3,0 92.7,16.4 128,44 v -16.3 c 0,-6.4 2.5,-12.5 7,-17 l 99.7,-99.7 c 9.4,-9.4 24.6,-9.4 33.9,0 l 28.3,28.3 c 9.4,9.4 9.4,24.6 0.1,34 z m -297,106.7 c -70.7,0 -128,57.2 -128,128 0,70.7 57.2,128 128,128 70.7,0 128,-57.2 128,-128 0,-70.7 -57.2,-128 -128,-128 z")
    magnifier.vertices -= magnifier.vertices.mean(axis=0)
    return magnifier


def indicate_inset_zoom(ax, x1, y1, x2, y2, **kwargs):
    default_kwargs = dict(linewidth=1, edgecolor='k', facecolor='none')
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, **{**default_kwargs, **kwargs})
    ax.add_patch(rect)
    return rect


def export_legend(legend, path, expand=None):
    if expand is None:
        expand = [-5, -5, 5, 5]
    fig = legend.figure
    legend.axes.axis('off')
    fig.canvas.draw()
    bbox = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.asarray(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(path, dpi='figure', bbox_inches=bbox, transparent=True)


ZOOM_COLOR = 'lightcoral'
# ZOOM_COLOR = 'gray'
METHODS_NAMES = {
    'multilevel_interp': 'Local Ranking (Ours)',
    'multilevel_interp_random': 'Random Selection',
    'multilevel_interp_opacity': 'Opacity Ranking',
    'multilevel_interp_scaling': 'Scale Ranking$\\nearrow$',
    'multilevel_interp_scaling2': 'Scale Ranking$\\searrow$',
    'multilevel_interp_geometry': 'Geometry Gradient Ranking',
    'multilevel_interp_xyz': 'Location Gradient Ranking',
    'multilevel_interp_global': 'Global Ranking',
    '50levels': 'GoDe + Multiple Anchors',
    'continuous': 'GoDe w/o Iterative Masking + Continuous Training',
}
METHODS_ARGS = {
    'multilevel_interp': dict(color='tab:blue'),
    'multilevel_interp_random': dict(color='tab:orange'),
    'multilevel_interp_opacity': dict(color='tab:purple'),
    'multilevel_interp_scaling': dict(color='tab:green'),
    'multilevel_interp_scaling2': dict(color='lime'),
    'multilevel_interp_geometry': dict(color='tab:cyan'),
    'multilevel_interp_xyz': dict(color='tab:olive'),
    'multilevel_interp_global': dict(color='tab:olive'),
    '50levels': dict(color='tab:green'),
    'continuous': dict(color='tab:orange'),
}
for method in METHODS_ARGS.keys():
    m_args = METHODS_ARGS[method]
    m_args.update(marker='o', markersize=3, linestyle='-')
    if method.startswith('multilevel_interp'):
        m_args.update(linewidth=2)
        if method != 'multilevel_interp':
            m_args.update(alpha=0.8)
    else:
        m_args.update(linewidth=1, alpha=0.8)
        if method == '50levels':
            m_args.update(linestyle=':', marker=get_anchor(), markersize=8)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default='output')
    parser.add_argument('--other_models_path', type=str, default=None)
    parser.add_argument('--models', type=str, nargs='+', default=[
        'multilevel_interp_random', 'multilevel_interp_opacity', 'multilevel_interp_scaling',
        'multilevel_interp'])
    parser.add_argument('--other_models', type=str, nargs='+', default=None)
    parser.add_argument('--datasets', type=str, nargs='+', default=['mipnerf360'])
    parser.add_argument('--distortion', type=str, choices=['psnr', 'ssim', 'lpips', 'fps'], default='psnr')
    parser.add_argument('--include_baseline', action='store_true')
    parser.add_argument('--export_legend', action='store_true')
    parser.add_argument('--zoom', action='store_true')
    args = parser.parse_args()
    return args


def load_csv(path, delimiter=',', offset=0, dtype=np.float64):
    with open(path, 'r') as f:
        lines = map(lambda _: _.strip(), f.readlines()[offset:])
        lines = [list(map(float, _.replace(' ', '').split(delimiter))) for _ in lines]
    return np.asarray(lines, dtype=dtype)


def to_list(l, default=None):
    if isinstance(l, list):
        return l
    return [l] if l is not None else default


def main():
    args = parse_args()
    if args.other_models_path is None:
        args.other_models_path = os.path.join(args.output_path, 'others')
    distortion_ind = {
        'psnr': 0,
        'ssim': 1,
        'lpips': 2,
        'fps': 4,
    }[args.distortion]

    datasets = to_list(args.datasets, sorted(os.listdir(args.output_path)))
    args.models = to_list(args.models)
    args.other_models = to_list(args.other_models, [])
    print(args.models)

    for dataset in datasets:
        res = {}
        for model in sorted(os.listdir(args.output_path)) if args.models is None else args.models:
            model_path = os.path.join(args.output_path, model)
            if not os.path.isdir(model_path):
                continue
            dataset_path = os.path.join(model_path, dataset)
            for result_file in list(filter(lambda _: _.endswith('.csv'), sorted(os.listdir(dataset_path)))):
                config = os.path.splitext(result_file)[0]
                result_path = os.path.join(dataset_path, result_file)
                result = load_csv(result_path, offset=1)
                if config not in res:
                    res[config] = {}
                if model not in res[config]:
                    res[config][model] = {}
                res[config][model] = result

        if 'summary' in res:
            if args.include_baseline:
                for model in res['summary'].keys():
                    for config in filter(lambda _: _ != 'summary', res.keys()):
                        res[config][model] = res['summary'][model]
            del res['summary']

        # plot
        for config in res.keys():
            fig, ax = plt.subplots()
            if args.zoom:
                axins = inset_axes(
                    # ax, width='80%', height='60%',
                    ax, width='40%', height='80%',
                    loc='lower right')
                axins.spines['bottom'].set_color(ZOOM_COLOR)
                axins.spines['top'].set_color(ZOOM_COLOR)
                axins.spines['right'].set_color(ZOOM_COLOR)
                axins.spines['left'].set_color(ZOOM_COLOR)
                axins.spines['bottom'].set_linewidth(2)
                axins.spines['top'].set_linewidth(2)
                axins.spines['right'].set_linewidth(2)
                axins.spines['left'].set_linewidth(2)

            for model, result in res[config].items():
                distortion = result[:, distortion_ind]
                size = result[:, -1]
                if model in ['multilevel', 'non_hierarchical']:
                    size += 0.031561851501464844  # checkpoints size

                extra_args = {}
                if model in METHODS_ARGS:
                    extra_args.update(METHODS_ARGS[model])
                if result.shape[0] == 1:
                    extra_args.update(linestyle='none')
                ax.plot(size, distortion,
                        label=METHODS_NAMES.get(model, model), **extra_args)
                if model == 'multilevel_interp':  # plot anchors (hard-coded)
                    ax.plot(size[::7], distortion[::7], color='r',
                            marker=get_anchor(), markersize=15, linestyle='none')
                if args.zoom:
                    axins.plot(size, distortion,
                               label=METHODS_NAMES.get(model, model), **extra_args)
                    if model == 'multilevel_interp':  # plot anchors (hard-coded)
                        axins.plot(size[::7], distortion[::7], color='r',
                                   marker=get_anchor(), markersize=15, linestyle='none')
            ax.set_xlabel('Size (MB)$\\downarrow$')
            ax.set_ylabel({
                              'psnr': 'PSNR$\\uparrow$',
                              'ssim': 'SSIM$\\uparrow$',
                              'lpips': 'LPIPS$\\downarrow$',
                              'fps': 'FPS$\\uparrow$',
                          }[args.distortion])
            ax.grid(True)
            if args.zoom:
                axins.set_xticklabels([])
                axins.set_yticklabels([])
                axins.grid(True)
                # axins.set_xlim(7.5, 22.5)
                # axins.set_ylim(26.85, 27.85)
                axins.set_xlim(3.5, 8.5)
                axins.set_ylim(24.85, 27)
                axins.plot([axins.get_xlim()[1] - .7], [axins.get_ylim()[0] + .1], marker=get_magnifier(),
                           markersize=25, color=ZOOM_COLOR, linestyle='none')
                indicate_inset_zoom(ax, axins.get_xlim()[0], axins.get_ylim()[0],
                                    axins.get_xlim()[1], axins.get_ylim()[1],
                                    edgecolor=ZOOM_COLOR, linewidth=2, zorder=5)
            # ax.legend(loc='best')
            if args.export_legend:
                leg = ax.legend(ncol=1)
                while len(ax.artists):
                    ax.artists[0].remove()
                while len(ax.lines):
                    ax.lines[0].remove()
                export_legend(leg, os.path.join(args.output_path, 'ablation_legend.svg'))
            fig.tight_layout()
            # fig.savefig(os.path.join(args.output_path, f'{dataset}_{args.distortion}.png'))
            fig.savefig(os.path.join(args.output_path, f'ablation_{dataset}_{args.distortion}.svg'), transparent=True)
            print(f'saved figure for {dataset}')
            plt.show()


if __name__ == '__main__':
    main()
