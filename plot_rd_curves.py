import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_path = 'D:\\data\\fonts\\lm\\fonts\\opentype\\public\\lm\\lmroman12-regular.otf'  # Your font path goes here
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()
plt.rcParams['font.size'] = 16

# plt.style.use('ggplot')


METHODS_NAMES = {
    '3dgs': '3DGS',
    'cat3dgs': 'CAT-3DGS',
    'compact3dgs': 'Compact-3DGS',
    'compgs': 'CompGS',
    'contextgs': 'ContextGS',
    'eagles': 'EAGLES',
    'elmgs': 'ELMGS',
    'hac': 'HAC',
    'hac++': 'HAC++',
    'lightgaussian': 'LightGaussian',
    'pcgs': 'PCGS',
    'rdo': 'RDO',
    'reduced_3dgs': 'Reduced-3DGS',
    'scaffold': 'Scaffold-GS',
    'sog': 'SOG',
    'trimmingthefat': 'Trimming the Fat',
    'multilevel': 'GoDe',
    # ours
    'multilevel_interp': 'RAVE (Ours)',
}
METHODS_ARGS = {
    '3dgs': dict(color='lime'),
    'cat3dgs': dict(color='skyblue', marker='d'),
    'compact3dgs': dict(color='teal', marker='<'),
    'compgs': dict(color='salmon', marker='v'),
    'contextgs': dict(color='tab:cyan', marker='8'),
    'eagles': dict(color='slategray', marker='*'),
    'elmgs': dict(color='maroon', marker='$*$'),
    'hac': dict(color='tab:brown', marker='P'),
    'hac++': dict(color='peru', marker='$\\boldsymbol{+}\\boldsymbol{+}$'),
    'lightgaussian': dict(color='tab:purple', marker='^'),
    'pcgs': dict(color='tab:pink', marker='o'),
    'rdo': dict(color='tab:orange', marker='s'),
    'reduced_3dgs': dict(color='tab:green', marker='D'),
    'scaffold': dict(color='gold', marker='2'),
    'sog': dict(color='tab:olive', marker='H'),
    'trimmingthefat': dict(color='tomato', marker='X'),
    'multilevel': dict(color='r', marker='.'),
    # ours
    'multilevel_interp': dict(color='tab:blue'),
}
for method in METHODS_ARGS.keys():
    m_args = METHODS_ARGS[method]
    if method.startswith('multilevel_interp'):
        m_args.update(marker='none', linestyle='-', linewidth=3)
    else:
        m_args.update(markersize=6, linestyle='--', linewidth=1, alpha=0.75)
        if method == 'pcgs':
            m_args.update(linestyle='-')
        if method == 'hac++':
            m_args.update(markersize=12)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default='output')
    parser.add_argument('--other_models_path', type=str, default=None)
    parser.add_argument('--models', type=str, nargs='+', default='multilevel_interp')
    parser.add_argument('--other_models', type=str, nargs='+', default=None)
    parser.add_argument('--datasets', type=str, nargs='+', default=['mipnerf360', 'tandt', 'db'])
    parser.add_argument('--distortion', type=str, choices=['psnr', 'ssim', 'lpips'], default='psnr')
    parser.add_argument('--include_baseline', action='store_true')
    parser.add_argument('--export_legend', action='store_true')
    args = parser.parse_args()
    # TODO: remove
    import platform
    if platform.system() == 'Windows':
        # args.other_models = ['pcgs', 'cat3dgs', 'contextgs', 'hac++', 'hac', 'rdo', 'elmgs', 'compgs', 'sog',
        #                      'reduced_3dgs', 'compact3dgs', 'trimmingthefat', 'lightgaussian'][::-1]
        args.other_models = ['pcgs', 'hac++', 'hac', 'rdo', 'compgs',
                             'reduced_3dgs', 'lightgaussian'][::-1]
    return args


def get_anchor():
    from svgpath2mpl import parse_path
    anchor = parse_path("M 12.971,160.00042 H 45.365 C 67.172,57.265419 181.944,4.1938112e-4 288,4.1938112e-4 394.229,4.1938112e-4 508.853,57.380419 530.635,160.00042 h 32.394 c 10.691,0 16.045,12.926 8.485,20.485 l -67.029,67.029 c -4.686,4.686 -12.284,4.686 -16.971,0 l -67.029,-67.029 c -7.56,-7.56 -2.206,-20.485 8.485,-20.485 h 35.146 C 443.826,105.68342 379.153,73.412419 319.999,65.985419 V 256.00042 h 52 c 6.627,0 12,5.373 12,12 v 40 c 0,6.627 -5.373,12 -12,12 h -52 v 5.47 c 37.281,13.178 63.995,48.725 64,90.518 0.006,52.24 -42.394,95.274 -94.629,96.002 -53.647,0.749 -97.37,-42.515 -97.37,-95.99 0,-41.798 26.716,-77.35 64,-90.53 v -5.47 h -52 c -6.627,0 -12,-5.373 -12,-12 v -40 c 0,-6.627 5.373,-12 12,-12 h 52 V 65.985419 c -58.936,7.399 -123.82,39.679001 -144.117,94.015001 h 35.146 c 10.691,0 16.045,12.926 8.485,20.485 l -67.029,67.029 c -4.686,4.686 -12.284,4.686 -16.971,0 l -67.029,-67.029 c -7.559,-7.559 -2.205,-20.485 8.486,-20.485 z m 275.029,288 c 17.645,0 32,-14.355 32,-32 0,-17.645 -14.355,-32 -32,-32 -17.645,0 -32,14.355 -32,32 0,17.645 14.355,32 32,32 z")
    anchor.vertices -= anchor.vertices.mean(axis=0)
    return anchor


def load_csv(path, delimiter=',', offset=0, dtype=np.float64):
    with open(path, 'r') as f:
        lines = map(lambda _: _.strip(), f.readlines()[offset:])
        lines = [list(map(float, _.replace(' ', '').split(delimiter))) for _ in lines]
    return np.asarray(lines, dtype=dtype)


def to_list(l, default=None):
    if isinstance(l, list):
        return l
    return [l] if l is not None else default


def export_legend(legend, path, expand=None):
    if expand is None:
        expand = [-5, -5, 5, 5]
    fig = legend.figure
    legend.axes.axis('off')
    fig.canvas.draw()
    bbox  = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.asarray(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(path, dpi='figure', bbox_inches=bbox, transparent=True)


def main():
    args = parse_args()
    if args.other_models_path is None:
        args.other_models_path = os.path.join(args.output_path, 'others')
    distortion_ind = {
        'psnr': 0,
        'ssim': 1,
        'lpips': 2,
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

            for other_model in args.other_models:
                other_model_path = os.path.join(args.other_models_path, dataset, other_model + '.csv')
                other_result = np.loadtxt(other_model_path, delimiter=',')
                if other_result.ndim == 1:
                    other_result = other_result[np.newaxis, :]
                distortion = other_result[:, distortion_ind]
                size = other_result[:, 3]
                extra_args = METHODS_ARGS[other_model]
                ax.plot(size, distortion, label=METHODS_NAMES.get(other_model, other_model), **extra_args)

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
                ax.plot(size, distortion, label=METHODS_NAMES.get(model, model), **extra_args)
                # if model == 'multilevel_interp':  # plot anchors (hard-coded)
                #     ax.plot(size[::7], distortion[::7], color='r',
                #             marker=get_anchor(), markersize=12, linestyle='none')
            ax.set_xlabel('Size (MB)$\\downarrow$')
            ax.set_ylabel({
                'psnr': 'PSNR$\\uparrow$',
                'ssim': 'SSIM$\\uparrow$',
                'lpips': 'LPIPS$\\downarrow$',
            }[args.distortion])
            ax.grid(True)
            # ax.legend(loc='best')
            if args.export_legend:
                leg = ax.legend(ncol=2)
                while len(ax.artists):
                    ax.artists[0].remove()
                while len(ax.lines):
                    ax.lines[0].remove()
                export_legend(leg, os.path.join(args.output_path, 'legend.svg'))
            fig.tight_layout()
            fig.savefig(os.path.join(args.output_path, f'{dataset}_{args.distortion}.svg'), transparent=True)
            print(f'saved figure for {dataset}')
            plt.show()


if __name__ == '__main__':
    main()
