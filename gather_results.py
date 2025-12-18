"""
python gather_results.py output/[model_path]
"""
import argparse
import os
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('output_path', type=str)
    parser.add_argument('--datasets', type=str, nargs='+', default=None)
    parser.add_argument('--scenes', type=str, nargs='+', default=None)
    parser.add_argument('--no_config', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()

    datasets = sorted(os.listdir(args.output_path)) if args.datasets is None else args.datasets
    for dataset in datasets:
        print(f'[Gathering results for {dataset}]')
        dataset_path = os.path.join(args.output_path, dataset)
        res = {}
        for scene in sorted(os.listdir(dataset_path)) if args.scenes is None else args.scenes:
            scene_path = os.path.join(dataset_path, scene)
            if not os.path.isdir(scene_path):
                continue
            for config in sorted(os.listdir(scene_path)) if not args.no_config else ['summary']:
                if not args.no_config:
                    config_path = os.path.join(scene_path, config)
                    result_csv = os.path.join(config_path, 'results.csv')
                else:
                    result_csv = os.path.join(scene_path, 'results.csv')
                if not os.path.exists(result_csv):
                    print('{} not exists, skipping'.format(result_csv))
                    continue
                with open(result_csv, 'r') as f:
                    lines = map(lambda _: _.strip(), f.readlines())
                    lines = [_.replace(' ', '').split(',')[1:] for _ in lines if not _.endswith('None')]
                    if not len(lines):
                        print('{} not finalized, skipping'.format(result_csv))
                        continue
                result = np.asarray(lines, dtype=np.float64)
                print(scene, result.shape)
                if config not in res:
                    res[config] = []
                res[config].append(result)
        for config in res.keys():
            n_scenes = len(res[config])
            res[config] = np.mean(res[config], axis=0)
            avg_result_csv = os.path.join(dataset_path, f'{config}.csv')
            print(f'{avg_result_csv}: {n_scenes} scenes')
            with open(avg_result_csv, 'w') as f:
                print('psnr_test, ssim_test, lpips_test, num_anchor, fps, num_gaussians, size', file=f)
                for line in res[config].tolist():
                    print(', '.join(map(str, line)), file=f)
        if len(res):
            print()


if __name__ == '__main__':
    main()
