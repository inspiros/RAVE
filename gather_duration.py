"""
python gather_results.py output/[model_path]
"""
import argparse
import os
import numpy as np
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('output_path', type=str)
    parser.add_argument('--datasets', type=str, nargs='+', default=None)
    parser.add_argument('--scenes', type=str, nargs='+', default=None)
    parser.add_argument('--no_config', action='store_true')
    return parser.parse_args()


def parse_time(info_line):
    time_str = info_line[:info_line.find(' - INFO')]
    return datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S,%f')


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
                    log_path = os.path.join(config_path, 'outputs.log')
                else:
                    log_path = os.path.join(scene_path, 'outputs.log')
                if not os.path.exists(log_path):
                    print('{} not exists, skipping'.format(log_path))
                    continue
                with open(log_path, 'r') as f:
                    lines = list(map(lambda _: _.strip(), f.readlines()))
                    if not len(lines):
                        print('{} not finalized, skipping'.format(log_path))
                        continue
                    l_start = next(filter(lambda _: _[_.find('INFO:') + 6:].startswith('args'), lines))
                    l_end = next(filter(lambda _: not _[_.find('INFO:') + 6:].startswith('args'), lines[1:]))
                    duration = (parse_time(l_end) - parse_time(l_start)).total_seconds()
                print(f'{scene}, {duration:.5f}')
                if config not in res:
                    res[config] = []
                res[config].append(duration)
        for config in res.keys():
            n_scenes = len(res[config])
            res[config] = np.mean(res[config], axis=0)
            print(f'avg_duration={res[config].item():.5f}: {n_scenes} scenes')
        if len(res):
            print()


if __name__ == '__main__':
    main()
