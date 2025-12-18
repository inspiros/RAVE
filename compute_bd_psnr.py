import argparse
import os
import scipy.interpolate
import scipy.integrate
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default='output')
    parser.add_argument('--models', type=str, nargs='+', default=[
        'multilevel_interp_random', 'multilevel_interp_scaling', 'multilevel_interp_scaling2', 'multilevel_interp_opacity',
        'multilevel_interp_geometry', 'multilevel_interp_global', 'multilevel_interp'])
    parser.add_argument('--base_model', type=str, default='multilevel_interp')
    parser.add_argument('--datasets', type=str, nargs='+', default=['mipnerf360'])
    parser.add_argument('--distortion', type=str, choices=['psnr', 'ssim', 'lpips', 'fps'], default='psnr')
    parser.add_argument('--method', choices=['trapezoid', 'simpson'], default='trapezoid')
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


def bd_psnr(R1, PSNR1, R2, PSNR2, piecewise=0):
    lR1 = np.log(R1)
    lR2 = np.log(R2)

    PSNR1 = np.array(PSNR1)
    PSNR2 = np.array(PSNR2)

    p1 = np.polyfit(lR1, PSNR1, 3)
    p2 = np.polyfit(lR2, PSNR2, 3)

    # integration interval
    min_int = max(min(lR1), min(lR2))
    max_int = min(max(lR1), max(lR2))

    # find integral
    if piecewise == 0:
        p_int1 = np.polyint(p1)
        p_int2 = np.polyint(p2)

        int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
        int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)
    else:
        # See https://chromium.googlesource.com/webm/contributor-guide/+/master/scripts/visual_metrics.py
        lin = np.linspace(min_int, max_int, num=100, retstep=True)
        interval = lin[1]
        samples = lin[0]
        v1 = scipy.interpolate.pchip_interpolate(np.sort(lR1), PSNR1[np.argsort(lR1)], samples)
        v2 = scipy.interpolate.pchip_interpolate(np.sort(lR2), PSNR2[np.argsort(lR2)], samples)
        # Calculate the integral using the trapezoid method on the samples.
        int1 = np.trapz(v1, dx=interval)
        int2 = np.trapz(v2, dx=interval)

    # find avg diff
    avg_diff = (int2-int1)/(max_int-min_int)
    return avg_diff


def main():
    args = parse_args()
    distortion_ind = {
        'psnr': 0,
        'ssim': 1,
        'lpips': 2,
        'fps': 4,
    }[args.distortion]
    integrate_func = getattr(scipy.integrate, args.method)

    datasets = to_list(args.datasets, sorted(os.listdir(args.output_path)))
    args.models = to_list(args.models)
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

        # plot
        for config in res.keys():
            base_model_result = res[config].pop(args.base_model)
            base_distortion = base_model_result[:, distortion_ind]
            base_size = base_model_result[:, -1]
            base_area = integrate_func(base_distortion, base_size)
            for model, result in res[config].items():
                distortion = result[:, distortion_ind]
                size = result[:, -1]
                if model in ['multilevel', 'non_hierarchical']:
                    size += 0.031561851501464844  # checkpoints size
                area = integrate_func(distortion, size)
                print(f'[{model}] BD-PSNR={area - base_area:.05f}, '
                      f'BD-PSNR={bd_psnr(base_size, base_distortion, size, distortion):.05f}')


if __name__ == '__main__':
    main()
