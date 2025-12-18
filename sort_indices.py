#!/usr/bin/env python3

import copy
import os
import numpy as np

import subprocess
# cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
# result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
# os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

os.system('echo $CUDA_VISIBLE_DEVICES')


import torch
import lpips
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import prefilter_voxel, render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams


def geometric_progression(tmin, tmax, L):
    b = np.exp((np.log(tmax) - np.log(tmin)) / (L - 1))
    arr = np.array([i for i in range(L)])
    return ((b ** arr) * tmin).astype(np.int32)


def accumulate_gradients(scene, gaussians, opt, pipe, background, mask):
    for viewpoint_cam in scene.getTrainCameras():
        voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background)
        if mask is not None:
            voxel_visible_mask = torch.logical_and(voxel_visible_mask, mask)

        render_pkg = render(viewpoint_cam, gaussians, pipe, background, visible_mask=voxel_visible_mask, retain_grad=True)
        image = render_pkg["render"]

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_loss = (1.0 - ssim(image, gt_image))
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss

        loss.backward(retain_graph=True)


def iterative_masking(scene, dataset, opt, pipe):
    gaussians = scene.gaussians
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    num_anchors = gaussians._anchor.shape[0]
    anchor_mask = torch.ones(num_anchors, dtype=torch.bool, device="cuda")
    anchor_indices = torch.zeros(num_anchors, dtype=torch.long)

    i = 0
    last_n = num_anchors
    for n in list(range(int(num_anchors*args.max), args.min - 1, -args.chunk_size)) + [args.min, 0]:
        if n == last_n:
            continue
        n_newly_pruned = last_n - n
        # k = num_anchors - n
        accumulate_gradients(scene, scene.gaussians, opt, pipe, background, anchor_mask)
        anchor_mask, indices = scene.gaussians.gradient_prune(n_newly_pruned, prune_type=args.prune_type,
                                                              mask=anchor_mask)
        anchor_indices[i:i + n_newly_pruned] = indices
        i += n_newly_pruned
        last_n = n
        gaussians.optimizer.zero_grad(set_to_none=True)

    # most important to the least important
    anchor_indices = torch.flip(anchor_indices, [0])
    torch.save(anchor_indices, os.path.join(args.model_path, 'anchor_indices.pt'))

    return anchor_indices


def init_model(dataset, opt, checkpoint, first_iter=30000):
    prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth,
                              dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank,
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist,
                              dataset.add_color_dist)

    scene = Scene(dataset, gaussians, shuffle=False)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    gaussians.load_ply_sparse_gaussian(os.path.join(args.pretrained_path, 'point_cloud', f'iteration_{first_iter}', 'point_cloud.ply'))
    gaussians.load_mlp_checkpoints(os.path.join(args.pretrained_path, 'point_cloud', f'iteration_{first_iter}'))
    gaussians.training_setup(opt)
    return scene


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))


def get_logger():
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    controlshow.setFormatter(formatter)

    logger.addHandler(controlshow)

    return logger


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str)
    parser.add_argument("--gpu", type=str, default = '-1')
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--num_levels", type=int, default=8)
    parser.add_argument("--min", type=int, default=50_000)
    parser.add_argument("--max", type=float, default=1.0)
    parser.add_argument("--chunk_size", type=int, default=1)
    parser.add_argument("--prune_type", type=str, default='gradient_feats')

    args = parser.parse_args(sys.argv[1:])

    lpips_loss = lpips.LPIPS(net='vgg').to('cuda')

    safe_state(args.quiet)

    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)

    logger = get_logger()
    logger.info(f'args: {args}')

    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        os.system("echo $CUDA_VISIBLE_DEVICES")
        logger.info(f'using GPU {args.gpu}')

        
    dataset = args.source_path.split('/')[-1]
    exp_name = f'{args.dataset_name}_{args.scene}'

    scene = init_model(lp.extract(args), op.extract(args), args.start_checkpoint)
    iterative_masking(scene, lp.extract(args), op.extract(args), pp.extract(args))
