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
import torchvision
import json
import wandb
import time
from os import makedirs
import shutil, pathlib
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
# from lpipsPyTorch import lpips
import lpips
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import prefilter_voxel, render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")
    
run_codec = True
    
def write(string):
    with open(os.path.join(args.model_path, 'results.csv'), 'a') as f:
        print(string, file=f)
        
        
def geometric_progression(tmin, tmax, L):
    b = np.exp((np.log(tmax) - np.log(tmin)) / (L - 1))
    print(b)
    arr = np.array([i for i in range(L)])
    return ((b ** arr) * tmin).astype(np.int32)


def test_baseline(dataset, opt, pipe, dataset_name, checkpoint, wandb=None, logger=None):
    first_iter = 30_000

    # init scene
    gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth,
                              dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank,
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist,
                              dataset.add_color_dist)

    scene = Scene(dataset, gaussians, shuffle=False)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    ply_path = os.path.join(args.model_path, 'point_cloud', 'iteration_30000', 'point_cloud.ply')
    checkpoint_path = os.path.join(args.model_path, 'point_cloud', 'iteration_30000', 'checkpoints.pth')
    gaussians.load_ply_sparse_gaussian(ply_path)
    gaussians.load_mlp_checkpoints(os.path.join(args.model_path, 'point_cloud', 'iteration_30000'))
    gaussians.training_setup(opt)

    gaussians = scene.gaussians

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None

    eval_lpips = True
    dump_images = True

    iter_start.record()

    gaussians.update_learning_rate(first_iter)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Pick a random Camera
    if not viewpoint_stack:
        viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

    # Render
    voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background)

    render_pkg = render(viewpoint_cam, gaussians, pipe, background, visible_mask=voxel_visible_mask,
                        retain_grad=False)

    image = render_pkg["render"]
    gt_image = viewpoint_cam.original_image.cuda()

    Ll1 = l1_loss(image, gt_image)
    ssim_loss = (1.0 - ssim(image, gt_image))
    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss
    loss.backward()

    iter_end.record()

    gaussians.optimizer.zero_grad(set_to_none=True)

    # save_path = os.path.join(args.model_path, 'data.npz')
    # gaussians.save_npz(save_path, None, False)
    # gaussians.load_npz(save_path)
    size = (os.path.getsize(ply_path) + os.path.getsize(checkpoint_path)) / 2**20
    test(None, dataset_name, first_iter, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
         scene, render, (pipe, background), wandb, logger, None, eval_lpips=eval_lpips,
         dump_images=dump_images, size=size)


@torch.no_grad()
def test_model(args_param, dataset, opt, pipe, dataset_name, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, wandb=None, logger=None):
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, 
                              dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, 
                              dataset.add_color_dist)
    scene = Scene(dataset, gaussians, shuffle=False)
    gaussians.load_mlp_checkpoints(f'{args.model_path}/point_cloud/iteration_30000/')

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    if args.quantize:
        data = os.listdir(args.model_path)
        to_load = sorted([f'{args.model_path}/{d}' for d in data if 'data_' in d])        
        for data in to_load:
            gaussians.load_npz(data)
            test(tb_writer, dataset_name, 0, None, None, l1_loss, None, scene, render, (pipe, background), wandb, logger)
    else:
        gaussians.load_ply_sparse_gaussian(f'{args.model_path}/point_cloud/iteration_60000/point_cloud.ply')
        test(tb_writer, dataset_name, 0, None, None, l1_loss, None, scene, render, (pipe, background), wandb, logger)


@torch.no_grad()
def test(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, elapsed, scene : Scene, renderFunc, renderArgs, wandb=None, logger=None, anchor_mask=None, eval_lpips=False, dump_images=False, size=None):
    
    if dump_images:
        os.makedirs(os.path.join(args.model_path, 'renders'), exist_ok=True)

    if anchor_mask is not None:
        scene_cpy = copy.deepcopy(scene)
        scene_cpy.gaussians._anchor = torch.nn.Parameter(scene_cpy.gaussians._anchor[anchor_mask])
        scene_cpy.gaussians._anchor_feat = torch.nn.Parameter(scene_cpy.gaussians._anchor_feat[anchor_mask])
        scene_cpy.gaussians._offset = torch.nn.Parameter(scene_cpy.gaussians._offset[anchor_mask])
        scene_cpy.gaussians._scaling = torch.nn.Parameter(scene_cpy.gaussians._scaling[anchor_mask])        
    else:
        scene_cpy = scene
    
    scene_cpy.gaussians.eval()
    torch.cuda.empty_cache()
    validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                            {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

    for config in validation_configs:
        if config['cameras'] and len(config['cameras']) > 0:
            l1_test = 0.0
            psnr_test = 0.0
            ssim_test = 0.0
            lpips_test = 0.0
            ms_per_render = []
            num_gaussians_list = []

            for idx, viewpoint in enumerate(config['cameras']):

                torch.cuda.synchronize();t_start = time.time()
                voxel_visible_mask = prefilter_voxel(viewpoint, scene_cpy.gaussians, *renderArgs)
                pkg = renderFunc(viewpoint, scene_cpy.gaussians, *renderArgs, visible_mask=voxel_visible_mask)
                torch.cuda.synchronize();t_end = time.time()
                ms = t_end - t_start
                ms_per_render.append(ms)
                image = torch.clamp(pkg["render"], 0.0, 1.0)
 
                ms_per_render.append(ms)
                num_gaussians_list.append(pkg['num_gaussians'])
                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                if dump_images and config['name'] == 'test':
                    np_img = (np.transpose(image.cpu().numpy(), (1, 2, 0)) * 255).astype(np.uint8)
                    img = Image.fromarray(np_img)
                    img.save(os.path.join(args.model_path, 'renders', f'{idx}.png'))

                l1_test += l1_loss(image, gt_image).mean().double()
                psnr_test += psnr(image, gt_image).mean().double()
                ssim_test += ssim(image, gt_image).mean().double()
                if eval_lpips:
                    lpips_test += lpips_loss(image, gt_image, normalize=True).item()
                fps = ((1.0 / np.array(ms_per_render))[1:]).mean()
                
            num_gaussians = np.array(num_gaussians_list).mean()
            psnr_test /= len(config['cameras'])
            ssim_test /= len(config['cameras'])
            
            if lpips_test != 0:
                lpips_test /= len(config['cameras'])
                
            l1_test /= len(config['cameras'])          
            num_anchor = scene_cpy.gaussians._anchor.shape[0] if anchor_mask is None else anchor_mask.sum()
            logger.info("\n[ITER {}] Evaluating {}: L1 {} PSNR {} FPS {} #A {} #W {}".format(iteration, config['name'], l1_test, psnr_test, fps, num_anchor, num_gaussians))
            
            if config['name'] == 'test':
                write(f'{iteration}, {psnr_test}, {ssim_test}, {lpips_test}, {num_anchor}, {fps}, {num_gaussians}, {size}')
            
            if tb_writer:
                tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
            if wandb is not None:
                wandb.log({f"{config['name']}_loss_viewpoint_l1_loss":l1_test, f"{config['name']}_PSNR":psnr_test})

    if tb_writer:
        # tb_writer.add_histogram(f'{dataset_name}/'+"scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        tb_writer.add_scalar(f'{dataset_name}/'+'total_points', scene_cpy.gaussians.get_anchor.shape[0], iteration)
    
    torch.cuda.empty_cache()
    scene_cpy.gaussians.train()


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

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


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
    parser.add_argument("--start_checkpoint", type=str)
    parser.add_argument("--gpu", type=str, default = '-1')
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--scene", type=str, default=None)

    args = parser.parse_args(sys.argv[1:])

    lpips_loss = lpips.LPIPS(net='vgg').to('cuda')

    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)

    logger = get_logger()
    logger.info(f'args: {args}')

    dataset = args.source_path.split('/')[-1]
    exp_name = f'{args.dataset_name}_{args.scene}'

    test_baseline(lp.extract(args), op.extract(args), pp.extract(args), dataset, args.start_checkpoint,
                  None, logger)
