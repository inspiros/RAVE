#!/usr/bin/env python3

import copy
import os
import numpy as np

# import subprocess
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
    arr = np.array([i for i in range(L)])
    return ((b ** arr) * tmin).astype(np.int32)


def training(scene, dataset, opt, pipe, dataset_name, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, masks, wandb=None, logger=None):
    first_iter = 30_000
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = scene.gaussians

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, first_iter + opt.iterations), desc="Training progress")
    first_iter += 1

    eval_lpips = False
    dump_images = False
    if args.weighted_sampling:
        progr = np.flip(geometric_progression(1, args.num_levels, len(masks)))
        probabilities = torch.tensor(progr / progr.sum(), dtype=torch.float32)
    else:
        probabilities = torch.tensor([1 for i in range(len(masks))], dtype=torch.float32)

    indices = torch.multinomial(probabilities, opt.iterations, replacement=True)

    for iteration in range(first_iter, first_iter + opt.iterations + 1):

        m_index = indices[iteration-first_iter-1]
        mask = masks[m_index]

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
            

        voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe,background)
        if mask is not None:
            voxel_visible_mask = torch.logical_and(voxel_visible_mask, mask)

        retain_grad = (iteration < opt.update_until and iteration >= 0)
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, visible_mask=voxel_visible_mask, retain_grad=retain_grad)
        
        image = render_pkg["render"]
        gt_image = viewpoint_cam.original_image.cuda()

        Ll1 = l1_loss(image, gt_image)
        ssim_loss = (1.0 - ssim(image, gt_image))
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss
        if args.quantize:
            loss = loss + args.lambda_l1 * torch.abs(gaussians._anchor_feat).mean()
        loss.backward()
        
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            # if iteration in [31_000, 32_500, 35_000, 40_000, 45_000, 50_000, 55_000, 60_000]:
            #     if iteration == 60_000:
            #         eval_lpips = True
            #         dump_images = True
            #     for i, mask in enumerate(masks):
            #         test(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
            #              testing_iterations, scene, render, (pipe, background), wandb, logger, mask, eval_lpips=eval_lpips,
            #              dump_images=dump_images, level=args.num_levels-i-1)
            # Optimizer step
            if iteration < first_iter + opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                
            if (iteration in saving_iterations):
                logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            
            if (iteration in checkpoint_iterations):
                logger.info("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    if args.quantize:
        with torch.no_grad():
            save_paths = []
            for i, mask in enumerate(masks):
                save_path = os.path.join(args.model_path, f'data_{i}.npz')
                save_paths.append(save_path)
                gaussians.save_npz(save_path, mask, args.compress)
            
            for path in save_paths:
                gaussians.load_npz(path)
                size = os.path.getsize(path) / 2**20
                test(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), 
                        testing_iterations, scene, render, (pipe, background), wandb, logger, None, eval_lpips=True,
                        dump_images=dump_images, level=args.num_levels-i-1, size=size)


def accumulate_gradients(scene, gaussians, dataset, opt, pipe, background, mask):
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


@torch.no_grad()
def test_model(args_param, dataset, opt, pipe, dataset_name, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, wandb=None, logger=None):
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, 
                              dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, 
                              dataset.add_color_dist, quantization=args.quantize)
    scene = Scene(dataset, gaussians, shuffle=False)
    gaussians.load_mlp_checkpoints(f'{args.model_path}/point_cloud/iteration_60000/')

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    if args.quantize:
        data = os.listdir(args.model_path)
        to_load = sorted([f'{args.model_path}/{d}' for d in data if 'data_' in d])        
        for data in to_load:
            gaussians.load_npz(data)
            test(tb_writer, dataset_name, 0, None, None, l1_loss, None, testing_iterations, scene, render, (pipe, background), wandb, logger)
    else:
        gaussians.load_ply_sparse_gaussian(f'{args.model_path}/point_cloud/iteration_60000/point_cloud.ply')
        test(tb_writer, dataset_name, 0, None, None, l1_loss, None, testing_iterations, scene, render, (pipe, background), wandb, logger)


def iterative_masking(args_param, dataset, opt, pipe, dataset_name, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, wandb=None, logger=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, 
                              dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, 
                              dataset.add_color_dist, quantization=args.quantize)
    
    scene = Scene(dataset, gaussians, shuffle=False)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt) 
    
    gaussians.load_ply_sparse_gaussian(os.path.join(args.pretrained_path, 'point_cloud', 'iteration_30000', 'point_cloud.ply'))
    gaussians.load_mlp_checkpoints(os.path.join(args.pretrained_path, 'point_cloud', 'iteration_30000'))
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    first_iter += 1

    anchor_mask = None
    anchor_masks = []
    num_anchors = gaussians._anchor.shape[0]
    k_s = list(np.flip((num_anchors - geometric_progression(args.min, num_anchors*args.max, args.num_levels))))
    # test(tb_writer, dataset_name, 0, None, None, l1_loss, None, testing_iterations, scene, render, (pipe, background), wandb, logger, anchor_mask=anchor_mask)

    for i in range(args.num_levels): 
        accumulate_gradients(scene, scene.gaussians, dataset, opt, pipe, background, anchor_mask)
        k = k_s[i]
        anchor_mask, _ = scene.gaussians.gradient_prune(k, prune_type=args.prune_type)
        anchor_masks.append(anchor_mask)
        gaussians.optimizer.zero_grad(set_to_none=True)
        # test(tb_writer, dataset_name, i, None, None, l1_loss, None, testing_iterations, scene, render, (pipe, background), wandb, logger, anchor_mask=anchor_mask)

    torch.save(anchor_masks, os.path.join(args.model_path, 'anchor_masks.pt'))

    return anchor_masks, scene


@torch.no_grad()
def test(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, wandb=None, logger=None, anchor_mask=None, level=None, eval_lpips=False, dump_images=False, size=None):
    
    if dump_images and level is not None:
        os.makedirs(os.path.join(args.model_path, 'renders', f'L{level}'), exist_ok=True)

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
                    img.save(os.path.join(args.model_path, 'renders', f'L{level}', f'{idx}.png'))

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


def get_logger(path):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO) 
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--warmup', action='store_true', default=False)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[3_000, 7_000, 30_000])
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[3_000, 7_000, 30_000])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1000, 5000, 7500, 10_000, 15_000, 20_000, 25_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[60_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[-1])
    parser.add_argument("--start_checkpoint", type=str)
    parser.add_argument("--gpu", type=str, default = '-1')
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--num_levels", type=int, default=8)
    parser.add_argument("--min", type=int, default=50_000)
    parser.add_argument("--max", type=float, default=0.85)
    parser.add_argument("--weighted_sampling", action='store_true', default=True)
    parser.add_argument("--G", type=int, default=4)
    parser.add_argument("--lambda_l1", type=float, default=0)
    parser.add_argument("--compress", action='store_true', default=False)
    parser.add_argument("--quantize", action='store_true', default=False)
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--prune_type", type=str, default='gradient_feats')

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    lpips_loss = lpips.LPIPS(net='vgg').to('cuda')

    safe_state(args.quiet)

    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)

    logger = get_logger(model_path)
    logger.info(f'args: {args}')

    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        os.system("echo $CUDA_VISIBLE_DEVICES")
        logger.info(f'using GPU {args.gpu}')

        
    dataset = args.source_path.split('/')[-1]
    exp_name = f'{args.dataset_name}_{args.scene}'
    
    if args.use_wandb:
        wandb.login()
        run = wandb.init(
            # Set the project where this run will be logged
            project=f"Scaffold-GS-{dataset}",
            name=exp_name,
            # Track hyperparameters and run metadata
            settings=wandb.Settings(start_method="fork"),
            config=vars(args)
        )
    else:
        wandb = None

    if args.test_only:
        test_model(args, lp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, 
            args.debug_from, wandb, logger)
    else:
        # training
        masks, scene = iterative_masking(args, lp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, 
                args.debug_from, wandb, logger)
        training(scene, lp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint,
                args.debug_from, masks, wandb, logger)

    # All done
    logger.info("\nTraining complete.")
