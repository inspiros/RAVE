#!/usr/bin/env python3

import copy
import os

import numpy as np

# cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
# result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
# os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

os.system('echo $CUDA_VISIBLE_DEVICES')

import torch
import wandb
import time
from PIL import Image
import lpips
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import prefilter_voxel, render, generate_neural_gaussians
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
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


def num_primitives_from_rate(r, tmin, tmax, L):
    b = np.exp((np.log(tmax) - np.log(tmin)) / (L - 1))
    arr = r * (L - 1)
    return ((b ** arr) * tmin).astype(np.int32)


def accumulate_gradients(scene, gaussians, opt, pipe, background, mask):
    for viewpoint_cam in scene.getTrainCameras():
        voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background)
        if mask is not None:
            voxel_visible_mask = torch.logical_and(voxel_visible_mask, mask)

        render_pkg = render(viewpoint_cam, gaussians, pipe, background, visible_mask=voxel_visible_mask,
                            retain_grad=True)
        image = render_pkg["render"]

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_loss = (1.0 - ssim(image, gt_image))
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss

        loss.backward(retain_graph=True)


@torch.no_grad()
def accumulate_opacities(scene, gaussians, mask=None, reduction='mean'):
    if reduction not in ['mean', 'sum']:
        raise ValueError('reduction must be either "mean" or "sum"')
    if mask is None:
        mask = slice(None)
    accum = torch.zeros(gaussians.get_anchor[mask].shape[0], device='cuda')
    for viewpoint_cam in scene.getTrainCameras():
        xyz, color, opacity, neural_scaling, neural_rot, neural_opacity, out_mask = \
            generate_neural_gaussians(viewpoint_cam, gaussians, visible_mask=mask, is_training=True)
        neural_opacity = neural_opacity.view(-1, gaussians.n_offsets)
        neural_opacity = neural_opacity.mean(1) if reduction == 'mean' else neural_opacity.sum(1)
        accum += neural_opacity.view_as(accum)
    if reduction == 'mean':
        accum /= len(scene.getTrainCameras())
    return accum


@torch.no_grad()
def accumulate_scaling(scene, gaussians, mask=None, reduction='mean'):
    if reduction not in ['mean', 'sum']:
        raise ValueError('reduction must be either "mean" or "sum"')
    if mask is None:
        mask = slice(None)
    accum = torch.zeros(gaussians.get_anchor[mask].shape[0], device='cuda')
    for viewpoint_cam in scene.getTrainCameras():
        xyz, color, opacity, neural_scaling, neural_rot, neural_opacity, out_mask = \
            generate_neural_gaussians(viewpoint_cam, gaussians, visible_mask=mask, is_training=True,
                                      ignore_opacity_mask=True)
        neural_scaling = neural_scaling.view(-1, gaussians.n_offsets, 3)
        neural_scaling = neural_scaling.mean([1, 2]) if reduction == 'mean' else neural_scaling.sum([1, 2])
        accum += neural_scaling.view_as(accum)
    if reduction == 'mean':
        accum /= len(scene.getTrainCameras())
    return accum


@torch.no_grad()
def test(tb_writer, dataset_name, iteration, scene: Scene, renderFunc, renderArgs, wandb=None, logger=None,
         anchor_mask=None, rate=None, eval_lpips=False, dump_images=False, size=None):
    if dump_images and rate is not None:
        os.makedirs(os.path.join(args.model_path, 'renders', f'r={rate:.03f}'), exist_ok=True)

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
    validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                          {'name': 'train',
                           'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                       range(5, 30, 5)]})

    for config in validation_configs:
        if config['cameras'] and len(config['cameras']) > 0:
            l1_test = 0.0
            psnr_test = 0.0
            ssim_test = 0.0
            lpips_test = 0.0
            ms_per_render = []
            num_gaussians_list = []

            for idx, viewpoint in enumerate(config['cameras']):

                torch.cuda.synchronize();
                t_start = time.time()
                voxel_visible_mask = prefilter_voxel(viewpoint, scene_cpy.gaussians, *renderArgs)
                pkg = renderFunc(viewpoint, scene_cpy.gaussians, *renderArgs, visible_mask=voxel_visible_mask)
                torch.cuda.synchronize();
                t_end = time.time()
                ms = t_end - t_start
                ms_per_render.append(ms)
                image = torch.clamp(pkg["render"], 0.0, 1.0)

                ms_per_render.append(ms)
                num_gaussians_list.append(pkg['num_gaussians'])
                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                if dump_images and config['name'] == 'test':
                    np_img = (np.transpose(image.cpu().numpy(), (1, 2, 0)) * 255).astype(np.uint8)
                    img = Image.fromarray(np_img)
                    img.save(os.path.join(args.model_path, 'renders', f'r={rate:.03f}', f'{idx}.png'))

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
            logger.info(
                "\n[ITER {}] Evaluating {}: L1 {} PSNR {} FPS {} #A {} #W {}".format(iteration, config['name'], l1_test,
                                                                                     psnr_test, fps, num_anchor,
                                                                                     num_gaussians))

            if config['name'] == 'test':
                write(
                    f'{iteration}, {psnr_test}, {ssim_test}, {lpips_test}, {num_anchor}, {fps}, {num_gaussians}, {size}')

            if tb_writer:
                tb_writer.add_scalar(f'{dataset_name}/' + config['name'] + '/loss_viewpoint - l1_loss', l1_test,
                                     iteration)
                tb_writer.add_scalar(f'{dataset_name}/' + config['name'] + '/loss_viewpoint - psnr', psnr_test,
                                     iteration)
            if wandb is not None:
                wandb.log({f"{config['name']}_loss_viewpoint_l1_loss": l1_test, f"{config['name']}_PSNR": psnr_test})

    if tb_writer:
        # tb_writer.add_histogram(f'{dataset_name}/'+"scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        tb_writer.add_scalar(f'{dataset_name}/' + 'total_points', scene_cpy.gaussians.get_anchor.shape[0], iteration)

    torch.cuda.empty_cache()
    scene_cpy.gaussians.train()


def init_model(dataset, opt, checkpoint, first_iter=60000):
    prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth,
                              dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank,
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist,
                              dataset.add_color_dist, quantization=args.quantize)

    scene = Scene(dataset, gaussians, shuffle=False)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    gaussians.load_ply_sparse_gaussian(
        os.path.join(args.pretrained_path, 'point_cloud', f'iteration_{first_iter}', 'point_cloud.ply'))
    gaussians.load_mlp_checkpoints(os.path.join(args.pretrained_path, 'point_cloud', f'iteration_{first_iter}'))
    gaussians.training_setup(opt)

    anchor_masks = torch.load(os.path.join(args.pretrained_path, 'anchor_masks.pt'))

    return scene, anchor_masks


def interp_and_save(scene, dataset, opt, pipe, dataset_name, iteration, masks):
    gaussians = scene.gaussians

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    interp_masks = {}
    num_anchors = gaussians._anchor.shape[0]
    k_s = np.flip((num_anchors - geometric_progression(args.min, num_anchors * args.max, args.num_levels)))
    gaussians.load_npz(os.path.join(args.pretrained_path, 'data_0.npz'))
    hyper_mask = masks[0].clone()

    # save_paths = []
    # for i in range(args.num_levels):
    #     level = args.num_levels - i - 1
    #     interp_mask = masks[i][hyper_mask]
    #     save_path = os.path.join(args.model_path, f'data_L{level}.npz')
    #     save_paths.append(save_path)
    #     gaussians.save_npz(save_path, interp_mask, args.compress)
    #
    # checkpoint_path = os.path.join(args.model_path, 'checkpoints.pth')
    # gaussians.save_mlp_checkpoints(checkpoint_path)
    #
    # torch.save(masks, os.path.join(args.model_path, 'interp_masks.pt'))
    #
    # mlp_size = os.path.getsize(checkpoint_path)
    # for i, path in enumerate(save_paths):
    #     level = args.num_levels - i - 1
    #     gaussians.load_npz(path)
    #     # no need to load checkpoints here as there is only one model
    #     size = (os.path.getsize(path) + mlp_size) / 2 ** 20
    #     test(None, dataset_name, iteration, scene, render, (pipe, background), wandb, logger, None,
    #          eval_lpips=True, dump_images=True, rate=level, size=size)
    # exit()
    save_paths = {}
    last_coarser_level = -1
    prune_criteria = None
    for i, r in enumerate(torch.linspace(0, 1, args.num_test_levels, dtype=torch.float32).tolist()):
        k = num_anchors - num_primitives_from_rate(r, args.min, num_anchors * args.max, args.num_levels)
        need_interp = True
        for j in range(args.num_levels):
            if k_s[j] >= k:
                if k_s[j] == k:
                    need_interp = False
                coarser_level = j
                break
        coarser_mask = masks[coarser_level]
        print(f'rate={r:.03f}, need_interp={need_interp}, last_n={coarser_mask.sum().item()}')
        if need_interp:
            finer_mask = masks[coarser_level - 1]
            context_mask = torch.logical_xor(finer_mask, coarser_mask) # if not args.global_ranking else coarser_mask
            need_recompute_ranking = False
            if coarser_level != last_coarser_level:
                need_recompute_ranking = True
                last_coarser_level = coarser_level
            if need_recompute_ranking:
                if args.prune_type.startswith('gradient'):
                    if not args.global_ranking or gaussians._anchor.grad is None:
                        gaussians.optimizer.zero_grad(set_to_none=True)
                        accumulate_gradients(scene, scene.gaussians, opt, pipe, background,
                                             finer_mask[hyper_mask] if not args.global_ranking else None)
                    prune_criteria = scene.gaussians.gradient_prune_criteria(
                        prune_type=args.prune_type,
                        mask=context_mask[hyper_mask])
                elif args.prune_type == 'opacity':
                    prune_criteria = accumulate_opacities(
                        scene, scene.gaussians,
                        mask=context_mask[hyper_mask])
                elif args.prune_type == 'scaling':
                    prune_criteria = accumulate_scaling(
                        scene, scene.gaussians,
                        mask=context_mask[hyper_mask])
                elif args.prune_type == 'random':
                    prune_criteria = torch.randn(context_mask[hyper_mask].sum().item(),
                                                 device=hyper_mask.device)
            m, _ = scene.gaussians.prune(
                k - k_s[coarser_level - 1], criteria=prune_criteria,
                mask=context_mask[hyper_mask],
                largest=False)  # True for scaling
            # m, _ = scene.gaussians.gradient_prune(k - k_s[coarser_level - 1], prune_type=args.prune_type,
            #                                       mask=context_mask[hyper_mask])
            interp_mask = hyper_mask.clone()
            interp_mask.scatter_(0, torch.where(hyper_mask)[0], m)
            interp_mask.logical_or_(coarser_mask)
            assert (torch.logical_or(finer_mask, interp_mask).eq(finer_mask).all().item() and
                    torch.logical_or(interp_mask, coarser_mask).eq(interp_mask).all().item())
        else:
            interp_mask = coarser_mask
        interp_masks[r] = interp_mask

        save_path = os.path.join(args.model_path, f'data_r={r:.03f}.npz')
        save_paths[r] = save_path
        gaussians.save_npz(save_path, interp_mask[hyper_mask], args.compress)
    gaussians.optimizer.zero_grad(set_to_none=True)

    checkpoint_path = os.path.join(args.model_path, 'checkpoints.pth')
    gaussians.save_mlp_checkpoints(checkpoint_path)

    torch.save(interp_masks, os.path.join(args.model_path, 'interp_masks.pt'))

    mlp_size = os.path.getsize(checkpoint_path)
    for r, path in save_paths.items():
        gaussians.load_npz(path)
        # no need to load checkpoints here as there is only one model
        size = (os.path.getsize(path) + mlp_size) / 2 ** 20
        test(None, dataset_name, iteration, scene, render, (pipe, background), wandb, logger, None,
             eval_lpips=True, dump_images=True, rate=r, size=size)


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
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
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[-1])
    parser.add_argument("--start_checkpoint", type=str)
    parser.add_argument("--gpu", type=str, default='-1')
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--num_levels", type=int, default=8)
    parser.add_argument("--num_test_levels", type=int, default=50)
    parser.add_argument("--min", type=int, default=50_000)
    parser.add_argument("--max", type=float, default=0.85)
    parser.add_argument("--weighted_sampling", action='store_true', default=True)
    parser.add_argument("--G", type=int, default=4)
    parser.add_argument("--lambda_l1", type=float, default=0)
    parser.add_argument("--compress", action='store_true', default=False)
    parser.add_argument("--quantize", action='store_true', default=False)
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--prune_type", type=str, default='gradient_feats')
    parser.add_argument("--global_ranking", action='store_true', default=False)

    args = parser.parse_args(sys.argv[1:])

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

    scene, masks = init_model(lp.extract(args), op.extract(args), args.start_checkpoint)
    interp_and_save(scene, lp.extract(args), op.extract(args), pp.extract(args), dataset, 'final', masks)

    # All done
    logger.info("\nAll complete.")
