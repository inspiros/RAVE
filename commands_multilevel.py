from argparse import ArgumentParser
import sys
import os
import subprocess


data_dir='/scratch/nerf/dataset'
exp_dir='./output/multilevel'

parser = ArgumentParser(description="Training script parameters")
parser.add_argument("--num_levels", type=int, default=8)
parser.add_argument("--min", type=int, default=50_000)
parser.add_argument("--max", type=float, default=0.85)
parser.add_argument("--weighted_sampling", action='store_true', default=False)
parser.add_argument("--G", type=int, default=5) # irrelevant if weighted_sampling is False
parser.add_argument("--quantize", action='store_true', default=True)
parser.add_argument("--compress", action='store_true', default=True)
parser.add_argument("--lambda_l1", type=float, default=0)

args = parser.parse_args(sys.argv[1:])

mipnerf360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"]
mipnerf360_outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]
tanks_and_temples_scenes = ["truck", "train"]
deep_blending_scenes = ["drjohnson", "playroom"]

'''nerf360 data'''
voxel_size=0.001
update_init_factor=16
appearance_dim=0
ratio=1

'''tandt data'''
voxel_size=0.01

'''db data'''
voxel_size=0.005

weighted_sampling = args.weighted_sampling

command = 'python3 scalable_train.py --eval --source_path {data_dir}/{dataset_name}/{scene} ' + \
        '--pretrained_path ./output/baseline/{dataset_name}/{scene} ' +\
        '--lod {lod} --voxel_size {vsize} --appearance_dim {appearance_dim} --ratio {ratio} ' + \
        '-m {exp_dir}/{dataset_name}/{scene}/L={num_levels}_min={min}_max={max}_compress={compress} ' + \
        '--G {G} --num_levels {num_levels} --min {min} --max {max}'
        
if args.compress:
    command += ' --compress'
if args.quantize:
    command += ' --quantize'

command += f' --lambda_l1 {args.lambda_l1}'


with open('commands.txt', 'a') as f:
    if args.weighted_sampling:
        command += ' --weighted_sampling'


    for scene in mipnerf360_indoor_scenes:
        
        formatted = command.format(data_dir=data_dir, exp_dir=exp_dir, dataset_name='nerf_real_360', scene=scene, lod=0, vsize=0.001, appearance_dim=0, 
                    ratio=1, min=args.min, max=args.max, G=args.G, num_levels=args.num_levels, weighted_sampling=weighted_sampling, compress=args.compress, lambda_l1=args.lambda_l1)
        #os.system(formatted)
        print(formatted, file=f)

    for scene in mipnerf360_outdoor_scenes:
            
        formatted = command.format(data_dir=data_dir, exp_dir=exp_dir, dataset_name='nerf_real_360', scene=scene, lod=0, vsize=0.001, appearance_dim=0, 
                    ratio=1, min=args.min, max=args.max, G=args.G, num_levels=args.num_levels, weighted_sampling=weighted_sampling, compress=args.compress, lambda_l1=args.lambda_l1)
        #os.system(formatted)
        print(formatted, file=f)
        
    for scene in tanks_and_temples_scenes:
            
        formatted = command.format(data_dir=data_dir, exp_dir=exp_dir, dataset_name='tandt', scene=scene, lod=0, vsize=0.01, appearance_dim=0, 
                    ratio=1, min=args.min, max=args.max, G=args.G, num_levels=args.num_levels, weighted_sampling=weighted_sampling, compress=args.compress, lambda_l1=args.lambda_l1)
        #os.system(formatted)
        print(formatted, file=f)
        
    for scene in deep_blending_scenes:
            
        formatted = command.format(data_dir=data_dir, exp_dir=exp_dir, dataset_name='db', scene=scene, lod=0, vsize=0.005, appearance_dim=0, 
                    ratio=1, min=args.min, max=args.max, G=args.G, num_levels=args.num_levels, weighted_sampling=weighted_sampling, compress=args.compress, lambda_l1=args.lambda_l1)
        #os.system(formatted)
        print(formatted, file=f)

