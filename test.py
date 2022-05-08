# encoding: utf-8
# Author: Bingxin Ke
# Created: 2021/10/6
"""
    Evaluation pipeline, adapted from train.py
    Require additional field in configuration file: test/check_point, indicating the path to trained model
        usage example: python test.py config/train_test/ImpliCity-0.yaml
"""

import argparse
import logging
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import os

from datetime import datetime, timedelta

import matplotlib
import numpy as np
import torch

import wandb

from torch.utils.data import DataLoader

from src.utils.libconfig import config
from src.DSMEvaluation import DSMEvaluator, print_statistics
from src.io.checkpoints import CheckpointIO, DEFAULT_MODEL_FILE
from src.dataset import ImpliCityDataset
from src.utils.libpc.PCTransforms import PointCloudSubsampler
from src.utils.libconfig import lock_seed
from src.model import get_model
from src.generation import DSMGenerator

from src.utils.libconfig import config_logging


matplotlib.use('Agg')

# clear environment variable for rasterio
if os.environ.get('PROJ_LIB') is not None:
    del os.environ['PROJ_LIB']

# Set t0
# t0 = time.time()
t_start = datetime.now()

# Arguments
parser = argparse.ArgumentParser(
    description='Train a 3D reconstruction model.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
# parser.add_argument('--no-wandb', action='store_true', help='run without wandb')
parser.add_argument('--exit-after', type=int, default=-1,
                    help='Checkpoint and exit after specified number of seconds'
                         'with exit code 3.')

args = parser.parse_args()
exit_after = args.exit_after
if not (os.path.exists(args.config) and os.path.isfile(args.config)):
    raise IOError(f"config file not exist: '{args.config}'")
cfg = config.load_config(args.config, None)

# shorthands
cfg_dataset = cfg['dataset']
cfg_loader = cfg['dataloader']
cfg_model = cfg['model']
cfg_training = cfg['training']
cfg_test = cfg['test']
# cfg_mesh = cfg['mesh_generation']
cfg_dsm = cfg['dsm_generation']

cfg_multi_class = cfg_model.get('multi_label', False)

batch_size = cfg_training['batch_size']
val_batch_size = cfg_training['val_batch_size']

learning_rate = cfg_training['learning_rate']
model_selection_metric = cfg_training['model_selection_metric']

# Output directory
out_dir = cfg_training['out_dir']
pure_run_name = cfg_training['run_name'] + '_test'
run_name = f"{t_start.strftime('%y_%m_%d-%H_%M_%S')}-{pure_run_name}"
out_dir_run = os.path.join(out_dir, run_name)
out_dir_tiff = os.path.join(out_dir_run, "tiff")
if not os.path.exists(out_dir_run):
    os.makedirs(out_dir_run)
if not os.path.exists(out_dir_tiff):
    os.makedirs(out_dir_tiff)

if cfg_training['lock_seed']:
    lock_seed(0)

# %% -------------------- config logging --------------------
config_logging(cfg['logging'], out_dir_run)
print(f"{'*' * 30} Start {'*' * 30}")

# %% save config file
_output_path = os.path.join(out_dir_run, "config.yaml")
with open(_output_path, 'w+') as f:
    yaml.dump(cfg, f, default_flow_style=None, allow_unicode=True, Dumper=Dumper)
logging.info(f"Config saved to {_output_path}")

# %% -------------------- disable wandb --------------------
wandb.init(mode='disabled')

# %% -------------------- Device --------------------
cuda_avail = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if cuda_avail else "cpu")

logging.info(f"Device: {device}")

# torch.cuda.synchronize(device)

# %% -------------------- Data --------------------

test_dataset = ImpliCityDataset('test', cfg_dataset=cfg_dataset, merge_query_occ=not cfg_multi_class,
                                random_sample=False,
                                flip_augm=False, rotate_augm=False)

n_workers = cfg_loader['n_workers']

# visualization dataloader
vis_loader = DataLoader(test_dataset, batch_size=1, num_workers=n_workers, shuffle=False)

logging.info(f"dataset path: '{cfg_dataset['path']}'")


# %% -------------------- Model --------------------
model = get_model(cfg, device)

wandb.watch(model)

# %% -------------------- Generator: generate DSM --------------------

generator_dsm = DSMGenerator(model=model, device=device, data_loader=vis_loader,
                             fill_empty=cfg_dsm['fill_empty'],
                             dsm_pixel_size=cfg_dsm['pixel_size'],
                             h_range=cfg_dsm['h_range'],
                             h_res_0=cfg_dsm['h_resolution_0'],
                             upsample_steps=cfg_dsm['h_upsampling_steps'],
                             half_blend_percent=cfg_dsm.get('half_blend_percent', None),
                             crs_epsg=cfg_dsm.get('crs_epsg', 32632))

gt_dsm_path = cfg_dataset['dsm_gt_path']
gt_mask_path = cfg_dataset['mask_files']['gt']
land_mask_path_dict = {}
for _mask_type in ['building', 'forest', 'water']:
    if cfg_dataset['mask_files'][_mask_type] is not None:
        land_mask_path_dict.update({_mask_type: cfg_dataset['mask_files'][_mask_type]})

evaluator = DSMEvaluator(gt_dsm_path, gt_mask_path, land_mask_path_dict)

# %% -------------------- Initialization --------------------
# Load checkpoint
checkpoint_io = CheckpointIO(out_dir_run, model=model, optimizer=None, scheduler=None)
# resume_from = cfg_training.get('resume_from', None)
resume_from = cfg_test.get('check_point', None)
resume_scheduler = False
try:
    _resume_from_file = resume_from if resume_from is not None else os.path.join(out_dir_run, DEFAULT_MODEL_FILE)
    logging.info(f"resume: {_resume_from_file}")
    # print(os.path.exists(_resume_from_file))
    load_dict = checkpoint_io.load(_resume_from_file, resume_scheduler=resume_scheduler)
    logging.info(f"Checkpoint loaded: '{_resume_from_file}'")
except FileExistsError:
    load_dict = dict()
    logging.info(f"Check point does NOT exist, can not inference.")
    exit()

# n_epoch = load_dict.get('n_epoch', 0)  # epoch numbers
n_iter = load_dict.get('n_iter', 0)  # total iterations
_last_train_seconds = load_dict.get('training_time', 0)
last_training_time = timedelta(seconds=_last_train_seconds)

if cfg['training']['model_selection_mode'] == 'maximize':
    model_selection_sign = 1  # metric * sign => larger is better
elif cfg['training']['model_selection_mode'] == 'minimize':
    model_selection_sign = -1  # metric * sign => larger is better
else:
    _msg = 'model_selection_mode must be either maximize or minimize.'
    logging.error(_msg)
    raise ValueError(_msg)
metric_val_best = load_dict.get('loss_val_best', -model_selection_sign * np.inf)
logging.info(f"Current best validation metric = {metric_val_best:.8f}")

# %% -------------------- Inference --------------------
n_parameters = sum(p.numel() for p in model.parameters())
logging.info(f"Total number of parameters = {n_parameters}")
logging.info(f"output path: '{out_dir_run}'")


def visualize():
    _output_path = os.path.join(out_dir_tiff, f"{pure_run_name}_dsm_{n_iter:06d}.tiff")
    dsm_writer = generator_dsm.generate_dsm(_output_path)
    logging.info(f"DSM saved to '{_output_path}'")
    _target_dsm = dsm_writer.get_data()
    # evaluate dsm
    output_dic, diff_arr = evaluator.eval(_target_dsm, dsm_writer.T)
    # wandb_dic = {f"test/{k}": v for k, v in output_dic['overall'].items()}
    _output_path = os.path.join(out_dir_tiff, f"{pure_run_name}_dsm_{n_iter:06d}_eval.txt")
    str_stat = print_statistics(output_dic, f"{pure_run_name}-iter{n_iter}", save_to=_output_path)
    logging.info(f"DSM evaluation saved to '{_output_path}")
    # residual
    dsm_writer.set_data(diff_arr)
    _output_path = os.path.join(out_dir_tiff, f"{pure_run_name}_residual_{n_iter:06d}.tiff")
    dsm_writer.write_to_file(_output_path)
    logging.info(f"DSM residual saved to '{_output_path}")
    _dsm_log_dic = {f'DSM/{k}/{k2}': v2 for k, v in output_dic.items() for k2, v2 in v.items()}
    # wandb.log(_dsm_log_dic, step=n_iter)


try:
    visualize()
except IOError as e:
    logging.error("Error: " + e.__str__())
