# encoding: utf-8
# Author: Bingxin Ke
# Created: 2021/10/6
"""
    Training pipeline
        usage example: python train.py config/train_test/ImpliCity-0.yaml
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
import torch.optim as optim
import wandb
from torch.optim.lr_scheduler import MultiStepLR, CyclicLR
from torch.utils.data import DataLoader

from src.utils.libconfig import config
from src.DSMEvaluation import DSMEvaluator, print_statistics
from src.io.checkpoints import CheckpointIO, DEFAULT_MODEL_FILE
from src.dataset import ImpliCityDataset

from src.utils.libconfig import lock_seed
from src.model import get_model
from src.generation import DSMGenerator
from src.Trainer import Trainer
from src.utils.libconfig import config_logging
from src.loss import wrapped_bce, wrapped_cross_entropy

# -------------------- Initialization --------------------
matplotlib.use('Agg')

# clear environment variable for rasterio
if os.environ.get('PROJ_LIB') is not None:
    del os.environ['PROJ_LIB']

# Set t0
# t0 = time.time()
t_start = datetime.now()

# -------------------- Arguments --------------------
parser = argparse.ArgumentParser(
    description='Train a 3D reconstruction model.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--no-wandb', action='store_true', help='run without wandb')
parser.add_argument('--exit-after', type=int, default=-1,
                    help='Checkpoint and exit after specified number of seconds'
                         'with exit code 3.')

args = parser.parse_args()
exit_after = args.exit_after
if not (os.path.exists(args.config) and os.path.isfile(args.config)):
    raise IOError(f"config file not exist: '{args.config}'")
cfg = config.load_config(args.config, None)

# -------------------- shorthands --------------------
cfg_dataset = cfg['dataset']
cfg_loader = cfg['dataloader']
cfg_model = cfg['model']
cfg_training = cfg['training']
cfg_test = cfg['test']
cfg_dsm = cfg['dsm_generation']
cfg_multi_class = cfg_model.get('multi_label', False)

batch_size = cfg_training['batch_size']
val_batch_size = cfg_training['val_batch_size']

learning_rate = cfg_training['learning_rate']
model_selection_metric = cfg_training['model_selection_metric']

print_every = cfg_training['print_every']
visualize_every = cfg_training['visualize_every']
validate_every = cfg_training['validate_every']
checkpoint_every = cfg_training['checkpoint_every']
backup_every = cfg_training['backup_every']

# -------------------- Output directory --------------------
out_dir = cfg_training['out_dir']
pure_run_name = cfg_training['run_name']
run_name = f"{t_start.strftime('%y_%m_%d-%H_%M_%S')}-{pure_run_name}"
out_dir_run = os.path.join(out_dir, run_name)
out_dir_ckpt = os.path.join(out_dir_run, "check_points")
out_dir_tiff = os.path.join(out_dir_run, "tiff")
if not os.path.exists(out_dir_run):
    os.makedirs(out_dir_run)
if not os.path.exists(out_dir_ckpt):
    os.makedirs(out_dir_ckpt)
if not os.path.exists(out_dir_tiff):
    os.makedirs(out_dir_tiff)

if cfg_training['lock_seed']:
    lock_seed(0)

# %% -------------------- config logging --------------------
config_logging(cfg['logging'], out_dir_run)
print(f"{'*' * 30} Start {'*' * 30}")

# %% -------------------- save config file --------------------
_output_path = os.path.join(out_dir_run, "config.yaml")
with open(_output_path, 'w+') as f:
    yaml.dump(cfg, f, default_flow_style=None, allow_unicode=True, Dumper=Dumper)
logging.info(f"Config saved to {_output_path}")

# %% -------------------- Config wandb --------------------
_wandb_out_dir = os.path.join(out_dir_run, "wandb")
if not os.path.exists(_wandb_out_dir):
    os.makedirs(_wandb_out_dir)
if args.no_wandb:
    wandb.init(mode='disabled')
else:
    wandb.init(project='PROJECT_NAME',
               config=cfg,
               name=os.path.basename(out_dir_run),
               dir=_wandb_out_dir,
               mode='online',
               settings=wandb.Settings(start_method="fork"))

# %% -------------------- Device --------------------
cuda_avail = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if cuda_avail else "cpu")

logging.info(f"Device: {device}")

# torch.cuda.synchronize(device)

# %% -------------------- Data --------------------

train_dataset = ImpliCityDataset('train', cfg_dataset=cfg_dataset, merge_query_occ=not cfg_multi_class,
                                 random_sample=True, random_length=cfg_training['random_dataset_length'],
                                 flip_augm=cfg_training['augmentation']['flip'],
                                 rotate_augm=cfg_training['augmentation']['rotate'])

val_dataset = ImpliCityDataset('val', cfg_dataset=cfg_dataset, merge_query_occ=not cfg_multi_class,
                               random_sample=False,
                               flip_augm=False, rotate_augm=False)

vis_dataset = ImpliCityDataset('vis', cfg_dataset=cfg_dataset, merge_query_occ=not cfg_multi_class,
                               random_sample=False,
                               flip_augm=False, rotate_augm=False)

n_workers = cfg_loader['n_workers']
# train dataloader
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=n_workers, shuffle=True,
                          # pin_memory=True
                          )
# val dataloader
val_loader = DataLoader(val_dataset, batch_size=val_batch_size, num_workers=n_workers, shuffle=False)
# visualization dataloader
vis_loader = DataLoader(vis_dataset, batch_size=1, num_workers=n_workers, shuffle=False)

logging.info(f"dataset path: '{cfg_dataset['path']}'")
logging.info(f"training data: n_data={len(train_dataset)}, batch_size={batch_size}")
logging.info(f"validation data: n_data={len(val_dataset)}, val_batch_size={val_batch_size}")

# %% -------------------- Model --------------------
model = get_model(cfg, device)

wandb.watch(model)

# %% -------------------- Optimizer --------------------
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Scheduler
cfg_scheduler = cfg_training['scheduler']
_scheduler_type = cfg_scheduler['type']
_scheduler_kwargs = cfg_scheduler['kwargs']
if 'MultiStepLR' == _scheduler_type:
    scheduler = MultiStepLR(optimizer=optimizer, gamma=_scheduler_kwargs['gamma'], milestones=_scheduler_kwargs['milestones'])
elif 'CyclicLR' == _scheduler_type:
    scheduler = CyclicLR(optimizer=optimizer,
                         base_lr=_scheduler_kwargs['base_lr'],
                         max_lr=_scheduler_kwargs['max_lr'],
                         mode=_scheduler_kwargs['mode'],
                         scale_mode=_scheduler_kwargs.get('scale_mode', 'cycle'),
                         gamma=_scheduler_kwargs['gamma'],
                         step_size_up=_scheduler_kwargs['step_size_up'],
                         step_size_down=_scheduler_kwargs['step_size_down'],
                         cycle_momentum=False)
else:
    raise ValueError("Unknown scheduler type")

# %% -------------------- Trainer --------------------
# Loss
if cfg_multi_class:
    criteria = wrapped_cross_entropy
else:
    criteria = wrapped_bce

trainer = Trainer(model=model, optimizer=optimizer, criteria=criteria, device=device,
                  optimize_every=cfg_training['optimize_every'], cfg_loss_weights=cfg_training['loss_weights'],
                  multi_class=cfg_multi_class, multi_tower_weights=cfg_training.get('multi_tower_weights', None),
                  balance_weight=cfg_training['loss_weights'].get('balance_building_weight', False))

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
land_mask_path_dict = {
    'building': cfg_dataset['mask_files']['building'],
    'forest': cfg_dataset['mask_files']['forest'],
    'water': cfg_dataset['mask_files']['water']
}
evaluator = DSMEvaluator(gt_dsm_path, gt_mask_path, land_mask_path_dict)

# %% -------------------- Initialize training --------------------
# Load checkpoint
checkpoint_io = CheckpointIO(out_dir_run, model=model, optimizer=optimizer, scheduler=scheduler)
resume_from = cfg_training.get('resume_from', None)
resume_scheduler = cfg_training.get('resume_scheduler', True)
try:
    _resume_from_file = resume_from if resume_from is not None else ""
    logging.info(f"resume: {_resume_from_file}")
    # print(os.path.exists(_resume_from_file))
    load_dict = checkpoint_io.load(_resume_from_file, resume_scheduler=resume_scheduler)
    logging.info(f"Checkpoint loaded: '{_resume_from_file}'")
except FileExistsError:
    load_dict = dict()
    logging.info(f"No checkpoint, train from beginning")

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

# %% -------------------- Training iterations --------------------
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
    wandb_dic = {f"test/{k}": v for k, v in output_dic['overall'].items()}
    _output_path = os.path.join(out_dir_tiff, f"{pure_run_name}_dsm_{n_iter:06d}_eval.txt")
    str_stat = print_statistics(output_dic, f"{pure_run_name}-iter{n_iter}", save_to=_output_path)
    logging.info(f"DSM evaluation saved to '{_output_path}")
    # residual
    dsm_writer.set_data(diff_arr)
    _output_path = os.path.join(out_dir_tiff, f"{pure_run_name}_residual_{n_iter:06d}.tiff")
    dsm_writer.write_to_file(_output_path)
    logging.info(f"DSM residual saved to '{_output_path}")
    _dsm_log_dic = {f'DSM/{k}/{k2}': v2 for k, v in output_dic.items() for k2, v2 in v.items()}
    wandb.log(_dsm_log_dic, step=n_iter)


try:
    while True:
        for batch in train_loader:
            # Train step
            _ = trainer.train_step(batch)

            if 0 == trainer.accumulated_steps:
                # Use gradient accumulation. Each optimize step is 1 iteration
                n_iter += 1

                training_time = datetime.now() - t_start + last_training_time

                loss = trainer.last_avg_loss_total
                loss_category = trainer.last_avg_loss_category
                wdb_dic = {
                    'iteration': n_iter,
                    'train/loss': loss,
                    'lr': scheduler.get_last_lr()[0],
                    'misc/training_time': training_time.total_seconds(),
                    'misc/n_query_points': trainer.last_avg_n_pts
                    # 'epoch': n_epoch
                }
                for _key, _value in trainer.last_avg_loss_category.items():
                    wdb_dic[f'train/{_key}'] = _value

                for _key, _value in trainer.last_avg_metrics_total.items():
                    wdb_dic[f'train/{_key}'] = _value
                for _key, _value in trainer.last_avg_metrics_category.items():
                    wdb_dic[f'train/{_key}'] = _value
                wandb.log(wdb_dic, step=n_iter)

                if print_every > 0 and (n_iter % print_every) == 0:
                    logging.info(f"iteration: {n_iter:6d}, loss ={loss:7.5f}, training_time = {training_time}")

                # Save checkpoint
                if checkpoint_every > 0 and (n_iter % checkpoint_every) == 0:
                    logging.info('Saving checkpoint')
                    _checkpoint_file = os.path.join(out_dir_ckpt, DEFAULT_MODEL_FILE)
                    checkpoint_io.save(_checkpoint_file, n_iter=n_iter, loss_val_best=metric_val_best,
                                       training_time=training_time.total_seconds())
                    logging.info(f"Checkpoint saved to: '{_checkpoint_file}'")

                # Backup if necessary
                if backup_every > 0 and (n_iter % backup_every) == 0:
                    logging.info('Backing up checkpoint')
                    _checkpoint_file = os.path.join(out_dir_ckpt, f'model_{n_iter}.pt')
                    checkpoint_io.save(_checkpoint_file, n_iter=n_iter, loss_val_best=metric_val_best,
                                       training_time=training_time.total_seconds())
                    logging.info(f"Backup to: {_checkpoint_file}")

                # Validation
                if validate_every > 0 and (n_iter % validate_every) == 0:
                    with torch.no_grad():
                        eval_dict = trainer.evaluate(val_loader)
                        metric_val = eval_dict[model_selection_metric]

                    logging.info(f"Model selection metric: {model_selection_metric} = {metric_val:.4f}")

                    wandb_dic = {f"val/{k}": v for k, v in eval_dict.items()}
                    # print('validation wandb_dic: ', wandb_dic)
                    wandb.log(wandb_dic, step=n_iter)

                    logging.info(
                        f"Validation: iteration {n_iter}, {', '.join([f'{k} = {eval_dict[k]}' for k in ['loss', 'iou']])}")

                    # save best model
                    if model_selection_sign * (metric_val - metric_val_best) > 0:
                        metric_val_best = metric_val
                        logging.info(f'New best model ({model_selection_metric}: {metric_val_best})')
                        _checkpoint_file = os.path.join(out_dir_ckpt, 'model_best.pt')
                        checkpoint_io.save(_checkpoint_file, n_iter=n_iter,
                                           loss_val_best=metric_val_best,
                                           training_time=training_time.total_seconds())
                        logging.info(f"Best model saved to: {_checkpoint_file}")

                # Visualization
                if visualize_every > 0 and (n_iter % visualize_every) == 0:
                    visualize()

                # Exit if necessary
                if 0 < exit_after <= (datetime.now() - t_start).total_seconds():
                    logging.info('Time limit reached. Exiting.')
                    _checkpoint_file = os.path.join(out_dir_ckpt, DEFAULT_MODEL_FILE)
                    checkpoint_io.save(_checkpoint_file, n_iter=n_iter, loss_val_best=metric_val_best,
                                       training_time=training_time.total_seconds())
                    exit(3)

                scheduler.step()
                # optimize step[end]
            # batch[end]
except IOError as e:
    logging.error("Error: " + e.__str__())
