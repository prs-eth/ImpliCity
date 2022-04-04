# encoding: utf-8
# Author: Bingxin Ke
# Created: 2021/9/27
"""
    Build dataset for training and inference.


    Usage example:
        python scripts/dataset_ImpliCity/build_dataset.py config/dataset/ImpliCity/ImpliCity.yaml

    Input:
        config file (yaml),
        including data path of:
            1. Point clouds.
                These point clouds are pre-aligned.
            2. Ground truth meshes
            3. Mask files
        and settings
    Output:
        1. Point cloud (chunks)
        2. Query points (chunks)
        3. Visualization
        4. Chunk info
"""
import sys
from collections import defaultdict

sys.path.append(".")

import logging
import os
import shutil
import sys
from typing import List, Dict

import numpy as np
import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import pymesh
from pymesh import Mesh as PyMesh
from tqdm import tqdm
import trimesh
import argparse
import open3d as o3d

from src.utils.libconfig import config, lock_seed
from src.utils.libpc import load_pc, save_pc_to_ply, crop_pc_2d
from src.utils.libtrimesh.crop import crop_mesh_2d
from src.utils.under_mesh import check_under_mesh
from src.io import RasterReader
from src.utils.libraster import dilate_mask
from src.utils.libcoord.coord_transform import extent_transform_to_points
# from src.utils.libmesh import check_mesh_contains
from src.utils.libconfig import config_logging

# %% Load config
parser = argparse.ArgumentParser(
    description='Build Resdepth dataset'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--del-old', action='store_true', default=False, help='Delete old folder.')

args = parser.parse_args()

config_file_path = args.config
cfg = config.load_config(config_file_path)

# config logging
config_logging(cfg['logging'])

# Shorthands
# cfg_aoi = cfg['area_of_interest']
build_training_data = cfg.get('build_training_data', False)
cfg_query = cfg['query_points']
cfg_chunk = cfg['chunk']
gt_mesh_folder = cfg['gt_mesh_folder']
if build_training_data:
    gt_mesh_files = {
        _key: os.path.join(gt_mesh_folder, _value) for _key, _value in cfg['gt_mesh_files'].items()
    }
else:
    gt_mesh_files = None

input_pc_merged = cfg.get('input_pointcloud_merged', None)
input_pc_folder = cfg.get('input_pointcloud_folder', None)
if input_pc_merged is not None:
    # If exist merged point cloud, use merged one
    input_pc_paths: List = [input_pc_merged]
elif input_pc_folder is not None:
    input_pc_paths: List = [
        os.path.join(input_pc_folder, _path) for _path in os.listdir(input_pc_folder)
    ]
else:
    logging.error("No input point cloud.")
    raise IOError("No input point cloud.")

cfg_output = cfg['output']
output_folder = cfg_output['output_folder']

display_warn = cfg['logging']['display_inmesh_warning']
save_vis = cfg_output['save_visualization_pc']

# lock seed
if cfg['lock_seed']:
    lock_seed(0)

# %% Generate chunks
chunk_x = cfg_chunk['chunk_x']
chunk_y = cfg_chunk['chunk_y']
chunks: Dict[int, Dict] = defaultdict(Dict)
for i, x_l in enumerate(chunk_x[:-1]):
    for j, y_b in enumerate(chunk_y[:-1]):
        _p_min = np.array([x_l, y_b])
        _p_max = np.array([chunk_x[i + 1], chunk_y[j + 1]])
        chunks[len(chunks)] = {'min_bound': _p_min, 'max_bound': _p_max}

# %% Clear target directory
if os.path.exists(output_folder):
    if args.del_old:
        _remove_old = 'y'
    else:
        _remove_old = input(f"Output folder exists at '{output_folder}', \n\r remove old one? (y/n): ")
    if 'y' == _remove_old:
        try:
            shutil.rmtree(output_folder)  # force remove
            logging.info(f"Removed old output folder: '{output_folder}'")
        except OSError as e:
            logging.error(e)
            logging.error("Build failed. Remove output folder manually and try again")
            sys.exit()
    if 'n' == _remove_old:
        logging.info("Remove output folder manually and try again")
        sys.exit()

# Create folders
patch_folder_ls = []
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

logging.info(f"Output folder ready at: '{output_folder}'")

# %% Load data
if build_training_data:
    mesh_dic = {key: trimesh.load_mesh(gt_mesh_files[key]) for key in tqdm(['roof', 'facade', 'terrain', 'buildings', 'gt'],
                                                                           desc="loading meshes")}
    mesh_terrain_pymesh: PyMesh = pymesh.load_mesh(gt_mesh_files['terrain'])
    logging.info("Meshes loaded")

# Load point clouds and merge
merged_pts: np.ndarray = np.empty((0, 3))
for _full_path in tqdm(input_pc_paths, desc="Loading point clouds"):
    _temp_points = load_pc(_full_path)
    merged_pts = np.append(merged_pts, _temp_points, 0)
# print('merged_pts: ', type(merged_pts), merged_pts.dtype)

del _temp_points
logging.info("Point clouds merged")

# Load masks

mask_keys = ['gt', 'building', 'forest', 'water']
cfg_mask_files = cfg['mask_files']
raster_masks: Dict[str, RasterReader] = {key: RasterReader(cfg_mask_files[key]) for key in mask_keys
                                         if cfg_mask_files[key] is not None}
dsm_gt: RasterReader = RasterReader(cfg['gt_dsm'])

# dilate building mask
dilate_build = cfg.get('dilate_building', None)
if dilate_build is not None:
    _mask = raster_masks['building'].get_data()
    _mask = dilate_mask(_mask, iterations=dilate_build)
    raster_masks['building'].set_data(_mask)

out_of_mask_value = cfg['out_of_mask_value']
logging.info("Raster masks loaded")


# %%
def meshes_to_bbox(terrain_mesh, building_mesh, min_z=0):
    bbox_terrain = terrain_mesh.bounding_box
    _ter_p1, _ter_p2 = extent_transform_to_points(bbox_terrain.primitive.extents, bbox_terrain.primitive.transform)
    try:
        bbox_roof = building_mesh.bounding_box
        _build_p1, _build_p2 = extent_transform_to_points(bbox_roof.primitive.extents, bbox_roof.primitive.transform)
    except AttributeError as e:
        # no building in this patch
        _build_p2 = _ter_p2
    p_min = np.array([_ter_p1[0], _ter_p1[1], _ter_p1[2]])
    z_max = max(_build_p2[2], _ter_p1[2] + min_z)
    p_max = np.array([_ter_p2[0], _ter_p2[1], z_max])
    return p_min, p_max


# def calculate_occupancy(query_pts_dict, building_mesh, terrain_pymesh):

# query_uniform_occ = np.concatenate(query_uniform_occ_ls, 0)
# del query_uniform_occ_ls

def is_under_dsm(points, dsm: RasterReader):
    dsm_value = dsm.query_value_3d_points(points)
    is_under = points[:, 2] <= dsm_value
    return is_under

# %% Main part
# initialize
chunk_safe_padding = cfg_chunk['chunk_safe_padding']
win_size = cfg_chunk['sliding_window']
win_step = cfg_chunk['sliding_step']
min_z = cfg_chunk['min_z']
padding_z = cfg_chunk['padding_z']
n_uniform_num = cfg_query['uniform_num']
grid_sample_dist = cfg_query.get('grid_sample_dist', None)

surface_offset_std = cfg_query['surface_offset_std']
surface_radius: Dict = {'roof': cfg_query['roof_radius'],
                        'facade': cfg_query['facade_radius'],
                        'terrain': cfg_query['terrain_radius']}
surface_count_max = cfg_query['surface_count_max']

chunk_info = defaultdict(dict)
# %%

# Split data for each chunk
# _chunk_idx = 1
for _chunk_idx in tqdm(chunks.keys(), desc="Chunks"):
    chunk_name = f"chunk_{_chunk_idx:03d}"
    chunk_dir = os.path.join(output_folder, chunk_name)
    os.makedirs(chunk_dir)
    _chunk_p1, _chunk_p2 = chunks[_chunk_idx]['min_bound'], chunks[_chunk_idx]['max_bound']
    chunk_info[_chunk_idx].update({
        'name': chunk_name,
    })
    if save_vis:
        vis_dir = os.path.join(chunk_dir, "vis")
        os.makedirs(vis_dir)

    if gt_mesh_folder is not None:
        _chunk_p1_pad = _chunk_p1 - np.array([chunk_safe_padding, chunk_safe_padding])
        _chunk_p2_pad = _chunk_p2 + np.array([chunk_safe_padding, chunk_safe_padding])
        chunk_building_mesh_pad = crop_mesh_2d(mesh_dic['buildings'], _chunk_p1_pad, _chunk_p2_pad)
        chunk_terrain_mesh_pad = crop_mesh_2d(mesh_dic['terrain'], _chunk_p1_pad, _chunk_p2_pad)
        chunk_building_mesh = crop_mesh_2d(mesh_dic['buildings'], _chunk_p1, _chunk_p2)
        chunk_terrain_mesh = crop_mesh_2d(mesh_dic['terrain'], _chunk_p1, _chunk_p2)

        # determine 3D bounding box
        chunk_p1_3d, chunk_p2_3d = meshes_to_bbox(chunk_terrain_mesh, chunk_building_mesh, min_z)
        assert (abs(chunk_p1_3d[:2] - _chunk_p1) < 1e-5).all()
        assert (abs(chunk_p2_3d[:2] - _chunk_p2) < 1e-5).all()
        chunk_info[_chunk_idx].update({
            'min_bound': chunk_p1_3d.tolist(),
            'max_bound': chunk_p2_3d.tolist(),
            'surface_point-offset-std': surface_offset_std,
        })
    else:
        chunk_info[_chunk_idx].update({
            'min_bound': _chunk_p1.tolist(),
            'max_bound': _chunk_p2.tolist(),
        })

    # Save input point cloud
    chunk_input_pc, _ = crop_pc_2d(merged_pts, _chunk_p1, _chunk_p2)
    _output_path = os.path.join(chunk_dir, 'input_point_cloud.npz')
    _out_data = {
        'pts': chunk_input_pc
    }
    np.savez(_output_path, **_out_data)

    if save_vis:
        _output_path = os.path.join(vis_dir, f"{chunk_name}-input_point_cloud.ply")
        save_pc_to_ply(_output_path, chunk_input_pc)

    if build_training_data:
        # Sample points
        # sample surface points
        query_pts_dict: Dict = defaultdict()
        for surface, radius in tqdm(surface_radius.items(), desc="Sampling on surface", leave=False, position=1):
            if radius > 0:
                _cropped_surface = crop_mesh_2d(mesh_dic[surface], _chunk_p1, _chunk_p2)
                # surface_pts, _ = trimesh.sample.sample_surface_even(mesh=_cropped_surface, count=surface_count_max, radius=radius)
                surface_pts, _ = trimesh.sample.sample_surface(mesh=_cropped_surface, count=surface_count_max)
                # print('surface_pts: ', surface_pts.shape)
                # downsample
                _pcd = o3d.geometry.PointCloud()
                _pcd.points = o3d.utility.Vector3dVector(surface_pts)
                ds_pcd = _pcd.voxel_down_sample(voxel_size=radius)
                surface_pts = np.asarray(ds_pcd.points)
                # print('downsampled: ', surface_pts.shape)
                # add offset
                offset = np.random.normal(0, surface_offset_std, (len(surface_pts), 3))
                surface_pts += offset
                query_pts_dict[surface] = surface_pts
                logging.debug(f"{len(surface_pts)} points sampled from surface: {surface}")
                # save radius
                chunk_info[_chunk_idx][f'surface-radius-{surface}'] = radius

        # sliding window to generate uniform query points
        if n_uniform_num > 0 or grid_sample_dist is not None:
            query_uniform_pts_ls = []
            for _patch_x1 in tqdm(np.arange(_chunk_p1[0], _chunk_p2[0], win_step), desc="Sampling uniformly", leave=False, position=1):
                _patch_x2 = _patch_x1 + win_size[0]
                for _patch_y1 in np.arange(_chunk_p1[1], _chunk_p2[1], win_step):
                    _patch_y2 = _patch_y1 + win_size[1]
                    # determine bounding box
                    patch_roof_mesh = crop_mesh_2d(chunk_building_mesh_pad, np.array([_patch_x1, _patch_y1]),
                                                   np.array([_patch_x2, _patch_y2]))
                    patch_terrain_mesh = crop_mesh_2d(chunk_terrain_mesh_pad, np.array([_patch_x1, _patch_y1]),
                                                      np.array([_patch_x2, _patch_y2]))
                    patch_p1_3d, patch_p2_3d = meshes_to_bbox(patch_terrain_mesh, patch_roof_mesh, min_z)
                    patch_p1_3d[2] -= padding_z[0]
                    patch_p2_3d[2] += padding_z[1]

                    # Sample uniformly
                    if n_uniform_num > 0:
                        _rand = np.random.rand(n_uniform_num * 3).reshape((-1, 3))
                        query_uniform_pts = _rand * (patch_p2_3d - patch_p1_3d) + patch_p1_3d
                        query_uniform_pts_ls.append(query_uniform_pts)

                    # Sample grid points
                    if grid_sample_dist is not None:
                        _grid_x = np.arange(_patch_x1+grid_sample_dist[0]/2, _patch_x2, grid_sample_dist[0])
                        _grid_y = np.arange(_patch_y1+grid_sample_dist[1]/2, _patch_y2, grid_sample_dist[1])
                        _grid_z = np.arange(patch_p1_3d[2], patch_p2_3d[2], grid_sample_dist[2])
                        _mesh_grid = np.meshgrid(_grid_x, _grid_y, _grid_z)
                        _grid_points = np.concatenate([a[..., np.newaxis] for a in _mesh_grid], 3)
                        query_uniform_pts_ls.append(_grid_points.reshape(-1, 3))

            query_pts_dict['uniform'] = np.concatenate(query_uniform_pts_ls, 0)
            del query_uniform_pts_ls
        else:
            query_pts_dict['uniform'] = np.empty((0, 3))

        # %%
        # Process query points
        for query_type, query_pts in tqdm(query_pts_dict.items(), desc="Processing query points", leave=None, position=1):
            n_pts = query_pts.shape[0]
            if n_pts > 0:
                # Occupancy
                MAX_OCC_PATCH = 2000000  # due to the limit of check_mesh_contains()
                occ_udsm_ls = []  # under GT DSM
                occ_ug_ls = []  # underground
                for pts in tqdm(np.array_split(query_pts, np.ceil(n_pts / MAX_OCC_PATCH)),
                                desc=f"Calculating occupancy ({n_pts} points)", leave=False, position=2):
                    _occ_udsm = is_under_dsm(pts, dsm_gt)
                    _occ_ug = check_under_mesh(mesh_terrain_pymesh, pts)
                    occ_udsm_ls.append(_occ_udsm)
                    occ_ug_ls.append(_occ_ug)
                occ_udsm = np.concatenate(occ_udsm_ls)
                occ_ug = np.concatenate(occ_ug_ls, 0)
                occ_build = occ_udsm & ~occ_ug
                # occ label
                occ_label = np.zeros(occ_build.shape)
                occ_label[occ_build] = 1
                occ_label[occ_udsm & ~occ_build] = 2
            else:
                occ_label = np.empty(0)

            _out_data = {
                'pts': query_pts,
                'occ': occ_label.astype(int)
            }

            # Mask
            for mask in tqdm(mask_keys, desc="Calculating masks", leave=False, position=2):
                if n_pts > 0:
                    mask_values = raster_masks[mask].query_value_3d_points(query_pts, band=1, outer_value=out_of_mask_value)
                    # mask_values = mask_values.astype(np.bool_)
                    _out_data[f"mask_{mask}"] = mask_values
                else:
                    _out_data[f"mask_{mask}"] = np.empty(0).astype(int)

            # Chunkinfo
            chunk_info[_chunk_idx][f'n_query_pts-{query_type}'] = n_pts

            # Save to file
            _output_path = os.path.join(chunk_dir, f'query--{query_type}.npz')
            np.savez(_output_path, **_out_data)

            if save_vis:
                if n_pts > 0:
                    for label in [0, 1, 2]:
                        _occ = (label == occ_label).astype(bool)
                        pts_to_save = query_pts[_occ]
                        save_pc_to_ply(os.path.join(vis_dir, f"{chunk_name}-{query_type}-occ={label}.ply"), pts_to_save)
                else:
                    logging.info(f"Skip saving visualization for {query_type}")

# %% chunk_info.yaml
_output_path = os.path.join(output_folder, "chunk_info.yaml")
with open(_output_path, 'w+') as f:
    yaml.dump(dict(chunk_info), f, default_flow_style=None, allow_unicode=True, Dumper=Dumper)
logging.info(f"chunk_info saved to: '{_output_path}'")
