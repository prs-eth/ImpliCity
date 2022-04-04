# encoding: utf-8
# Created: 2021/10/13

import logging
import math
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
import transformations
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
from torch.utils import data
from tqdm import tqdm

from src.utils.libpc import crop_pc_2d, crop_pc_2d_index
from src.utils.libcoord.coord_transform import invert_transform, apply_transform
from src.io.RasterIO import RasterReader, RasterData

LAND_TYPES = ['building', 'forest', 'water']
LAND_TYPE_IDX = {LAND_TYPES[i]: i for i in range(len(LAND_TYPES))}

# constant for data augmentation
_origin = np.array([0., 0., 0.])
_x_axis = np.array([1., 0., 0.])
_y_axis = np.array([0., 1., 0.])
z_axis = np.array([0., 0., 1.])

# Rotation matrix: rotate nx90 deg clockwise
rot_mat_dic: Dict[int, torch.Tensor] = {
    0: torch.eye(4).double(),
    1: torch.as_tensor(transformations.rotation_matrix(-90. * math.pi / 180., z_axis)).double(),
    2: torch.as_tensor(transformations.rotation_matrix(-180. * math.pi / 180., z_axis)).double(),
    3: torch.as_tensor(transformations.rotation_matrix(-270. * math.pi / 180., z_axis)).double(),
}
# Flip matrix
flip_mat_dic: Dict[int, torch.Tensor] = {
    -1: torch.eye(4).double(),
    0: torch.as_tensor(transformations.reflection_matrix(_origin, _x_axis)).double(),  # flip on x direction (x := -x)
    1: torch.as_tensor(transformations.reflection_matrix(_origin, _y_axis)).double()  # flip on y direction (y := -y)
}


class ImpliCityDataset(data.Dataset):
    """ Load ResDepth Dataset
        for train/val:      {'name', 'inputs', 'transform', 'query_pts', 'query_occ', 'mask_gt', 'mask_building', 'mask_forest', 'mask_water'}
        for test/vis:   {'name', 'inputs', 'transform'}
    """

    # pre-defined filenames
    INPUT_POINT_CLOUD = "input_point_cloud.npz"
    QUERY_POINTS = "query--%s.npz"
    CHUNK_INFO = "chunk_info.yaml"

    def __init__(self, split: str, cfg_dataset: Dict, random_sample=False, merge_query_occ: bool = True,
                 random_length=None, flip_augm=False, rotate_augm=False):
        """
        Args:
            split: 'train', 'val', 'test', 'vis'
            cfg_dataset: dataset configurations
            random_sample: randomly sample patches. if False, use sliding window (parameters are given in cfg_dataset)
            random_length: length of dataset, valid only if random_sample is True,
            merge_query_occ: merge occupancy labels to binary case
            flip_augm: data augmentation by flipping
            rotate_augm: data augmentation by rotation
        """

        # shortcuts
        self.split = split
        self._dataset_folder = cfg_dataset['path']
        self._cfg_data = cfg_dataset
        self._n_input_pts = cfg_dataset['n_input_points']
        self._n_query_pts = cfg_dataset['n_query_points']
        if self.split in ['val'] and not cfg_dataset.get('subsample_val', False):
            self._n_query_pts = None
        self.patch_size = torch.tensor(cfg_dataset['patch_size'], dtype=torch.float64)

        # initialize
        self.images: List[RasterData] = []
        self.data_dic = defaultdict()
        self.dataset_chunk_idx_ls: List = cfg_dataset[f"{split}_chunks"]
        dataset_dir = self._cfg_data['path']
        with open(os.path.join(dataset_dir, self.CHUNK_INFO), 'r') as f:
            self.chunk_info: Dict = yaml.load(f, Loader=Loader)
        self.chunk_info_ls: List = [self.chunk_info[i] for i in self.dataset_chunk_idx_ls]

        # -------------------- Load satellite image --------------------
        images_dic = self._cfg_data.get('satellite_image', None)
        if images_dic is not None:
            image_folder = images_dic['folder']
            for image_name in images_dic['pairs']:
                _path = os.path.join(image_folder, image_name)
                reader = RasterReader(_path)
                self.images.append(reader)
                logging.debug(f"Satellite image loaded: {image_name}")
            assert len(self.images) <= 2, "Only support single image or stereo image"
            assert self.images[-1].T == self.images[0].T
            temp_ls = []
            for _img in self.images:
                _img_arr = _img.get_data().astype(np.int32)
                temp_ls.append(torch.from_numpy(_img_arr[None, :, :]))
            self.norm_image_data: torch.Tensor = torch.cat(temp_ls, 0).long()
            self.norm_image_data = self.norm_image_data.reshape(
                (-1, self.norm_image_data.shape[-2], self.norm_image_data.shape[-1]))  # n_img x h_image x w_image
            # Normalize values
            self._image_mean = images_dic['normalize']['mean']
            self._image_std = images_dic['normalize']['std']
            self.norm_image_data: torch.Tensor = (self.norm_image_data.double() - self._image_mean) / self._image_std
        self.n_images = len(self.images)
        if self.n_images > 0:
            self._image_pixel_size = torch.as_tensor(self.images[0].pixel_size, dtype=torch.float64)
            self._image_patch_shape = self.patch_size / self._image_pixel_size
            assert torch.all(torch.floor(self._image_patch_shape) == self._image_patch_shape),\
                "Patch size should be integer multiple of image pixel size"
            self._image_patch_shape = torch.floor(self._image_patch_shape).long()

        # -------------------- Load point data by chunks --------------------
        for chunk_idx in tqdm(self.dataset_chunk_idx_ls, desc=f"Loading {self.split} data to RAM"):
            info = self.chunk_info[chunk_idx]
            chunk_name = info['name']
            chunk_full_path = os.path.join(dataset_dir, chunk_name)

            # input points
            inputs = np.load(os.path.join(chunk_full_path, self.INPUT_POINT_CLOUD))

            chunk_data = {
                'name': chunk_name,
                'inputs': torch.from_numpy(inputs['pts']).double(),
            }

            # query points
            if self.split in ['train', 'val']:
                query_types = ['uniform']
                use_surface = self._cfg_data['use_surface']
                if use_surface is not None:
                    query_types.extend(use_surface)
                query_pts_ls: List = []
                query_occ_ls: List = []
                masks_ls: Dict[str, List] = {f'mask_{_m}': [] for _m in ['gt', 'building', 'forest', 'water']}
                for surface_type in query_types:
                    file_path = os.path.join(chunk_full_path, self.QUERY_POINTS % surface_type)
                    _loaded = np.load(file_path)
                    query_pts_ls.append(_loaded['pts'])
                    query_occ_ls.append(_loaded['occ'])
                    for _m in masks_ls.keys():  # e.g. mask_gt
                        masks_ls[_m].append(_loaded[_m])
                query_pts: np.ndarray = np.concatenate(query_pts_ls, 0)
                query_occ: np.ndarray = np.concatenate(query_occ_ls, 0)
                masks: Dict[str, np.ndarray] = {_m: np.concatenate(masks_ls[_m], 0) for _m in masks_ls.keys()}
                if merge_query_occ:
                    query_occ = (query_occ > 0).astype(bool)
                del query_pts_ls, query_occ_ls, masks_ls
                chunk_data.update({
                    'query_pts': torch.from_numpy(query_pts).double(),
                    'query_occ': torch.from_numpy(query_occ).float(),
                    'mask_gt': torch.from_numpy(masks['mask_gt']).bool(),
                    'mask_building': torch.from_numpy(masks['mask_building']).bool(),
                    'mask_forest': torch.from_numpy(masks['mask_forest']).bool(),
                    'mask_water': torch.from_numpy(masks['mask_water']).bool(),
                })

            self.data_dic[chunk_idx] = chunk_data

        self.random_sample = random_sample
        self.random_length = random_length
        if self.random_sample and random_length is None:
            logging.warning("random_length not provided when random_sample = True")
            self.random_length = 10

        self.flip_augm = flip_augm
        self.rotate_augm = rotate_augm

        # -------------------- Generate Anchors --------------------
        # bottom-left point of a patch, for regular patch
        self.anchor_points: List[Dict] = []  # [{chunk_idx: int, anchors: [np.array(2), ...]}, ... ]
        if not self.random_sample:
            self.slide_window_strip = cfg_dataset['sliding_window'][f'{self.split}_strip']
            for chunk_idx in self.dataset_chunk_idx_ls:
                chunk_info = self.chunk_info[chunk_idx]
                _min_bound_np = np.array(chunk_info['min_bound'])
                _max_bound_np = np.array(chunk_info['max_bound'])
                _chunk_size_np = _max_bound_np - _min_bound_np
                patch_x_np = np.arange(_min_bound_np[0], _max_bound_np[0] - self.patch_size[0], self.slide_window_strip[0])
                patch_x_np = np.concatenate([patch_x_np, np.array([_max_bound_np[0] - self.patch_size[0]])])
                patch_y_np = np.arange(_min_bound_np[1], _max_bound_np[1] - self.patch_size[1], self.slide_window_strip[1])
                patch_y_np = np.concatenate([patch_y_np, np.array([_max_bound_np[1] - self.patch_size[1]])])
                # print('patch_y', patch_y.shape, patch_y)
                xv, yv = np.meshgrid(patch_x_np, patch_y_np)
                anchors = np.concatenate([xv.reshape((-1, 1)), yv.reshape((-1, 1))], 1)
                anchors = torch.from_numpy(anchors).double()
                # print('anchors', anchors.shape)
                for anchor in anchors:
                    self.anchor_points.append({
                        'chunk_idx': chunk_idx,
                        'anchor': anchor
                    })

        # -------------------- normalization factors --------------------
        _x_range = cfg_dataset['normalize']['x_range']
        _y_range = cfg_dataset['normalize']['y_range']
        self._min_norm_bound = [_x_range[0], _y_range[0]]
        self._max_norm_bound = [_x_range[1], _y_range[1]]
        self.z_std = cfg_dataset['normalize']['z_std']
        self.scale_mat = torch.diag(torch.tensor([self.patch_size[0] / (_x_range[1] - _x_range[0]),
                                                  self.patch_size[1] / (_y_range[1] - _y_range[0]),
                                                  self.z_std,
                                                  1], dtype=torch.float64))
        self.shift_norm = torch.cat([torch.eye(4, 3, dtype=torch.float64),
                                     torch.tensor([(_x_range[1] - _x_range[0]) / 2.,
                                                   (_y_range[1] - _y_range[0]) / 2., 0, 1]).reshape(-1, 1)], 1)  # shift from [-0.5, 0.5] to [0, 1]

    def __len__(self):
        if self.random_sample:
            return self.random_length
        else:
            return len(self.anchor_points)

    def __getitem__(self, idx):
        """
        Get patch data and assemble point clouds for training.
        Args:
            idx: index of data

        Returns: a dict of data
        """
        # -------------------- Get patch anchor point --------------------
        if self.random_sample:
            # randomly choose anchors
            # chunk_idx = idx % len(self.dataset_chunk_idx_ls)
            chunk_idx = self.dataset_chunk_idx_ls[idx % len(self.dataset_chunk_idx_ls)]
            chunk_info = self.chunk_info[chunk_idx]
            _min_bound = torch.tensor(chunk_info['min_bound'], dtype=torch.float64)
            _max_bound = torch.tensor(chunk_info['max_bound'], dtype=torch.float64)
            _chunk_size = _max_bound - _min_bound
            _rand = torch.rand(2, dtype=torch.float64)
            anchor = _rand * (_chunk_size[:2] - self.patch_size[:2])
            if self.n_images > 0:
                anchor = torch.floor(anchor / self._image_pixel_size) * self._image_pixel_size
                # print('anchor: ', anchor)
            anchor += _min_bound[:2]
        else:
            # regular patches
            _anchor_info = self.anchor_points[idx]
            chunk_idx = _anchor_info['chunk_idx']
            anchor = _anchor_info['anchor']
        min_bound = anchor
        max_bound = anchor + self.patch_size.double()
        assert chunk_idx in self.dataset_chunk_idx_ls
        assert torch.float64 == min_bound.dtype  # for geo-coordinate, must use float64

        # -------------------- Input point cloud --------------------
        # Crop inputs
        chunk_data = self.data_dic[chunk_idx]
        inputs, _ = crop_pc_2d(chunk_data['inputs'], min_bound, max_bound)
        shift_strategy = self._cfg_data['normalize']['z_shift']
        if 'median' == shift_strategy:
            z_shift = torch.median(inputs[:, 2]).double().reshape(1)
        elif '20quantile' == shift_strategy:
            z_shift = torch.tensor([np.quantile(inputs[:, 2].numpy(), 0.2)])
        elif 'mean' == shift_strategy:
            z_shift = torch.mean(inputs[:, 2]).double().reshape(1)
        else:
            raise ValueError(f"Unknown shift strategy: {shift_strategy}")

        # subsample inputs
        if self._n_input_pts is not None and inputs.shape[0] > self._n_input_pts:
            _idx = np.random.choice(inputs.shape[0], self._n_input_pts)
            inputs = inputs[_idx]
        # print('inputs: ', inputs.min(0)[0], inputs.max(0)[0])
        # print('min_bound: ', min_bound)
        # print('z_shift: ', z_shift)
        # print('inputs first point: ', inputs[0])
        # print('diff: ', inputs[0] - torch.cat([min_bound, z_shift]))

        # -------------------- Augmentation --------------------
        if self.rotate_augm:
            rot_times = list(rot_mat_dic.keys())[np.random.choice(len(rot_mat_dic))]
            # rot_mat = rot_mat_ls[np.random.choice(len(rot_mat_ls))]
        else:
            # rot_mat = rot_mat_ls[0]
            rot_times = 0
        rot_mat = rot_mat_dic[rot_times]

        if self.flip_augm:
            flip_dim_pc = list(flip_mat_dic.keys())[np.random.choice(len(flip_mat_dic))]
            # flip_mat = flip_mat_ls[np.random.choice(len(flip_mat_ls))]
        else:
            # flip_mat = flip_mat_ls[0]
            flip_dim_pc = -1
        flip_mat = flip_mat_dic[flip_dim_pc]

        # -------------------- Normalization --------------------
        # Normalization matrix
        # Transformation matrix: normalize to [-0.5, 0.5]
        transform_mat = self.scale_mat.clone()
        transform_mat[0:3, 3] = torch.cat([(min_bound + max_bound)/2., z_shift], 0)
        normalize_mat = self.shift_norm.double() @ flip_mat.double() @ rot_mat.double() \
                        @ invert_transform(transform_mat).double()
        transform_mat = invert_transform(normalize_mat)
        assert torch.float64 == transform_mat.dtype

        # Normalize inputs
        inputs_norm = apply_transform(inputs, normalize_mat)
        inputs_norm = inputs_norm.float()
        # print('normalized inputs, first point: ', inputs_norm[0])
        # crop again (in case of calculation error)
        inputs_norm, _ = crop_pc_2d(inputs_norm, self._min_norm_bound, self._max_norm_bound)
        # # debug only
        # assert torch.min(inputs_norm[:, :2]) > 0
        # assert torch.max(inputs_norm[:, :2]) < 1

        out_data = {
            'name': f"{chunk_data['name']}-patch{idx}",
            'inputs': inputs_norm,
            'transform': transform_mat.double().clone(),
            'min_bound': min_bound.double().clone(),
            'max_bound': max_bound.double().clone(),
            'flip': flip_dim_pc,
            'rotate': rot_times
        }

        # -------------------- Query points --------------------
        if self.split in ['train', 'val']:
            # Crop query points
            points, points_idx = crop_pc_2d(chunk_data['query_pts'], min_bound, max_bound)
            # print('points, first point: ', points[0])
            # print('diff: ', points[0] - torch.cat([min_bound, z_shift]))
            # Normalize query points
            points_norm = apply_transform(points, normalize_mat)
            # print('normalized points, first point: ', points_norm[0])
            points_norm = points_norm.float()
            # crop again in case of calculation error
            points_idx2 = crop_pc_2d_index(points_norm, self._min_norm_bound, self._max_norm_bound)
            points_norm = points_norm[points_idx2]
            points_idx = points_idx[points_idx2]
            # # debug only
            # assert torch.min(points_norm[:, :2]) > 0
            # assert torch.max(points_norm[:, :2]) < 1

            # points_idx = crop_pc_2d_index(chunk_data['query_pts'], min_bound, max_bound)
            # print(points_idx.shape)
            if self._n_query_pts is not None and points_idx.shape[0] > self._n_query_pts:
                _idx = np.random.choice(points_idx.shape[0], self._n_query_pts)
                points_norm = points_norm[_idx]
                points_idx = points_idx[_idx]
            # points = chunk_data['query_pts'][points_idx]
            # print(points.shape)

            out_data['query_pts'] = points_norm
            out_data['query_occ'] = chunk_data['query_occ'][points_idx].float()
            # assign occ and mask_land
            for _m in ['mask_gt', 'mask_building', 'mask_forest', 'mask_water']:
                out_data[_m] = chunk_data[_m][points_idx]

        # -------------------- Image --------------------
        if self.n_images > 0:
            # index of bottom-left pixel center
            _anchor_pixel_center = anchor + self._image_pixel_size / 2.
            _col, _row = self.images[0].query_col_row(_anchor_pixel_center[0], _anchor_pixel_center[1])
            _image_patches = []
            shape = self._image_patch_shape
            image_tensor = self.norm_image_data[:, _row-shape[0]+1:_row+1, _col:_col+shape[1]]  # n_img x h_patch x w_patch
            # Augmentation
            if rot_times > 0:
                image_tensor = image_tensor.rot90(rot_times, [-1, -2])  # rotate clockwise
            if flip_dim_pc >= 0:
                if 0 == flip_dim_pc:  # points flip on x direction (along y), image flip columns
                    image_tensor = image_tensor.flip(-1)
                if 1 == flip_dim_pc:  # points flip on y direction (along x), image flip rows
                    image_tensor = image_tensor.flip(-2)
                # image_tensor = image_tensor.flip(flip_axis+1)  # dim 0 is image dimension
            assert torch.Size([self.n_images, shape[0], shape[1]]) == image_tensor.shape, f"shape: {torch.Size([self.n_images, shape[0], shape[1]])}image_tensor.shape: {image_tensor.shape}, _row: {_row}, _col: {_col}"
            out_data['image'] = image_tensor.float()

        return out_data
