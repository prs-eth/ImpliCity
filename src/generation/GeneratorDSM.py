# encoding: utf-8
# Author: Bingxin Ke
# Created: 2021/10/21
import logging
import math
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.io import RasterWriter, RasterData
from src.dataset import ImpliCityDataset
from src.model import ImpliCityONet, ConvolutionalOccupancyNetwork


class DSMGenerator:
    NODATA_VALUE = -9999
    DEFAULT_SHIFT_H = 1000  # a number that is large enough, for finding largest h in each pixel

    def __init__(self, model: nn.Module, device, data_loader: DataLoader, dsm_pixel_size, fill_empty=False,
                 h_range=None, h_res_0=0.25, upsample_steps=3, points_batch_size=300000, half_blend_percent=None,
                 crs_epsg=32632):
        if half_blend_percent is None:
            half_blend_percent = [0.5, 0.5]

        self.model: nn.Module = model
        self.device = device
        self.fill_empty = fill_empty
        self.data_loader: DataLoader = data_loader
        self.pixel_size = torch.tensor(dsm_pixel_size, dtype=torch.float64)
        self.half_blend_percent = half_blend_percent
        self.crs_epsg = crs_epsg
        if h_range is None:
            self.h_range = torch.tensor([-50, 100])
        else:
            self.h_range = torch.tensor(h_range)
        self.h_res_0 = h_res_0
        self.upsample_steps = upsample_steps
        assert self.upsample_steps >= 1

        self.points_batch_size = points_batch_size

        self._dataset: ImpliCityDataset = data_loader.dataset
        self.z_scale = self._dataset.z_std
        self.patch_size: torch.Tensor = self._dataset.patch_size.double()
        self.patch_strip: torch.Tensor = torch.tensor(self._dataset.slide_window_strip).double()

        # only allows regular cropping
        assert self._dataset.random_sample is False, "Only regular patching is accepted"
        assert 1 == self.data_loader.batch_size, "Only batch size == 1 is accepted"

        # get boundary of _data
        self.l_bound = np.inf
        self.b_bound = np.inf
        self.r_bound = -np.inf
        self.t_bound = -np.inf
        for info in self._dataset.chunk_info_ls:
            l_bound, b_bound = info['min_bound'][:2]
            r_bound, t_bound = info['max_bound'][:2]
            self.l_bound = l_bound if l_bound < self.l_bound else self.l_bound
            self.b_bound = b_bound if b_bound < self.b_bound else self.b_bound
            self.r_bound = r_bound if r_bound > self.r_bound else self.r_bound
            self.t_bound = t_bound if t_bound > self.t_bound else self.t_bound

        self.dsm_shape = RasterWriter.cal_dsm_shape([self.l_bound, self.b_bound], [self.r_bound, self.t_bound],
                                                    self.pixel_size)

        self._default_query_grid = self._generate_query_grid().to(self.device)
        self._default_true = torch.ones(self._default_query_grid.shape[:2]).bool().to(self.device)
        self._default_false = torch.zeros(self._default_query_grid.shape[:2]).bool().to(self.device)

        self.patch_weight = self._linear_blend_patch_weight(self._default_query_grid.shape[:2],
                                                            self.half_blend_percent).to(self.device)
        assert torch.float64 == self.patch_weight.dtype

    def _generate_query_grid(self):
        pzs = torch.arange(self.h_range[0].item(), self.h_range[1].item(), self.h_res_0) / self.z_scale

        _grid_xy_shape = torch.round(self.patch_size / self.pixel_size).long()
        shape = [_grid_xy_shape[0].item(), _grid_xy_shape[1].item(), pzs.shape[0]]
        _size = shape[0] * shape[1] * shape[2]
        pxs = torch.linspace(0., 1., _grid_xy_shape[0].item())
        pys = torch.linspace(1., 0., _grid_xy_shape[1].item())

        pxs = pxs.reshape((1, -1, 1)).expand(*shape)
        pys = pys.reshape((-1, 1, 1)).expand(*shape)
        pzs = pzs.reshape((1, 1, -1)).expand(*shape)

        query_grid = torch.stack([pxs, pys, pzs], dim=3)
        return query_grid

    @staticmethod
    def _linear_blend_patch_weight(grid_shape_2d, half_blend_percent):
        """

        Args:
            grid_shape_2d:
            half_blend_percent: defines the percentage of linear slop of linear blend with shape [0 ... 1 ... 1 ... 0]
                both x y direction should be < 0.5
                e.g. [0.3, 0.3]

        Returns:

        """
        assert 0 <= half_blend_percent[0] <= 0.5, "half_blend_percent value should between [0, 0.5]"
        assert 0 <= half_blend_percent[1] <= 0.5, "half_blend_percent value should between [0, 0.5]"
        MIN_WEIGHT = 1e-3
        weight_tensor_x = torch.ones(grid_shape_2d, dtype=torch.float64)
        weight_tensor_y = torch.ones(grid_shape_2d, dtype=torch.float64)
        idx_x = math.floor(grid_shape_2d[0] * half_blend_percent[0])
        idx_y = math.floor(grid_shape_2d[1] * half_blend_percent[1])
        if idx_x > 0:
            weight_tensor_x[:, 0:idx_x] = torch.linspace(MIN_WEIGHT, 1, idx_x, dtype=torch.float64).reshape((1, -1)).expand((grid_shape_2d[0], idx_x))
            weight_tensor_x[:, -idx_x:] = torch.linspace(1, MIN_WEIGHT, idx_x, dtype=torch.float64).reshape((1, -1)).expand((grid_shape_2d[0], idx_x))
        if idx_y > 0:
            weight_tensor_y[0:idx_y, :] = torch.linspace(MIN_WEIGHT, 1, idx_y, dtype=torch.float64).reshape((-1, 1)).expand((idx_y, grid_shape_2d[1]))
            weight_tensor_y[-idx_y:, :] = torch.linspace(1, MIN_WEIGHT, idx_y, dtype=torch.float64).reshape((-1, 1)).expand((idx_y, grid_shape_2d[1]))
        # weight_tensor = (weight_tensor_x + weight_tensor_y) / 2.
        weight_tensor = weight_tensor_x * weight_tensor_y
        return weight_tensor

    def generate_dsm(self, save_to: str):
        """ assume height > 0 ? """
        device = self.device
        patch_weight = self.patch_weight.detach().to(device)
        default_query_grid = self._default_query_grid.detach().to(device)

        tiff_data = RasterData()
        tiff_data.set_transform(bl_bound=[self.l_bound, self.b_bound], tr_bound=[self.r_bound, self.t_bound],
                                pixel_size=self.pixel_size, crs_epsg=self.crs_epsg)

        dsm_tensor = torch.zeros(self.dsm_shape, dtype=torch.float64).to(device)
        weight_tensor = torch.zeros(self.dsm_shape, dtype=torch.float64).to(device)

        start_time = time.time()

        for vis_data in tqdm(self.data_loader, desc="Generating DSM"):

            min_bound = vis_data['min_bound'].squeeze().double()
            max_bound = vis_data['max_bound'].squeeze().double()
            transform = vis_data['transform'].squeeze().double()

            # Use pixel center height to represent
            min_bound_center = min_bound + self.pixel_size / 2.
            max_bound_center = max_bound - self.pixel_size / 2.

            z_shift = transform[2, 3].item()

            # generate 3d grid
            query_grid = default_query_grid.clone()

            # query patch dsm
            h_grid_norm, is_empty = self._query_patch_dsm(vis_data, query_grid)

            if self.fill_empty:
                h_grid_norm[is_empty] = 0.
                is_empty = self._default_false

            h_grid = h_grid_norm * self.z_scale + z_shift

            # add weighted dsm to tensor
            l_col, b_row = tiff_data.query_col_row(min_bound_center[0].item(), min_bound_center[1].item())
            r_col, t_row = tiff_data.query_col_row(max_bound_center[0].item(), max_bound_center[1].item())

            weighted_h_grid = h_grid * patch_weight

            dsm_tensor[t_row:b_row + 1, l_col:r_col + 1] += weighted_h_grid * ~is_empty
            weight_tensor[t_row:b_row + 1, l_col:r_col + 1] += patch_weight * ~is_empty


        is_empty_hole = 0 == weight_tensor
        dsm_tensor[is_empty_hole] = self.NODATA_VALUE
        weight_tensor[is_empty_hole] = 1

        dsm_tensor = dsm_tensor / weight_tensor

        # fix edge
        # logging.debug("Fix edge")
        # dsm_tensor[:, -1] = torch.where(dsm_tensor[:, -1] <= 0, dsm_tensor[:, -2], dsm_tensor[:, -1])
        # dsm_tensor[:, 0] = torch.where(dsm_tensor[:, 0] <= 0, dsm_tensor[:, 1], dsm_tensor[:, 0])
        # dsm_tensor[-1, :] = torch.where(dsm_tensor[-1, :] <= 0, dsm_tensor[-2, :], dsm_tensor[-1, :])
        # dsm_tensor[0, :] = torch.where(dsm_tensor[0, :] <= 0, dsm_tensor[1, :], dsm_tensor[0, :])

        # fill negative and nan values
        # dsm_tensor = dsm_tensor.cpu()
        # _rows, _cols = torch.where(is_empty_hole)
        # logging.debug(f"{len(_rows)} empty pixels in DSM")
        # if self.fill_empty:
        #     # print('_rows:', _rows)
        #     # print('_cols:', _cols)
        #     for k in tqdm(range(len(_rows)), desc="Filling empty values in DSM"):
        #         i = (max(_rows[k] - 2, 0), min(_rows[k] + 3, dsm_tensor.shape[0] - 1))
        #         j = (max(_cols[k] - 2, 0), min(_cols[k] + 3, dsm_tensor.shape[1] - 1))
        #         neighbor = dsm_tensor[i[0]:i[1], j[0]:j[1]]
        #         dsm_tensor[_rows[k], _cols[k]] = torch.mean(neighbor[neighbor > 0])
        # else:
        #     dsm_tensor[dsm_tensor <= 0] = self.NODATA_VALUE

        # print('dsm_tensor', dsm_tensor.shape, dsm_tensor.max(), dsm_tensor.min(), dsm_tensor.mean())

        end_time = time.time()
        process_time = end_time - start_time
        logging.info(f"DSM Generation time: {process_time}")

        tiff_data.set_data(dsm_tensor, 1)
        tiff_writer = RasterWriter(tiff_data)
        tiff_writer.write_to_file(save_to)
        return tiff_writer

    def _query_patch_dsm(self, data, query_grid):
        self.model.eval()
        device = self.device
        shape = query_grid.shape
        inputs = data.get('inputs').to(device)
        query_grid = query_grid.to(device)
        current_h_res = self.h_res_0 / self.z_scale
        is_empty = torch.zeros(query_grid.shape[:2]).bool().to(device)
        with torch.no_grad():
            if isinstance(self.model, ConvolutionalOccupancyNetwork):
                kwargs = {}
            elif isinstance(self.model, ImpliCityONet):
                input_img: torch.Tensor = data.get('image').to(device)
                kwargs = {'input_img': input_img}
            else:
                raise NotImplemented
            c = self.model.encode_inputs(inputs, **kwargs)
            kwargs = {}
            for i in range(self.upsample_steps+1):
                query_p = query_grid.reshape((-1, 3)).to(device)
                occ = self._eval_points(query_p, c, **kwargs)

                # occ grid
                occ = occ.reshape(query_grid.shape[:3])
                occupied_h_grid = (query_grid[:, :, :, 2] + self.DEFAULT_SHIFT_H) * occ - self.DEFAULT_SHIFT_H
                largest_h_grid = occupied_h_grid.max(2).values.reshape(query_grid.shape[:2], 1)

                # if self.fill_empty:
                is_empty = is_empty | torch.where(largest_h_grid <= -self.DEFAULT_SHIFT_H,
                                                  self._default_true, self._default_false)
                largest_h_grid = largest_h_grid * ~is_empty

                if i < self.upsample_steps:
                    current_h_res = current_h_res / 4
                    delta_h = torch.tensor(
                        [-current_h_res, 0, current_h_res, current_h_res * 2, current_h_res * 3]).reshape((-1, 1))\
                        .to(device)
                    expanded = largest_h_grid.reshape((*shape[:2], 1, 1)).expand((*shape[:2], len(delta_h), 1))
                    expanded = expanded + delta_h
                    query_grid = torch.cat([
                        query_grid[:, :, 0:1, 0:1].expand((*shape[:2], len(delta_h), 1)),
                        query_grid[:, :, 0:1, 1:2].expand((*shape[:2], len(delta_h), 1)),
                        expanded
                    ], 3)

        return largest_h_grid, is_empty

    def _eval_points(self, p, c, **kwargs):
        p_split = torch.split(p, self.points_batch_size)
        occ_hats = []
        for pi in p_split:
            pi = pi.unsqueeze(0).to(self.device)
            with torch.no_grad():
                pred = self.model.decode(pi, c, **kwargs)
                occ_hat = self.model.pred2occ(pred)
                occ_hat = occ_hat > 0
            occ_hats.append(occ_hat.squeeze(0).detach())
        occ_hat = torch.cat(occ_hats, dim=0)
        return occ_hat
