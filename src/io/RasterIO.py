# encoding: utf-8
# Author: Bingxin Ke
# Created: 2021/11/18

import logging
import math
from collections import defaultdict
from typing import Dict, List, Union

import numpy as np
import rasterio
import torch
from rasterio.transform import Affine
from scipy import ndimage


class RasterData:
    def __init__(self):
        self._editable = True
        # data of different bands
        self._data: Dict = defaultdict()
        self._n_rows: int = None
        self._n_cols: int = None
        # transformation
        self.T: Affine = None
        self.T_inv: Affine = None
        self.pixel_size: List[float] = None
        self.crs: rasterio.crs.CRS = None
        # tiff file info
        self.tiff_file: str = None

    def get_data(self, band=1) -> np.ndarray:
        out = self._data.get(band, None)
        if out is not None:
            out = out.copy()
        return out

    def set_data(self, data, band=1):
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        if self._is_shape_consistent({band: data}):
            self._data[band] = data
            self._n_rows, self._n_cols = data.shape
        else:
            logging.warning("Cant set data: Data shape not consistent")

    def _is_shape_consistent(self, data_dict: dict):
        _n_rows = self._n_rows
        _n_cols = self._n_cols
        for k, v in data_dict.items():
            height, width = v.shape
            if _n_rows is None or _n_cols is None:
                _n_rows = height
                _n_cols = width
            else:
                if (_n_rows != height) or (_n_cols != width):
                    return False
        return True

    def set_transform(self, bl_bound, tr_bound, pixel_size, crs_epsg):
        if self._editable:
            self.pixel_size = np.array(pixel_size).tolist()

            self.T: Affine = Affine(self.pixel_size[0], 0.0, bl_bound[0],
                                    0.0, -1 * self.pixel_size[1], tr_bound[1])
            self.T_inv: Affine = ~self.T

            self.crs = rasterio.crs.CRS.from_epsg(crs_epsg)
        else:
            logging.warning("Can't edit this RasterData")

    def set_transform_from(self, target_data):
        if self._editable:
            self.pixel_size = target_data.pixel_size
            self.T: Affine = target_data.T
            self.T_inv = target_data.T_inv
            self.crs = target_data.crs
        else:
            logging.warning("Can't edit this RasterData")


    @staticmethod
    def cal_dsm_shape(bl_bound, tr_bound, pixel_size):
        """
        Given bounding box, calculate DSM raster n_rows and n_cols.
        DSM will not exceed the bounding box (i.e. round down)
        Args:
            bl_bound: bottom-left bounding point
            tr_bound: top-right bounding point
            pixel_size: DSM pixel size

        Returns: n_rows, n_cols

        """
        bl_bound = np.array(bl_bound).astype(np.float64)
        tr_bound = np.array(tr_bound).astype(np.float64)
        pixel_size = np.array(pixel_size).astype(np.float64)
        _n_rows = math.floor((tr_bound[1] - bl_bound[1]) / pixel_size[1])
        _n_cols = math.floor((tr_bound[0] - bl_bound[0]) / pixel_size[0])
        return _n_rows, _n_cols

    def is_complete(self):
        flag = (len(self._data) > 0) \
               & self._is_shape_consistent(self._data) \
               & (self._n_rows is not None) \
               & (self._n_cols is not None)\
               & (self.T is not None)\
               & (self.T_inv is not None) \
               & (self.crs is not None)
        return flag

    def query_value(self, x, y, band=1):
        # col, row = np.floor(self.T_inv * np.array([x, y]).transpose()).astype(int)
        col, row = self.query_col_row(x, y)
        if self.is_in(col, row, band):
            pix = self._data[band][row, col]
        else:
            pix = None
        return pix

    def is_in(self, col, row, band):
        shape = self._data[band].shape
        if isinstance(col, (int, np.int_, np.int16, np.int32)) and isinstance(row, (int, np.int_, np.int16, np.int32)):
            flag = (0 <= row) and (row < shape[0]) and (0 <= col) and (col < shape[1])
            return flag
        elif isinstance(col, np.ndarray) and isinstance(row, np.ndarray):
            is_in_arr = np.where(((0 <= row) & (row < shape[0]) & (0 <= col) & (col < shape[1])), 1, 0).astype(bool)
            return is_in_arr
        else:
            raise TypeError("col and row should both be int or np.ndarray")

    def query_col_row(self, x, y):
        cols, rows = self.query_col_rows(np.array([[x, y]]))
        return cols[0], rows[0]

    def query_col_rows(self, xy_arr: np.ndarray):
        cols, rows = np.floor(self.T_inv * xy_arr.transpose()).astype(int)
        return cols, rows

    def query_values(self, xy_arr: np.ndarray, band=1, outer_value=-99999):
        cols, rows = self.query_col_rows(xy_arr)
        tiff_data = self._data[band]
        is_in = self.is_in(cols, rows, band)
        rows = rows[is_in]
        cols = cols[is_in]
        pixels = np.empty(xy_arr.shape[0]).astype(tiff_data.dtype)
        pixels[is_in] = np.array([tiff_data[rows[i], cols[i]] for i in range(len(rows))])
        pixels[~is_in] = outer_value
        return pixels

    def query_value_3d_points(self, points, band=1, outer_value=0):
        if 0 == points.shape[0]:
            return np.empty(0)
        xy_arr = points[:, 0:2]
        pixes = self.query_values(xy_arr, band, outer_value)
        return pixes


class RasterReader(RasterData):
    def __init__(self, tiff_file):
        super().__init__()
        self.tiff_file = tiff_file
        self.dataset_reader: rasterio.DatasetReader = rasterio.open(tiff_file)
        self.from_reader(self.dataset_reader)

    def from_reader(self, tiff_obj: rasterio.DatasetReader):
        if self._editable:
            self._data = {i: tiff_obj.read(i) for i in range(1, tiff_obj.count + 1)}
            self.T: Affine = tiff_obj.transform
            self.T_inv = ~self.T
            self.pixel_size = [self.T.a, -self.T.e]
            self.crs = self.dataset_reader.crs
            self._editable = False
        else:
            logging.warning("Can't edit this dataset (from_reader called)")


class RasterWriter(RasterData):
    dataset_writer: rasterio.io.DatasetWriter

    def __init__(self, raster_data: RasterData, dtypes=('float32')):
        super().__init__()
        super().__dict__.update(raster_data.__dict__)
        self.dtypes = dtypes

    def write_to_file(self, filename: str):
        if self.is_complete():
            n_channel = len(self._data)
            self.tiff_file = filename
            self._open_file(filename)
            for c in range(1, n_channel + 1):
                self.dataset_writer.write(self._data[c].astype(np.float32), c)
            self._close_file()
            return True
        else:
            logging.warning("RasterData is not complete, can't write to tiff file")
            return False

    def _open_file(self, filename):
        self.dataset_writer = rasterio.open(
            filename,
            'w+',
            driver='GTiff',
            height=self._n_rows,
            width=self._n_cols,
            count=len(self._data),
            dtype=self.dtypes,
            crs=self.crs,
            transform=self.T
        )

    def _close_file(self):
        self.dataset_writer.close()
