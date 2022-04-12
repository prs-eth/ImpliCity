# encoding: utf-8
# Author: Bingxin Ke
# Created: 2021/10/28
from collections import defaultdict

from src.io import RasterReader, RasterWriter
from src.utils.libraster import dilate_mask
import numpy as np
import os
from typing import Dict
from rasterio.transform import Affine
from tabulate import tabulate
from datetime import datetime

DEFAULT_VALUE = -9999


class DSMEvaluator:
    def __init__(self, gt_dsm_path: str, gt_mask_path: str = None, other_mask_path_dict: Dict[str, str] = None):
        # self.gt_dsm: np.ndarray = gt_dsm
        self._gt_dsm_reader = RasterReader(gt_dsm_path)
        self.gt_dsm = self._gt_dsm_reader.get_data()
        # load gt mask
        if gt_mask_path is not None:
            self._gt_mask_reader = RasterReader(gt_mask_path)
            self.gt_mask = self._gt_mask_reader.get_data().astype(np.bool)
        else:
            self.gt_mask = np.ones(self.gt_dsm.shape)
        # load other masks
        if len(other_mask_path_dict) > 0:
            self.other_mask: Dict[str, np.ndarray] = {key: RasterReader(path).get_data().astype(np.bool)
                                                      for key, path in other_mask_path_dict.items()}
            if 'building' in self.other_mask.keys():
                # dilate building mask by 2 pixels
                self.other_mask['building'] = dilate_mask(self.other_mask['building'], iterations=2)
                # terrain
                self.other_mask['terrain'] = ~self.other_mask['building']
                if 'water' in self.other_mask.keys():
                    self.other_mask['terrain_wo_water'] = self.other_mask['terrain'] & ~self.other_mask['water']
                if 'forest' in self.other_mask.keys():
                    self.other_mask['terrain_wo_forest'] = self.other_mask['terrain'] & ~self.other_mask['forest']
        else:
            self.other_mask = None

    def eval(self, target_dsm: np.ndarray, T: Affine, save_to: str = None):
        # gt_dsm = self.gt_dsm
        target_shape = target_dsm.shape
        # T_inv = ~T

        # clip gt dsm and masks
        tl_bound = T * np.array([0, 0])
        # br_bound = T * np.array([target_shape[1], target_shape[0]])
        # _edge_length = (np.array(br_bound) - np.array(tl_bound)) * np.array([1, -1])
        # area = _edge_length[0] * _edge_length[1]
        l_col, t_row = np.floor(self._gt_dsm_reader.T_inv * tl_bound).astype(int)
        gt_dsm_clip_arr = self.gt_dsm[t_row:t_row + target_shape[0], l_col:l_col + target_shape[1]]
        gt_mask_clip_arr = self.gt_mask[t_row:t_row + target_shape[0], l_col:l_col + target_shape[1]]
        # print(np.where(np.isnan(target_dsm) == True))
        # print('gt_mask_clip_arr', gt_mask_clip_arr)
        # print(gt_dsm_clip_arr.shape)
        # print(gt_dsm_clip_arr[gt_mask_clip_arr].shape)

        # original residual
        residuals_arr = target_dsm - gt_dsm_clip_arr

        # output_dic statistics
        output_statistics = defaultdict()

        # Overall residual
        # apply gt mask
        residuals_arr_gt = residuals_arr[gt_mask_clip_arr]
        # remove nan values
        residuals_arr_gt = residuals_arr_gt[np.where(np.isnan(residuals_arr_gt) == False)]
        # statistics
        _statistics = self.calculate_statistics(residuals_arr_gt)
        output_statistics['overall'] = _statistics

        # Different land types
        if self.other_mask is not None:
            for land_type, mask in self.other_mask.items():
                # clip mask
                _mask_clip = mask[t_row:t_row + target_shape[0], l_col:l_col + target_shape[1]]
                # operation 'and' with gt mask
                gt_land_mask = gt_mask_clip_arr & _mask_clip
                masked_residual = residuals_arr[gt_land_mask]
                # remove nan values
                masked_residual = masked_residual[np.where(np.isnan(masked_residual) == False)]
                _statistics = self.calculate_statistics(masked_residual)
                output_statistics[land_type] = _statistics

        # Residual dsm
        diff_arr = residuals_arr * gt_mask_clip_arr
        diff_arr[~gt_mask_clip_arr] = np.nan

        return output_statistics, diff_arr

    @staticmethod
    def calculate_statistics(residual: np.ndarray):
        if residual.shape[0] > 0:
            residual_abs = np.abs(residual)
            output_dic = defaultdict(float)
            output_dic['max'] = np.max(residual)
            output_dic['min'] = np.min(residual)
            output_dic['MAE'] = np.mean(residual_abs)  # mean absolute error
            output_dic['RMSE'] = np.sqrt(np.mean(residual**2))
            output_dic['abs_median'] = np.median(residual_abs)
            output_dic['median'] = np.median(residual)
            output_dic['n_pixel'] = residual.size

            # Normalized median absolute deviation
            output_dic['NMAD'] = 1.4826 * np.median(np.abs(residual - output_dic['abs_median']))
        else:
            output_dic = {'max': None, 'min': None, 'MAE': None, 'RMSE': None, 'abs_median': None, 'median': None, 'n_pixel': None, 'NMAD': None}
        return output_dic


def print_statistics(statistic_dic: Dict, title: str, save_to: str = None, include_time=True):
    head_line_keys = {  # head line: statistics keys
        'MAE[m]': 'MAE',
        'RMSE[m]': 'RMSE',
        'MedAE[m]': 'abs_median',
        'Max[m]': 'max',
        'Min[m]': 'min',
        'Median[m]': 'median',
        'NMAD[m]': 'NMAD',
        '#Pixels': 'n_pixel'
    }
    output_str = "DSM Evaluation"
    output_str += '\t' * 3 + 'created: ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n\n'
    # title
    output_str += title + '\n\n'
    output_str += "Performance Evaluation \n"
    output_str += "=" * 20 + '\n'
    # Table
    head_line = list(head_line_keys.keys())
    content = []
    for mask_type, dic in statistic_dic.items():
        line = [mask_type.capitalize()]
        for metric in head_line:
            key = head_line_keys[metric]
            line.append(dic[key])
        content.append(line)

    head_line.insert(0, 'Type')
    output_str += tabulate(content, headers=head_line, tablefmt="simple", floatfmt=".4f") + '\n'

    # Description
    output_str += '-' * 20 + '\n'
    output_str += """ Metrics:
        MAE: Mean Absolute residual Error
        RMSE: Root Mean Square Error
        MedAE: Median Absolute Error
        Max: Maximum value
        Min: Minimum value
        Median: Median value
        NMAD: Normalised Median Absolute Deviation
        #pixels:  Number of pixels
    """

    if save_to is not None:
        with open(save_to, 'w+') as f:
            f.write(output_str)

    return output_str
