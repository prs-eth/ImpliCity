# encoding: utf-8
# Author: Bingxin Ke
# Created: 2021/9/28


import logging
from typing import Dict, Union

import numpy as np

from src.utils.libcoord.coord_transform import normalize_pc, invert_normalize_pc


class PointCloudSubsampler(object):
    """ Point cloud subsampling transformation class.

    A transformer to subsample the point cloud data.

    Args:
        N (int): number of points in output point cloud
        allow_repeat (bool): if size of input point cloud < N, allow to use repeat number
    """

    def __init__(self, N: int, allow_repeat=False):
        self.N = N
        self._allow_repeat = allow_repeat

    def __call__(self, data: Union[Dict, np.ndarray]):
        """ Calls the transformation.

        Args:
            data (dict or array)
        Returns: same format as data
        """
        # check arrays have same dim-0 length if it's Dict
        data_num: int = -1
        if isinstance(data, dict):
            for key, arr in data.items():
                if data_num < 0:
                    data_num = arr.shape[0]  # init value
                assert arr.shape[0] == data_num, f"Size not consistent in data: {arr.shape[0]} != {data_num}"
        elif isinstance(data, np.ndarray):
            data_num = data.shape[0]
        else:
            raise AssertionError("Unknown data type. Should be array or Dict")
        if data_num < self.N:
            logging.warning(f"data_num({data_num}) < self.N ({self.N}):")
            if self._allow_repeat:
                random_inx = np.random.randint(0, data_num, self.N)
            else:
                # if not allow repeat, no subsample
                n_selected = min(data_num, self.N)
                random_inx = np.random.choice(data_num, n_selected, replace=False)  # select without repeat
        else:
            random_inx = np.random.choice(data_num, self.N, replace=False)  # select without repeat

        output = data.copy()
        if isinstance(output, dict):
            for key, arr in output.items():
                output[key] = arr[random_inx]
        elif isinstance(output, np.ndarray):
            output = output[random_inx]
        return output


# class PointCloudScaler(object):
#     """
#         Scaling (normalizing) point cloud.
#         data * scale + shift
#     """
#
#     def __init__(self, scale_factor_3d: np.ndarray, shift_3d: np.ndarray = np.array([0, 0, 0])):
#         assert 3 == len(scale_factor_3d.reshape(-1)), "Wrong dimension for scale factors"
#         self.scale_factor_3d = scale_factor_3d.reshape(3)
#         self.shift_3d = shift_3d.reshape(3)
#
#     def __call__(self, data: Union[Dict, np.ndarray]):
#         if isinstance(data, Dict):
#             out = {}
#             for key, value in data.items():
#                 out[key] = value * self.scale_factor_3d + self.shift_3d
#         elif isinstance(data, np.ndarray):
#             out = data * self.scale_factor_3d + self.shift_3d
#         else:
#             raise TypeError("Unknown data type")
#         return out
#
#     def inverse(self, data: Union[Dict, np.ndarray]):
#         if isinstance(data, Dict):
#             out = {}
#             for key, value in data.items():
#                 out[key] = (value - self.shift_3d) / self.scale_factor_3d
#         elif isinstance(data, np.ndarray):
#             out = (data - self.shift_3d) / self.scale_factor_3d
#         else:
#             raise TypeError("Unknown data type")
#         return out


class PointCloudNormalizer(object):
    def __init__(self, scales, center_shift):
        self.scales = scales
        self.center_shift = center_shift

    def __call__(self, points):
        return normalize_pc(points, self.scales, self.center_shift)

    def inverse(self, points):
        return invert_normalize_pc(points, self.scales, self.center_shift)


class ShiftPoints(object):
    def __init__(self, shift_3d: np.ndarray):
        self.shift_3d = np.array(shift_3d).reshape(3)

    def __call__(self, points, plane='xy'):
        if 'xy' == plane:
            xy = points[:, :, [0, 1]]
            xy[:, :, 0] = xy[:, :, 0] + self.shift_3d[0]
            xy[:, :, 1] = xy[:, :, 1] + self.shift_3d[1]
            return xy
        # # f there are outliers out of the range
        # if xy_new.max() >= 1:
        #     xy_new[xy_new >= 1] = 1 - 10e-6
        # if xy_new.min() < 0:
        #     xy_new[xy_new < 0] = 0.0
        # return xy_new

        # if isinstance(points, np.ndarray):
        #     return points + self.shift_3d
        # elif isinstance(points, torch.Tensor):
        #     return points + torch.from_numpy(self.shift_3d)
        # else:
        #     raise TypeError

    def inverse(self, points, plane='xy'):
        if 'xy' == plane:
            xy = points[:, :, [0, 1]]
            xy[:, :, 0] = xy[:, :, 0] - self.shift_3d[0]
            xy[:, :, 1] = xy[:, :, 1] - self.shift_3d[1]
            return xy
        # if isinstance(points, np.ndarray):
        #     return points - self.shift_3d
        # elif isinstance(points, torch.Tensor):
        #     return points - torch.from_numpy(self.shift_3d)
        # else:
        #     raise TypeError



if __name__ == '__main__':
    # test subsample
    sampler = PointCloudSubsampler(5)
    dummy_array = np.random.randint(0, 50, 20)
    print(f"dummy_array: {dummy_array}")
    sub_arr = sampler(dummy_array)
    print(f"sub_arr: {sub_arr}")
    print(f"dummy_array: {dummy_array}")

    dummy_dic = {'1': dummy_array, 'None': dummy_array}
    sub_dic = sampler(dummy_dic)
    print(f"dummy_dic: {dummy_dic}")
    print(f"sub_dic: {sub_dic}")
