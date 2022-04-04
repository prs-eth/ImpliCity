# encoding: utf-8
# Author: Bingxin Ke
# Created: 2021/10/4

"""
    Homogeneous coordinate transformation

"""
from typing import Union

import numpy as np
import open3d as o3d
import torch


def points_to_transform(p1, p2):
    raise NotImplemented


def extent_transform_to_points(extents, transform):
    # _p1 = np.array([0, 0, 0, 1]).reshape((4, 1))
    _half_extents = extents / 2.0
    _p1 = np.concatenate([_half_extents * -1, [1]]).reshape((4, 1)) * -1
    _p2 = np.concatenate([_half_extents, [1]]).reshape((4, 1))

    _p1 = np.matmul(np.array(transform), _p1).squeeze()
    _p2 = np.matmul(np.array(transform), _p2).squeeze()

    _p1 = _p1 / _p1[3]
    _p2 = _p2 / _p2[3]
    return _p1[:3], _p2[:3]


def normalize_pc(points: Union[np.ndarray, o3d.geometry.PointCloud], scales, center_shift):
    """
        Normalize a point cloud: x_norm = (x_ori - center_shift) / scale
    Args:
        points: input point cloud
        scales: scale of source data
        center_shift: shift of original center (in original crs)

    Returns:

    """
    if isinstance(points, o3d.geometry.PointCloud):
        points = np.asarray(points.points)
    norm_pc = (points - center_shift) / scales
    return norm_pc


def invert_normalize_pc(points: Union[np.ndarray, o3d.geometry.PointCloud], scales, center_shift):
    """
        Invert normalization of a point cloud: x_ori = scales * x_norm + center_shift
    Args:
        points:
        scales:
        center_shift:

    Returns:

    """
    if isinstance(points, o3d.geometry.PointCloud):
        points = np.asarray(points.points)
    ori_pc = points * scales + center_shift
    return ori_pc


def apply_transform(p, M):
    if isinstance(p, np.ndarray):
        p = p.reshape((-1, 3))
        p = np.concatenate([p, np.ones((p.shape[0], 1))], 1).transpose()
        p2 = np.matmul(M, p).squeeze()
        p2 = p2 / p2[3, :]
        return p2[0:3, :].transpose()
    elif isinstance(p, torch.Tensor):
        p = p.reshape((-1, 3))
        p = torch.cat([p, torch.ones((p.shape[0], 1)).to(p.device)], 1).transpose(0, 1)
        p2 = torch.matmul(M.double(), p.double()).squeeze()
        p2 = p2 / p2[3, :]
        return p2[0:3, :].transpose(0, 1).to(p.dtype)
    else:
        raise TypeError


def invert_transform(M):
    if isinstance(M, np.ndarray):
        return np.linalg.inv(M)
    elif isinstance(M, torch.Tensor):
        return torch.inverse(M.double()).to(M.dtype)
    else:
        raise TypeError


def stack_transforms(M_ls):
    """
    M_out = M_ls[0] * M_ls[1] * M_ls[2] * ...
    Args:
        M_ls:

    Returns:

    """
    M_out = M_ls[0]
    if isinstance(M_out, np.ndarray):
        for M in M_ls[1:]:
            M_out = np.matmul(M_out, M)
        return M_out
    elif isinstance(M_out, torch.Tensor):
        for M in M_ls[1:]:
            M_out = torch.matmul(M_out, M)
        return M_out
    else:
        raise TypeError
