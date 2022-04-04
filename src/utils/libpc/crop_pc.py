# encoding: utf-8
# Author: Bingxin Ke
# Created: 2021/11/18
from typing import Union, Tuple

import laspy
import numpy as np
import open3d as o3d
import torch


def crop_pc_2d_index(points: Union[np.ndarray, torch.Tensor], p_min, p_max):
    if isinstance(points, np.ndarray):
        points = points.squeeze()
        index = np.where((points[:, 0] > p_min[0]) & (points[:, 0] < p_max[0]) &
                         (points[:, 1] > p_min[1]) & (points[:, 1] < p_max[1]))[0]
        return index
    elif isinstance(points, torch.Tensor):
        points = points.squeeze()
        index = torch.where((points[:, 0] > p_min[0]) & (points[:, 0] < p_max[0]) &
                            (points[:, 1] > p_min[1]) & (points[:, 1] < p_max[1]),
                            # torch.ones((points.shape[0], 1)),
                            # torch.zeros((points.shape[0], 1))
                            # 1,
                            # 0
                            )[0]
        return index
    else:
        raise NotImplemented


def crop_pc_2d(points: Union[np.ndarray, torch.Tensor], p_min, p_max) -> Union[
    Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Crop point cloud according to x-y bounding box
    Args:
        points: input points
        p_min: bottom-left point of bounding box
        p_max: top-right point of bounding box

    Returns:

    """
    if isinstance(points, np.ndarray) or isinstance(points, torch.Tensor):
        index = crop_pc_2d_index(points, p_min, p_max)
        new_points = points[index]
        return new_points, index
    else:
        raise NotImplemented
    # logging.debug(f"type(points) = {type(points)}")
    # if isinstance(points, o3d.geometry.PointCloud):
    #     pcd = points
    #     _temp_points = np.asarray(pcd.points)
    #     z_min = _temp_points[:, 2].min()
    #     z_max = _temp_points[:, 2].max()
    # else:
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(points)
    #     z_min = points[:, 2].min()
    #     z_max = points[:, 2].max()
    # p_min = np.array([p_min[0], p_min[1], z_min - safe_padding_z])
    # p_max = np.array([p_max[0], p_max[1], z_max + safe_padding_z])
    # # print(f"z_min = {z_min}, z_max = {z_max}")
    # # print(f"p_min = {p_min}, p_max = {p_max}")
    # bbox = o3d.geometry.AxisAlignedBoundingBox(p_min, p_max)
    # cropped_pcd = pcd.crop(bbox)
    # return np.asarray(cropped_pcd.points), cropped_pcd


def crop_pc_3d(points: Union[np.ndarray, o3d.geometry.PointCloud], p_min, p_max) -> Tuple[
    np.ndarray, o3d.geometry.PointCloud]:
    if isinstance(points, o3d.geometry.PointCloud):
        pcd = points
    else:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
    bbox = o3d.geometry.AxisAlignedBoundingBox(p_min, p_max)
    cropped_pcd = pcd.crop(bbox)
    return np.asarray(cropped_pcd.points).copy(), cropped_pcd

