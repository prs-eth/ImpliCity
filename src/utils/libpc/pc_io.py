# encoding: utf-8
# Author: Bingxin Ke
# Created: 2021/11/18
from typing import Union, Tuple

import laspy
import numpy as np
import open3d as o3d
import torch


def load_pc(pc_path: str) -> np.ndarray:
    extension = pc_path.split('.')[-1].lower()
    points: np.ndarray
    if 'las' == extension:
        points = load_las_as_numpy(pc_path)
    elif 'npy' == extension:
        points = np.load(pc_path)
    elif extension in ['xyz', 'ply', 'pcd', 'pts', 'xyzn', 'xyzrgb']:
        pcd = o3d.io.read_point_cloud(pc_path)
        points = np.asarray(pcd.points)
    else:
        raise TypeError(f"Unknown type: {extension}")
    return points


def load_las_as_numpy(las_path: str) -> np.ndarray:
    """
    Load .las point cloud and convert into numpy array
    This one is slow, because laspy returns a list of tuple, which can't be directly transformed into numpy array
    Args:
        las_path: full path to las file

    Returns:

    """
    with laspy.open(las_path) as f:
        _las = f.read()
    x = np.array(_las.x).reshape((-1, 1))
    y = np.array(_las.y).reshape((-1, 1))
    z = np.array(_las.z).reshape((-1, 1))
    points = np.concatenate([x, y, z], 1)
    # points = _las.points.array
    # points = np.asarray(points.tolist())[:, 0:3].astype(np.float)
    return points


def save_pc_to_ply(pc_path: str, points: Union[np.ndarray, o3d.geometry.PointCloud], colors: np.ndarray = None):
    if isinstance(points, o3d.geometry.PointCloud):
        pcd = points
    else:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    pc_path = pc_path + ".ply" if ".ply" != pc_path[-4:].lower() else pc_path
    o3d.io.write_point_cloud(pc_path, pcd)
