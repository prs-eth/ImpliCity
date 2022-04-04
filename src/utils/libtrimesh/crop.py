# encoding: utf-8
# Author: Bingxin Ke
# Created: 2021/9/28


import numpy as np
import trimesh
from trimesh.base import Trimesh
from trimesh.points import PointCloud
import laspy
import math


def crop_mesh_2d(mesh: Trimesh, p_min: np.ndarray, p_max: np.ndarray) -> Trimesh:
    """
    Crop mesh with 2D rectangle axis-aligned box.
        - Attention: normals of cropped mesh are missing
    Args:
        mesh: Trimesh mesh type
        p_min: 2D array, bottom-left point of crop box
        p_max: 2D array, top-right point of crop box

    Returns: cropped mesh

    """
    bbox = mesh.bounding_box
    SAFE_PADDING = 10
    z_min = math.floor(bbox.primitive.transform[2, 3] - bbox.primitive.extents[2] / 2) - SAFE_PADDING
    z_max = math.ceil(bbox.primitive.transform[2, 3] + bbox.primitive.extents[2] / 2) + SAFE_PADDING
    p_min = np.concatenate([p_min, [z_min, 0]])
    p_max = np.concatenate([p_max, [z_max, 0]])
    transform = np.eye(4) + np.concatenate([np.zeros((4, 3)), (p_min + p_max).reshape(4, 1) / 2], 1)
    extents = (p_max - p_min)[:3]
    box = trimesh.creation.box(extents=extents, transform=transform)
    new_mesh = mesh.slice_plane(box.facets_origin, -box.facets_normal)
    return new_mesh


if __name__ == '__main__':
    # test crop mesh
    gt_mesh_path = "/scratch2/bingxin/IPA/Data/ZUR1/Ground_Truth_3D/merged_buildings.obj"
    mesh: Trimesh = trimesh.load_mesh(gt_mesh_path)
    p1 = np.array([464320.0, 5249040.0])
    p2 = np.array([464420.0, 5249140.0])
    cropped = crop_mesh_2d(mesh, p1, p2)
    export_file = "/scratch2/bingxin/IPA/Data/tempData/trimesh_crop_test.obj"
    cropped.export(export_file)
