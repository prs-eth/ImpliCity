# encoding: utf-8
# Author: Bingxin Ke
# Created: 2021/11/18

# from .pc_utils import *
from .PCTransforms import PointCloudNormalizer, PointCloudSubsampler, ShiftPoints
from .crop_pc import crop_pc_3d, crop_pc_2d, crop_pc_2d_index
from .pc_io import save_pc_to_ply, load_pc, load_las_as_numpy


if __name__ == '__main__':
    # test module

    import numpy as np
    import torch

    # test load pc
    folder1 = "/scratch2/bingxin/IPA/Data/ZUR1/Point_Clouds/"
    # folder2 = "/scratch2/bingxin/IPA/Data/ZUR1/Point_Clouds_npy/"
    # for file in os.listdir(folder1):
    #     full_path = os.path.join(folder1, file)
    #     points = load_pc(full_path)

    # test crop pc
    mesh_path = "/scratch2/bingxin/IPA/Data/ZUR1/Ground_Truth_3D/merged_dach_wand_terrain.obj"
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    pcd = mesh.sample_points_uniformly(number_of_points=10000000)
    pts = np.asarray(pcd.points)
    p1 = np.array([463328., 5248140.])
    p2 = np.array([463412., 5248224.])
    # idx = crop_pc_2d_index(pts, p1, p2)
    out_points, idx = crop_pc_2d(pts, p1, p2)


    pts = torch.from_numpy(pts)
    out_points2, idx2 = crop_pc_2d(pts, p1, p2)

    diff = out_points2 - out_points
    save_pc_to_ply("/scratch2/bingxin/IPA/Data/tempData/cropped_pc2.ply", out_points)
