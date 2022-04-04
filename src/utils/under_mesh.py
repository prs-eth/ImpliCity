# encoding: utf-8
# Author: Bingxin Ke
# Created: 2021/10/5

import numpy as np
import pymesh
from pymesh import Mesh as pyMesh


def check_under_mesh(mesh_pymesh: pyMesh, points):
    dist, _, closest_p = pymesh.distance_to_mesh(mesh_pymesh, points)

    points = np.array(points)
    is_underground = np.sign(closest_p[:, 2] - np.array(points[:, 2]))
    is_underground[is_underground < 0] = 0
    return is_underground.astype(bool)
