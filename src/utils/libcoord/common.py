# encoding: utf-8
# Author: Bingxin Ke
# Created: 2021/10/6
import math

import numpy as np
import torch


def coordinate2index(x, reso, coord_type='2d'):
    """ Generate grid index of points

    Args:
        x (tensor): points (normalized to [0, 1])
        reso (int): defined resolution
        coord_type (str): coordinate type
    """
    x = (x * reso).long()
    if coord_type == '2d':  # plane
        index = x[:, :, 0] + reso * x[:, :, 1]  # [B, N, 1]
    index = index[:, None, :]  # [B, 1, N]
    return index


def normalize_coordinate(p, padding=0, plane='xz', scale=1.0):
    """ Normalize coordinate to [0, 1] for unit cube experiments

    Args:
        p (tensor): point
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        plane (str): plane feature type, ['xz', 'xy', 'yz']
        scale: normalize scale
    """
    raise  NotImplemented
    if 'xz' == plane:
        xy = p[:, :, [0, 2]]
    elif 'xy' == plane:
        xy = p[:, :, [0, 1]]
    else:
        xy = p[:, :, [1, 2]]

    # xy_new = xy / (1 + padding + 10e-6)  # (-0.5, 0.5)
    # xy_new = xy / (1 + padding + 10e-6) / 2  # (-0.5, 0.5)  # TODO my scale [-1, 1] -> [-0.5, 0.5]
    xy_new = xy * scale  # (-0.5, 0.5)  # TODO my scale [-1, 1] -> [-0.5, 0.5]
    xy_new = xy_new + 0.5  # range (0, 1)

    # f there are outliers out of the range
    if xy_new.max() >= 1:
        xy_new[xy_new >= 1] = 1 - 10e-6
    if xy_new.min() < 0:
        xy_new[xy_new < 0] = 0.0
    return xy_new


def normalize_3d_coordinate(p, padding=0):
    """ Normalize coordinate to [0, 1] for unit cube experiments.
        Corresponds to our 3D model

    Args:
        p (tensor): point
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    """
    raise NotImplemented

    p_nor = p / (1 + padding + 10e-4)  # (-0.5, 0.5)
    p_nor = p_nor + 0.5  # range (0, 1)
    # f there are outliers out of the range
    if p_nor.max() >= 1:
        p_nor[p_nor >= 1] = 1 - 10e-4
    if p_nor.min() < 0:
        p_nor[p_nor < 0] = 0.0
    return p_nor


class map2local(object):
    """ Add new keys to the given input

    Args:
        s (float): the defined voxel size
        pos_encoding (str): method for the positional encoding, linear|sin_cos
    """

    def __init__(self, s, pos_encoding='linear'):
        super().__init__()
        self.s = s
        self.pe = positional_encoding(basis_function=pos_encoding)

    def __call__(self, p):
        p = torch.remainder(p, self.s) / self.s  # always possitive
        # p = torch.fmod(p, self.s) / self.s # same sign as input p!
        p = self.pe(p)
        return p


class positional_encoding(object):
    ''' Positional Encoding (presented in NeRF)

    Args:
        basis_function (str): basis function
    '''

    def __init__(self, basis_function='sin_cos'):
        super().__init__()
        self.func = basis_function

        L = 10
        freq_bands = 2. ** (np.linspace(0, L - 1, L))
        self.freq_bands = freq_bands * math.pi

    def __call__(self, p):
        if self.func == 'sin_cos':
            out = []
            p = 2.0 * p - 1.0  # chagne to the range [-1, 1]
            for freq in self.freq_bands:
                out.append(torch.sin(freq * p))
                out.append(torch.cos(freq * p))
            p = torch.cat(out, dim=2)
        return p


def make_3d_grid(bb_min, bb_max, shape):
    ''' Makes a 3D grid.

    Args:
        bb_min (tuple): bounding box minimum
        bb_max (tuple): bounding box maximum
        shape (tuple): output shape
    '''
    size = shape[0] * shape[1] * shape[2]

    pxs = torch.linspace(bb_min[0], bb_max[0], int(shape[0]))
    pys = torch.linspace(bb_min[1], bb_max[1], int(shape[1]))
    pzs = torch.linspace(bb_min[2], bb_max[2], int(shape[2]))

    pxs = pxs.view(-1, 1, 1).expand(*shape).contiguous().view(size)
    pys = pys.view(1, -1, 1).expand(*shape).contiguous().view(size)
    pzs = pzs.view(1, 1, -1).expand(*shape).contiguous().view(size)
    p = torch.stack([pxs, pys, pzs], dim=1)

    return p
