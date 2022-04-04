from typing import Union, List

import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_max

from src.utils.libcoord.common import coordinate2index
from src.model.block import ResnetBlockFC
from src.model.encoder.unet import UNet
from src.model.encoder.unet3d import UNet3D


class LocalPoolPointnet(nn.Module):
    ''' PointNet-based encoder network with ResNet block for each point.
        Number of input points are fixed.
    
    Args:
        feature_dim (int): output feature dimension
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
        scatter_type (str): feature aggregation when doing local pooling
        unet (bool): weather to use U-Net
        unet_kwargs (str): U-Net parameters
        unet3d (bool): weather to use 3D U-Net
        unet3d_kwargs (str): 3D U-Net parameters
        plane_resolution (int): defined resolution for plane feature
        plane_type (List[str]): feature type, 'xz' - 1-plane, ['xz', 'xy', 'yz'] - 3-plane, ['grid'] - 3D grid volume
        n_blocks (int): number of block ResNetBlockFC layers
    '''

    def __init__(self, feature_dim=128, dim=3, hidden_dim=128, scatter_type='max',
                 unet=False, unet_kwargs=None, unet3d=False, unet3d_kwargs=None,
                 plane_resolution=None, plane_type=None, n_blocks=5):
        super().__init__()
        self.c_dim = feature_dim

        self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
        self.blocks = nn.ModuleList([
            ResnetBlockFC(2 * hidden_dim, hidden_dim) for i in range(n_blocks)
        ])
        self.fc_c = nn.Linear(hidden_dim, feature_dim)

        self.actvn = nn.ReLU()
        self.hidden_dim = hidden_dim

        self.unet: Union[UNet, bool]
        if unet:
            self.unet = UNet(feature_dim, in_channels=feature_dim, **unet_kwargs)
        else:
            self.unet = None

        if unet3d:
            self.unet3d = UNet3D(**unet3d_kwargs)
        else:
            self.unet3d = None

        self.reso_plane: int = plane_resolution
        # self.reso_grid: int = grid_resolution
        self.plane_type: List[str] = plane_type if plane_type is not None else ['xy']
        # self.padding: float = padding

        if scatter_type == 'max':
            self.scatter = scatter_max
        elif scatter_type == 'mean':
            self.scatter = scatter_mean
        else:
            raise ValueError('incorrect scatter type')

    def forward(self, inputs: torch.Tensor, **kwargs):
        """

        Args:
            inputs: input point cloud, shape=(b, n, 3), should be normalized to [0, 1]
            **kwargs:

        Returns:

        """
        # acquire the index for each point
        coord = {}
        index = {}
        if 'xy' in self.plane_type:
            coord['xy'] = inputs.clone()[:, :, [0, 1]]
            index['xy'] = coordinate2index(coord['xy'], self.reso_plane)

        # inputs = inputs.float()
        net = self.fc_pos(inputs)

        net = self.blocks[0](net)  # hidden_dim
        for block in self.blocks[1:]:
            pooled = self.pool_local(coord, index, net)  # hidden_dim
            net = torch.cat([net, pooled], dim=2)  # 2 * hidden_dim
            net = block(net)  # hidden_dim

        # TODO activation?
        net = self.actvn(net)
        c = self.fc_c(net)

        fea = {}
        if 'xy' in self.plane_type:
            fea['xy'] = self.generate_plane_features(inputs, c, plane='xy')

        return fea

    def pool_local(self, xy, index, c):
        bs, fea_dim = c.size(0), c.size(2)
        plane_keys = xy.keys()

        c_out = 0
        for key in plane_keys:
            # scatter plane features from points
            # if key == 'grid':
            #     fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.reso_grid ** 3)
            # else:
            fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.reso_plane ** 2)
            if self.scatter == scatter_max:
                fea = fea[0]
            # gather feature back to points
            fea = fea.gather(dim=2, index=index[key].expand(-1, fea_dim, -1))
            c_out += fea
        return c_out.permute(0, 2, 1)

    def generate_plane_features(self, p, c, plane='xz'):
        # acquire indices of features in plane
        xy = p.clone()[:, :, [0, 1]]

        index = coordinate2index(xy, self.reso_plane)

        # scatter plane features from points
        fea_plane = c.new_zeros(p.size(0), self.c_dim, self.reso_plane ** 2)  # [B, C, reso^2]
        c = c.permute(0, 2, 1)  # B x C x T
        fea_plane = scatter_mean(c, index, out=fea_plane)  # B x C x reso^2
        fea_plane = fea_plane.reshape(p.size(0), self.c_dim, self.reso_plane,
                                      self.reso_plane)  # sparce matrix (B x C x reso x reso)

        # process the plane features with UNet
        if self.unet is not None:
            fea_plane = self.unet(fea_plane)

        return fea_plane


# class PatchLocalPoolPointnet(nn.Module):
#     ''' PointNet-based encoder network with ResNet block.
#         First transform input points to local system based on the given voxel size.
#         Support non-fixed number of point cloud, but need to precompute the index
#
#     Args:
#         feature_dim (int): dimension of latent code c
#         dim (int): input points dimension
#         hidden_dim (int): hidden dimension of the network
#         scatter_type (str): feature aggregation when doing local pooling
#         unet (bool): weather to use U-Net
#         unet_kwargs (str): U-Net parameters
#         unet3d (bool): weather to use 3D U-Net
#         unet3d_kwargs (str): 3D U-Net parameters
#         plane_resolution (int): defined resolution for plane feature
#         grid_resolution (int): defined resolution for grid feature
#         plane_type (str): feature type, 'xz' - 1-plane, ['xz', 'xy', 'yz'] - 3-plane, ['grid'] - 3D grid volume
#         padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
#         n_blocks (int): number of block ResNetBlockFC layers
#         local_coord (bool): whether to use local coordinate
#         pos_encoding (str): method for the positional encoding, linear|sin_cos
#         unit_size (float): defined voxel unit size for local system
#     '''
#
#     def __init__(self, feature_dim=128, dim=3, hidden_dim=128, scatter_type='max',
#                  unet=False, unet_kwargs=None, unet3d=False, unet3d_kwargs=None,
#                  plane_resolution=None, grid_resolution=None, plane_type='xz', padding=0.1, n_blocks=5,
#                  local_coord=False, pos_encoding='linear', unit_size=0.1):
#         super().__init__()
#         self.feature_dim = feature_dim
#
#         self.block = nn.ModuleList([
#             ResnetBlockFC(2 * hidden_dim, hidden_dim) for i in range(n_blocks)
#         ])
#         self.fc_c = nn.Linear(hidden_dim, feature_dim)
#
#         self.actvn = nn.ReLU()
#         self.hidden_dim = hidden_dim
#         self.reso_plane = plane_resolution
#         self.reso_grid = grid_resolution
#         self.plane_type = plane_type
#         self.padding = padding
#
#         if unet:
#             self.unet = UNet(feature_dim, in_channels=feature_dim, **unet_kwargs)
#         else:
#             self.unet = None
#
#         if unet3d:
#             self.unet3d = UNet3D(**unet3d_kwargs)
#         else:
#             self.unet3d = None
#
#         if scatter_type == 'max':
#             self.scatter = scatter_max
#         elif scatter_type == 'mean':
#             self.scatter = scatter_mean
#         else:
#             raise ValueError('incorrect scatter type')
#
#         if local_coord:
#             self.map2local = map2local(unit_size, pos_encoding=pos_encoding)
#         else:
#             self.map2local = None
#
#         if pos_encoding == 'sin_cos':
#             self.fc_pos = nn.Linear(60, 2 * hidden_dim)
#         else:
#             self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
#
#     def generate_plane_features(self, index, c):
#         c = c.permute(0, 2, 1)
#         # scatter plane features from points
#         if index.max() < self.reso_plane ** 2:
#             fea_plane = c.new_zeros(c.size(0), self.feature_dim, self.reso_plane ** 2)
#             fea_plane = scatter_mean(c, index, out=fea_plane)  # B x feature_dim x reso^2
#         else:
#             fea_plane = scatter_mean(c, index)  # B x feature_dim x reso^2
#             if fea_plane.shape[-1] > self.reso_plane ** 2:  # deal with outliers
#                 fea_plane = fea_plane[:, :, :-1]
#
#         fea_plane = fea_plane.reshape(c.size(0), self.feature_dim, self.reso_plane, self.reso_plane)
#
#         # process the plane features with UNet
#         if self.unet is not None:
#             fea_plane = self.unet(fea_plane)
#
#         return fea_plane
#
#     def generate_grid_features(self, index, c):
#         # scatter grid features from points
#         c = c.permute(0, 2, 1)
#         if index.max() < self.reso_grid ** 3:
#             fea_grid = c.new_zeros(c.size(0), self.feature_dim, self.reso_grid ** 3)
#             fea_grid = scatter_mean(c, index, out=fea_grid)  # B x feature_dim x reso^3
#         else:
#             fea_grid = scatter_mean(c, index)  # B x feature_dim x reso^3
#             if fea_grid.shape[-1] > self.reso_grid ** 3:  # deal with outliers
#                 fea_grid = fea_grid[:, :, :-1]
#         fea_grid = fea_grid.reshape(c.size(0), self.feature_dim, self.reso_grid, self.reso_grid, self.reso_grid)
#
#         if self.unet3d is not None:
#             fea_grid = self.unet3d(fea_grid)
#
#         return fea_grid
#
#     def pool_local(self, index, c):
#         bs, fea_dim = c.size(0), c.size(2)
#         keys = index.keys()
#
#         c_out = 0
#         for key in keys:
#             # scatter plane features from points
#             if key == 'grid':
#                 fea = self.scatter(c.permute(0, 2, 1), index[key])
#             else:
#                 fea = self.scatter(c.permute(0, 2, 1), index[key])
#             if self.scatter == scatter_max:
#                 fea = fea[0]
#             # gather feature back to points
#             fea = fea.gather(dim=2, index=index[key].expand(-1, fea_dim, -1))
#             c_out += fea
#         return c_out.permute(0, 2, 1)
#
#     def forward(self, inputs):
#         inputs = inputs['points']
#         index = inputs['index']
#
#         batch_size, T, D = inputs.size()
#
#         if self.map2local:
#             pp = self.map2local(inputs)
#             net = self.fc_pos(pp)
#         else:
#             net = self.fc_pos(inputs)
#
#         net = self.block[0](net)
#         for block in self.block[1:]:
#             pooled = self.pool_local(index, net)
#             net = torch.cat([net, pooled], dim=2)
#             net = block(net)
#
#         c = self.fc_c(net)
#
#         fea = {}
#         if 'grid' in self.plane_type:
#             fea['grid'] = self.generate_grid_features(index['grid'], c)
#         if 'xz' in self.plane_type:
#             fea['xz'] = self.generate_plane_features(index['xz'], c)
#         if 'xy' in self.plane_type:
#             fea['xy'] = self.generate_plane_features(index['xy'], c)
#         if 'yz' in self.plane_type:
#             fea['yz'] = self.generate_plane_features(index['yz'], c)
#
#         return fea
