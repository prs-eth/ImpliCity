# encoding: utf-8
# Author: Bingxin Ke
# Created: 2021/10/7

import torch.nn as nn
import torch.nn.functional as F
from src.model.block import ResnetBlockFC

#
# class LocalDecoder(nn.Module):
#     ''' Decoder.
#         Instead of conditioning on global features, on plane/volume local features.
#
#     Args:
#         dim (int): input dimension
#         feature_dim (int): dimension of latent conditioned code c
#         hidden_size (int): hidden size of Decoder network
#         n_blocks (int): number of block ResNetBlockFC layers
#         leaky (bool): whether to use leaky ReLUs
#         sample_mode (str): sampling feature strategy, bilinear|nearest
#         padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
#     '''
#
#     def __init__(self, dim=3, feature_dim=128,
#                  hidden_size=256, n_blocks=5, leaky=False, sample_mode='bilinear', padding=0.1):
#         super().__init__()
#         self.feature_dim = feature_dim
#         self.n_blocks = n_blocks
#
#         if feature_dim != 0:
#             self.fc_c = nn.ModuleList([
#                 nn.Linear(feature_dim, hidden_size) for i in range(n_blocks)
#             ])
#
#         self.fc_p = nn.Linear(dim, hidden_size)
#
#         self.blocks = nn.ModuleList([
#             ResnetBlockFC(hidden_size) for i in range(n_blocks)
#         ])
#
#         self.fc_out = nn.Linear(hidden_size, 1)
#
#         if not leaky:
#             self.actvn = F.relu
#         else:
#             self.actvn = lambda x: F.leaky_relu(x, 0.2)
#
#         self.sample_mode = sample_mode
#         self.padding = padding
#
#     def sample_plane_feature(self, p, feature_plane, plane):
#         # xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding)  # normalize to the range of (0, 1)
#
#         xy = p.clone()[:, :, [0, 1]]
#         # if self.shift_normalized is None:
#         #     xy = p.clone()
#         # else:
#         #     xy = self.shift_normalized(p.clone(), plane=plane)
#         # print('sample_plane_feature, xy.shape', xy.shape)
#         xy = xy[:, :, None].float()
#         vgrid = 2.0 * xy - 1.0  # normalize to (-1, 1) for grid_sample
#         # vgrid = xy
#         features = F.grid_sample(feature_plane, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1)  # features: (1, feature_dim, n_pts)
#         return features
#
#     # def sample_grid_feature(self, p, c):
#     #     p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)  # normalize to the range of (0, 1)
#     #     p_nor = p_nor[:, :, None, None].float()
#     #     vgrid = 2.0 * p_nor - 1.0  # normalize to (-1, 1)
#     #     # acutally trilinear interpolation if mode = 'bilinear'
#     #     c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(
#     #         -1).squeeze(-1)
#     #     return c
#
#     def forward(self, p, c_plane, **kwargs):
#         """
#
#         Args:
#             p:
#             c_plane: dict of feature planes, (B x feature_dim x res x res)
#             **kwargs:
#
#         Returns:
#
#         """
#         if self.feature_dim != 0:
#             plane_type = list(c_plane.keys())
#             c = 0
#             # if 'grid' in plane_type:
#             #     c += self.sample_grid_feature(p, feature_planes['grid'])
#             # if 'xz' in plane_type:
#             #     c += self.sample_plane_feature(p, feature_planes['xz'], plane='xz')
#             if 'xy' in plane_type:
#                 c += self.sample_plane_feature(p, c_plane['xy'], plane='xy')
#             # if 'yz' in plane_type:
#             #     c += self.sample_plane_feature(p, feature_planes['yz'], plane='yz')
#             c = c.transpose(1, 2)
#
#         p = p.float()
#         net = self.fc_p(p)
#
#         for i in range(self.n_blocks):
#             if self.feature_dim != 0:
#                 net = net + self.fc_c[i](c)
#
#             net = self.blocks[i](net)
#
#         out = self.fc_out(self.actvn(net))
#         out = out.squeeze(-1)
#
#         return out


class MultiClassLocalDecoder(nn.Module):
    ''' Decoder.
        Decode the local feature(s).

    Args:
        dim (int): input dimension
        feature_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of block ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
    '''

    def __init__(self, dim=3, feature_dim=128, hidden_size=256, n_blocks=5, out_dim=1, leaky=False, sample_mode='bilinear'):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_blocks = n_blocks

        if feature_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(feature_dim, hidden_size) for i in range(n_blocks)
            ])

        self.fc_p = nn.Linear(dim, hidden_size)

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])

        self.fc_out = nn.Linear(hidden_size, out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        # self.padding = padding

    def sample_plane_feature(self, p, c):
        # p: [B, N, 2]
        # c: [B, C, H, W]
        # xy = p.clone()[:, :, [0, 1]]
        xy = p.clone()
        xy = xy[:, :, None].float()  # [B, N, 1, 2]
        vgrid = 2.0 * xy - 1.0  # normalize to (-1, 1) for grid_sample
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1)  # [B, C, N]
        return c

    def forward(self, p, feature_planes):
        if self.feature_dim != 0:
            plane_type = list(feature_planes.keys())
            c = 0

            if 'xy' in plane_type:
                c += self.sample_plane_feature(p[:, :, [0, 1]], feature_planes['xy'])
            if 'image' in plane_type:
                c += self.sample_plane_feature(p[:, :, [0, 1]], feature_planes['image'])
            c = c.transpose(1, 2)

        p = p.float()
        net = self.fc_p(p)

        for i in range(self.n_blocks):
            if self.feature_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        # out = out.squeeze(-1)

        return out