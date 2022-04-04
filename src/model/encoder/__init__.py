# encoding: utf-8
# Author: Bingxin Ke
# Created: 2021/10/6

from src.model.encoder import pointnet
from src.model.encoder import HGFilters


encoder_dict = {
    'pointnet_local_pool': pointnet.LocalPoolPointnet,
    'hg_filter': HGFilters.HGFilter,
    # 'pointnet_crop_local_pool': pointnet.PatchLocalPoolPointnet,
    # 'pointnet_plus_plus': pointnetpp.PointNetPlusPlus,
    # 'voxel_simple_local': voxels.LocalVoxelEncoder,
}
