# encoding: utf-8
# Author: Bingxin Ke
# Created: 2021/10/6
from src.model.decoder import LocalDecoder

# Decoder dictionary
decoder_dict = {
    # 'simple_local': local_decoder.LocalDecoder,
    'simple_local_multi_class': LocalDecoder.MultiClassLocalDecoder,
    # 'one_plane_local_decoder': OnePlaneLocalDecoder.OnePlaneLocalDecoder,
    # 'simple_local_crop': decoder.PatchLocalDecoder,
    # 'simple_local_point': decoder.LocalPointDecoder
}
