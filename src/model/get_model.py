# encoding: utf-8
# Author: Bingxin Ke
# Created: 2021/11/18

import logging

from src.model.conv_onet import ConvolutionalOccupancyNetwork, ImpliCityONet
from src.model.decoder import decoder_dict
from src.model.encoder import encoder_dict


def get_model(cfg, device=None):
    cfg_model = cfg['model']

    dim = cfg_model['data_dim']

    # encoder(s)
    encoder_str = cfg_model['encoder']
    encoder_kwargs = cfg_model['encoder_kwargs']
    encoder = encoder_dict[encoder_str](dim=dim, **encoder_kwargs)

    # decoder
    decoder_str = cfg_model['decoder']
    decoder_kwargs = cfg_model['decoder_kwargs']
    if 'simple_local_multi_class' == decoder_str or 'one_plane_local_decoder' == decoder_str:
        decoder_kwargs['out_dim'] = 3 if cfg_model['multi_label'] else 1
    decoder = decoder_dict[decoder_str](dim=dim, **decoder_kwargs)
    logging.debug(f"Decoder: {decoder_str}, kwargs={decoder_kwargs}")

    # conv-onet
    if 'conv_onet' == cfg_model['method']:
        model = ConvolutionalOccupancyNetwork(
            encoder=encoder,
            decoder=decoder,
            device=device,
            multi_class=cfg_model.get('multi_label', False),
            threshold=cfg['test']['threshold']
        )
    elif 'implicity_onet' == cfg_model['method']:
        image_encoder_str = cfg_model.get('encoder2')
        image_encoder_kwarg = cfg_model.get('encoder2_kwargs', {})
        image_encoder = encoder_dict[image_encoder_str](**image_encoder_kwarg)
        logging.debug(f"Image Encoder: {image_encoder_str}, kwargs={image_encoder_kwarg}")

        model = ImpliCityONet(
            point_encoder=encoder,
            image_encoder=image_encoder,
            decoder=decoder,
            device=device,
            multi_class=cfg_model.get('multi_label', False),
            threshold=cfg['test']['threshold']
        )
    # elif 'city_conv_onet_multi_tower' == cfg_model['method']:
    #     image_encoder_str = cfg_model.get('image_encoder')
    #     image_encoder_kwarg = cfg_model.get('image_encoder_kwargs', {})
    #     image_encoder = encoder_dict[image_encoder_str](**image_encoder_kwarg)
    #     logging.debug(f"Image Encoder: {image_encoder_str}, kwargs={image_encoder_kwarg}")
    #
    #     out_dim = 3 if cfg_model['multi_label'] else 1
    #     point_decoder_str = cfg_model.get('point_decoder')
    #     point_decoder_kwarg = cfg_model.get('point_decoder_kwargs', {})
    #     point_decoder_kwarg['out_dim'] = out_dim
    #     point_decoder = decoder_dict[point_decoder_str](**point_decoder_kwarg)
    #     logging.debug(f"Point Decoder: {point_decoder_str}, kwargs={point_decoder_kwarg}")
    #
    #     image_decoder_str = cfg_model.get('image_decoder')
    #     image_decoder_kwarg = cfg_model.get('image_decoder_kwargs', {})
    #     image_decoder_kwarg['out_dim'] = out_dim
    #     image_decoder = decoder_dict[image_decoder_str](**image_decoder_kwarg)
    #     logging.debug(f"Image Decoder: {image_decoder_str}, kwargs={image_decoder_kwarg}")
    #
    #     model = CityConvONetMultiTower(
    #         point_encoder=encoder,
    #         image_encoder=image_encoder,
    #         point_decoder=point_decoder,
    #         image_decoder=image_decoder,
    #         joint_decoder=decoder,
    #         device=device,
    #         multi_class=cfg_model.get('multi_label', False),
    #         threshold=cfg['test']['threshold']
    #     )
    else:
        raise ValueError("Unknown method")

    return model


if __name__ == '__main__':
    # Run at project root

    from src.utils.libconfig import config
    import torch

    cfg_file_path = "config/train/conv_2d3d/conv_2d3d_101_base.yaml"

    _cfg = config.load_config(cfg_file_path, None)

    _cuda_avail = torch.cuda.is_available()
    _device = torch.device("cuda" if _cuda_avail else "cpu")

    _model = get_model(_cfg, device=_device)
