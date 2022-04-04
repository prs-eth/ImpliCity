# encoding: utf-8
# Author: Bingxin Ke
# Created: 2021/10/13

import torch
import torch.nn as nn
from torch import distributions as dist
from typing import Dict


class ImpliCityONet(nn.Module):
    """ Convolutional Occupancy Network with 2 encoders.

    Args:
        point_encoder (nn.Module): encoder for point cloud
        image_encoder (nn.Module): encoder for image(s)
        decoder (nn.Module): decoder network
        device (device): torch device
    """

    def __init__(self, point_encoder, image_encoder, decoder, device=None, multi_class=False, threshold=0.5):
        super().__init__()

        # self.encoder = encoder.to(device)
        self.point_encoder = point_encoder.to(device)

        self.image_encoder = image_encoder.to(device)

        self.decoder = decoder.to(device)

        self.multi_class = multi_class

        self._device = device

        self.threshold = threshold

    def forward(self, p, input_pc, input_img, **kwargs):
        """ Performs a forward pass through the network.

        Args:
            p (tensor): query points
            input_pc (tensor): conditioning input (point cloud)
            input_img (tensor): conditioning input (images)
        """
        # batch_size = p.size(0)
        feature_planes = self.encode_inputs(input_pc, input_img)
        p_r = self.decode(p, feature_planes, **kwargs)
        return p_r

    def encode_inputs(self, input_pc, input_img):
        """ Encodes the input.

        Args:
            input_pc (tensor): input point cloud
            input_img (tensor): input images
        """
        feature_planes: Dict = self.point_encoder(input_pc)

        image_features: torch.Tensor = self.image_encoder(input_img)

        # flip image features
        image_features = image_features.flip(-2)

        feature_planes['image'] = image_features
        return feature_planes

    def decode(self, p, feature_planes, **kwargs):
        """ Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            feature_planes (tensor): feature plane of latent conditioned code c (B x feature_dim x res x res)
        """

        pred = self.decoder(p, feature_planes, **kwargs)
        return pred

    def pred2occ(self, decoded_pred):
        if self.multi_class:
            _, pred_occ = torch.max(decoded_pred, -1)
        else:
            pred_bernoulli = torch.distributions.Bernoulli(logits=decoded_pred)
            pred_occ = (pred_bernoulli.probs >= self.threshold).float()
        return pred_occ

    def to(self, device):
        """ Puts the model to the device.

        Args:
            device (device): pytorch device
        """
        model = super().to(device)
        model._device = device
        return model
