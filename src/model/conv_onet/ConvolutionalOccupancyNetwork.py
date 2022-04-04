# encoding: utf-8

import torch
import torch.nn as nn


class ConvolutionalOccupancyNetwork(nn.Module):
    """ Convolutional Occupancy Network class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        device (device): torch device
    """

    def __init__(self, encoder, decoder, device=None, multi_class=False, threshold=0.5):
        super().__init__()

        self.encoder = encoder.to(device)

        self.decoder = decoder.to(device)

        self.multi_class = multi_class

        self._device = device

        self.threshold = threshold

    def forward(self, p, inputs, **kwargs):
        """ Performs a forward pass through the network.

        Args:
            p (tensor): sampled points
            inputs (tensor): conditioning input
            # sample (bool): whether to sample for z
        """
        # batch_size = p.size(0)
        feature_planes = self.encode_inputs(inputs)
        p_r = self.decode(p, feature_planes, **kwargs)
        return p_r

    def encode_inputs(self, inputs, **kwargs):
        """ Encodes the input.

        Args:
            input (tensor): the input
        """

        # if self.encoder is not None:
        c = self.encoder(inputs, **kwargs)
        # else:
        #     # Return inputs?
        #     c = torch.empty(inputs.size(0), 0)

        return c

    def decode(self, p, feature_planes, **kwargs):
        """ Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            feature_planes (tensor): feature plane of latent conditioned code c (B x feature_dim x res x res)
        """

        pred = self.decoder(p, feature_planes, **kwargs)
        # p_r = dist.Bernoulli(logits=logits)
        # return p_r
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
