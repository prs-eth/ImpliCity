# encoding: utf-8
# Author: Bingxin Ke
# Created: 2021/11/19

from scipy import ndimage


def dilate_mask(mask_in, iterations=1):
    """
    Dilates a binary mask.
    :param mask_in:     np.array, binary mask to be dilated
    :param iterations:  int, number of dilation iterations
    :return:            np.array, dilated binary mask
    """

    return ndimage.morphology.binary_dilation(mask_in, iterations=iterations)
