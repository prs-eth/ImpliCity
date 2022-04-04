# encoding: utf-8
# Author: Bingxin Ke
# Created: 2021/9/28

import numpy as np
import random
import torch


def lock_seed(seed: int = 0):
    """
    Set seed to get reproducible result
    Args:
        seed: (int)

    Returns:

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(0)
