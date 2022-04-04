# encoding: utf-8
# Author: Bingxin Ke
# Created: 2021/10/13
from collections import defaultdict
from typing import List, Union

import numpy as np


def concat_dict_data(dict_ls: List[dict]):
    # for dic in dict_ls:
    #     if not isinstance(dic, dict):
    #         raise TypeError(f"dict expected, got {type(dic)} instead")
    out_dic = defaultdict()
    keys = set([list(dic.keys())[i] for dic in dict_ls for i in range(len(dic.keys()))])
    for key in keys:
        if isinstance(dict_ls[0][key], (List, np.ndarray)):
            temp_ls = [np.array(dic[key]) for dic in dict_ls]
            out_dic[key] = np.concatenate(temp_ls, 0)
        else:
            out_dic[key] = dict_ls[0][key]
    return out_dic


def index_dict_data(dict_data: dict, indices: Union[np.ndarray, List]):
    """
        Use the same index array to subsample data. Do the same operation for each data in the dictionary
    Args:
        dict_data:
        indices:

    Returns:

    """
    # if not isinstance(dict_data, dict):
    #     raise TypeError(f"dict expected, got {type(dict_data)} instead")

    out_dic = defaultdict()
    indices = np.asarray(indices)
    for key, value in dict_data.items():
        if isinstance(value, (List, np.ndarray)):
            value = np.asarray(value)
            out_dic[key] = value[indices]
        else:
            out_dic[key] = value

    return out_dic


if __name__ == '__main__':
    # test
    dic_1 = {
        'pts': np.random.randint(1, 10, 10),
        'occ': np.random.randint(0, 5, 10),
        'name': 's'
    }
    dic_2 = {
        'pts': np.random.randint(1, 10, 10),
        'occ': np.random.randint(0, 2, 10),
        'name': 's'
    }
    dic_out = concat_dict_data([dic_1, dic_2])

    indices = [3, 2, 0, 1]
    dic_out_2 = index_dict_data(dic_1, indices)

