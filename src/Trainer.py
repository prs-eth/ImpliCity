# encoding: utf-8
# Author: Bingxin Ke
# Created: 2021/10/7

from collections import defaultdict
from typing import Dict

import torch
import torch.nn as nn
from tqdm import tqdm

from src.dataset import LAND_TYPES
from src.metric import compute_iou, Accuracy, Precision, Recall
from src.model import ConvolutionalOccupancyNetwork, ImpliCityONet


class Trainer:
    """ Trainer object for the Occupancy Network.

    Args:
        model (nn.Module): Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device
        optimize_every: gradient accumulation steps
        cfg_loss_weights: configuration of weighted loss
        multi_class: train in a multi-class classification manner
        multi_tower_weights: (not used in this version)
        balance_weight: (not used in this version)
    """

    def __init__(self, model: nn.Module, optimizer, criteria, device=None, optimize_every=1, cfg_loss_weights=None,
                 multi_class=False, multi_tower_weights=None, balance_weight=False):
        self.model: nn.Module = model
        self.optimizer = optimizer
        self.device = device
        self.balance_building_weight = balance_weight  # if true, calculate balanced building weight based on number of points

        self.loss_func = criteria

        self.multi_class = multi_class
        if self.multi_class:
            self.n_classes = 3
        else:
            self.n_classes = 2

        self.optimizer.zero_grad()

        # weighted loss
        self.multi_tower_weights = multi_tower_weights
        # if isinstance(self.model, CityConvONetMultiTower):
        #     assert self.multi_tower_weights is not None, "Training CityConvONetMultiTower requires weights"
        self.cfg_loss_weights = cfg_loss_weights
        # self.loss_weights_query = self.cfg_loss_weights['query']
        self.loss_weights_land = self.cfg_loss_weights['land']
        self.loss_weights_gt = self.cfg_loss_weights['gt']

        # binary metrics
        self.binary_metrics = {
            'accuracy': Accuracy(n_class=self.n_classes),
            'precision': Precision(n_class=self.n_classes),
            'recall': Recall(n_class=self.n_classes),
            # 'F1-score': lambda a, b: f1_score(a, b, average='binary'),
        }

        # gradient accumulation
        self.optimize_every = optimize_every
        self.accumulated_steps = 0

        self.accumulated_n_pts = 0
        self.last_avg_n_pts = 0

        self.accumulated_loss = 0.
        self.last_avg_loss_total = 0.  # averaged loss for last accumulation round

        self.acc_loss_category: Dict = {key: 0. for key in LAND_TYPES}
        if self.multi_tower_weights is not None:
            for key in self.multi_tower_weights.keys():
                self.acc_loss_category[key] = 0.
        self.last_avg_loss_category = defaultdict(float)
        self.acc_metrics_total: Dict = {key: 0. for key in self.binary_metrics.keys()}
        self.last_avg_metrics_total = defaultdict(float)
        self.acc_metrics_category: Dict = {f'{metric}/{cat}': 0. for metric in self.binary_metrics.keys() for cat in LAND_TYPES}
        self.last_avg_metrics_category: Dict = defaultdict(float)

    def train_step(self, data):
        """ Performs a training step.

        Args:
            data (dict): data dictionary
        """
        device = self.device
        query_pts: torch.Tensor = data.get('query_pts').to(device)
        query_occ: torch.Tensor = data.get('query_occ').to(device)
        inputs = data.get('inputs').to(device)

        # mask_land
        mask_gt = data.get('mask_gt').to(device)
        mask_land: Dict = defaultdict(torch.Tensor)
        for key in LAND_TYPES:
            mask_land[key] = data.get(f'mask_{key}').to(device)

        self.model.train()

        if isinstance(self.model, ConvolutionalOccupancyNetwork):
            pred = self.model.forward(p=query_pts, inputs=inputs)
            loss_i = self.loss_func(pred.squeeze(), query_occ.squeeze())
        elif isinstance(self.model, ImpliCityONet):
            input_img: torch.Tensor = data.get('image').to(device)
            pred = self.model.forward(p=query_pts, input_pc=inputs, input_img=input_img)
            loss_i = self.loss_func(pred.squeeze(), query_occ.squeeze())
        # elif isinstance(self.model, CityConvONetMultiTower):
        #     input_img: torch.Tensor = data.get('image').to(device)
        #     pred_dict = self.model.forward_multi_tower(p=query_pts, input_pc=inputs, input_img=input_img)
        #     loss_i = 0.
        #     for _key, branch_pred in pred_dict.items():  # point, image, joint
        #         branch_loss = self.loss_func(branch_pred.squeeze(), query_occ.squeeze())
        #         self.acc_loss_category[_key] += branch_loss.mean(-1).mean()
        #         loss_i += self.multi_tower_weights[_key] * branch_loss
        #     pred = pred_dict['joint']
        else:
            raise NotImplemented

        # Weighted loss
        loss, loss_category = self.compute_weighted_loss(loss_i=loss_i, mask_gt=mask_gt, weight_gt=self.loss_weights_gt,
                                                         mask_land=mask_land, weight_land=self.loss_weights_land,
                                                         balance_building_weight=self.balance_building_weight,
                                                         device=device)
        loss.backward()
        self.accumulated_steps += 1

        self.accumulated_loss += loss.detach()
        for key in LAND_TYPES:
            self.acc_loss_category[key] += loss_category[key]
        self.accumulated_n_pts += query_pts.shape[1]

        with torch.no_grad():
            # prediction label
            pred_occ: torch.Tensor = self.model.pred2occ(pred)

            # Other metrics
            for met_key, func in self.binary_metrics.items():
                self.acc_metrics_total[met_key] += func(pred_occ, query_occ)
                for cat_key in LAND_TYPES:
                    mask = mask_land[cat_key]
                    _metric = func(pred_occ[mask], query_occ[mask])
                    self.acc_metrics_category[f'{met_key}/{cat_key}'] += _metric

        # gradient accumulation
        if self.accumulated_steps == self.optimize_every:
            self.optimizer.step()
            with torch.no_grad():
                # loss
                self.last_avg_loss_total = self.accumulated_loss / self.optimize_every
                self.last_avg_loss_category = {f'loss/{key}': self.acc_loss_category[key] / self.optimize_every for key
                                               in self.acc_loss_category.keys()}
                self.acc_loss_category = {key: 0. for key in self.acc_loss_category.keys()}
                # other metrics
                for met_key in self.binary_metrics.keys():
                    self.last_avg_metrics_total[met_key] = self.acc_metrics_total[met_key] / self.optimize_every
                    self.acc_metrics_total[met_key] = 0
                for _key, value in self.acc_metrics_category.items():
                    self.last_avg_metrics_category[_key] = self.acc_metrics_category[_key] / self.optimize_every
                    self.acc_metrics_category[_key] = 0.
                self.last_avg_n_pts = self.accumulated_n_pts / self.optimize_every
            self.accumulated_loss = 0.
            self.accumulated_steps = 0
            self.accumulated_n_pts = 0
            self.optimizer.zero_grad()
            return self.last_avg_loss_total
        return loss.detach()

    @staticmethod
    def compute_weighted_loss(loss_i: torch.Tensor, mask_gt: torch.Tensor, weight_gt, mask_land: Dict,
                              weight_land, balance_building_weight, device):
        W_gt: torch.Tensor = mask_gt * weight_gt[1] + ~mask_gt * weight_gt[0]
        loss_i = W_gt * loss_i

        if balance_building_weight:
            # calculate weight online
            n_building_points = torch.sum(mask_land['building'], 1).item()
            n_total_points = mask_land['building'].shape[1]
            _weight_terrain = 1.0 * n_building_points / n_total_points
            _weight_build = 1.0 - _weight_terrain
            weight_land['building'] = [_weight_terrain, _weight_build]

        W_land = torch.ones(mask_gt.shape).to(device)
        loss_category = {}
        loss_i_detach = loss_i.detach()
        for key in LAND_TYPES:
            mask = mask_land[key]
            _weight = mask * weight_land[key][1] + ~mask * weight_land[key][0]
            loss_category[key] = (loss_i_detach * _weight).mean(-1).mean()
            W_land *= _weight

        loss_i = W_land * loss_i
        loss = loss_i.mean(-1).mean()

        return loss, loss_category

    def evaluate(self, val_loader):
        """
        Performs an evaluation.
        Args:
            val_loader: (dataloader): pytorch dataloader

        Returns: metric_dict: dict of metrics

        """
        metric_ls_dict = defaultdict(list)
        # _i = 0
        for data in tqdm(val_loader, desc="Validation"):
            eval_step_dict = self.eval_step(data)

            for k, v in eval_step_dict.items():
                metric_ls_dict[k].append(v)

        metric_dict = defaultdict(float)
        for k, v in metric_ls_dict.items():
            metric_dict[k] = torch.tensor(metric_ls_dict[k]).mean().float()
        return metric_dict

    def eval_step(self, data):
        """ Performs an evaluation step.

        Args:
            data (dict): data dictionary
        """
        self.model.eval()

        device = self.device
        eval_dict = {}

        query_pts: torch.Tensor = data.get('query_pts').to(device)
        query_occ: torch.Tensor = data.get('query_occ').to(device)
        inputs = data.get('inputs').to(device)

        eval_dict['n_query_points'] = float(query_pts.shape[1])

        # mask_land
        mask_gt = data.get('mask_gt').to(device)
        mask_land: Dict = defaultdict(torch.Tensor)
        for key in LAND_TYPES:
            mask_land[key] = data.get(f'mask_{key}').to(device)

        with torch.no_grad():
            if isinstance(self.model, ConvolutionalOccupancyNetwork):
                pred = self.model.forward(p=query_pts, inputs=inputs)
                loss_i = self.loss_func(pred.squeeze(), query_occ.squeeze())
            elif isinstance(self.model, ImpliCityONet):
                input_img: torch.Tensor = data.get('image').to(device)
                pred = self.model.forward(p=query_pts, input_pc=inputs, input_img=input_img)
                loss_i = self.loss_func(pred.squeeze(), query_occ.squeeze())
            # elif isinstance(self.model, CityConvONetMultiTower):
            #     input_img: torch.Tensor = data.get('image').to(device)
            #     pred_dict = self.model.forward_multi_tower(p=query_pts, input_pc=inputs, input_img=input_img)
            #     loss_i = 0.
            #     for _key, branch_pred in pred_dict.items():  # point, image, joint
            #         branch_loss = self.loss_func(branch_pred.squeeze(), query_occ.squeeze())
            #         eval_dict[f'loss/{_key}'] = branch_loss.mean(-1).mean()
            #         loss_i += self.multi_tower_weights[_key] * branch_loss
            #     pred = pred_dict['joint']
            else:
                raise NotImplemented

            # Compute loss
            loss, loss_category = self.compute_weighted_loss(loss_i=loss_i, mask_gt=mask_gt,
                                                             weight_gt=self.loss_weights_gt,
                                                             mask_land=mask_land, weight_land=self.loss_weights_land,
                                                             balance_building_weight=self.balance_building_weight,
                                                             device=device)
            eval_dict['loss'] = loss
            for _key, _value in loss_category.items():
                eval_dict[f'loss/{_key}'] = _value

            # prediction label
            pred_occ: torch.Tensor = self.model.pred2occ(pred)

            # Other metrics
            for key, func in self.binary_metrics.items():
                eval_dict[key] = func(pred_occ, query_occ)
                for cat_key in LAND_TYPES:
                    mask = mask_land[cat_key]
                    _metric = func(pred_occ[mask], query_occ[mask])
                    eval_dict[f'{key}/{cat_key}'] = _metric

            # compute IoU (intersection over union)
            occ_iou_gt_np = query_occ.cpu().numpy()
            occ_iou_hat_np = pred_occ.cpu().numpy()
            iou = compute_iou(occ_iou_gt_np, occ_iou_hat_np)
            eval_dict['iou'] = iou
        return eval_dict
