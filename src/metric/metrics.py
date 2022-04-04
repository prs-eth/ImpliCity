# encoding: utf-8
# Author: Bingxin Ke
# Created: 2021/10/14

import torch
import numpy as np


class Accuracy:
    def __init__(self, n_class=2):
        self.n_class = n_class

    def __call__(self, label_pred: torch.Tensor, label_gt: torch.Tensor):
        label_gt = label_gt.detach().int().reshape(-1)
        label_pred = label_pred.detach().int().reshape(-1)
        n_correct = (label_gt == label_pred).sum().item()
        n_total = label_gt.shape[0]
        if n_total > 0:
            return n_correct / n_total
        else:
            return 0.


class Precision:
    def __init__(self, n_class=2):
        self.n_class = n_class

    def __call__(self, label_pred: torch.Tensor, label_gt: torch.Tensor):
        label_gt = label_gt.detach().int().reshape(-1)
        label_pred = label_pred.detach().int().reshape(-1)
        # average precision
        precision_ls = []
        pred_correct = torch.as_tensor(label_gt == label_pred)
        if self.n_class > 2:
            for cls in range(0, self.n_class):
                n_tp = pred_correct[label_pred == cls].sum().item()
                n_pred_as_true = torch.sum(label_pred == cls).item()
                if n_pred_as_true > 0:
                    precision_ls.append(n_tp / n_pred_as_true)
                else:
                    precision_ls.append(0.)
            avg_precision = torch.mean(torch.as_tensor(precision_ls)).item()
        else:
            n_tp = pred_correct[label_pred > 0.5].sum().item()
            n_pred_as_true = torch.sum(label_pred > 0.5).item()
            if n_pred_as_true > 0:
                avg_precision = n_tp / n_pred_as_true
            else:
                avg_precision = 0.
        return avg_precision


class Recall:
    def __init__(self, n_class=2):
        self.n_class = n_class

    def __call__(self, label_pred: torch.Tensor, label_gt: torch.Tensor):
        label_gt = label_gt.detach().int().reshape(-1)
        label_pred = label_pred.detach().int().reshape(-1)
        # average precision
        recall_ls = []
        pred_correct = torch.as_tensor(label_gt == label_pred)
        if self.n_class > 2:
            for cls in range(0, self.n_class):
                n_tp = pred_correct[label_pred == cls].sum().item()
                n_actual_true = torch.sum(label_gt == cls).item()
                if n_actual_true > 0:
                    recall_ls.append(n_tp / n_actual_true)
                else:
                    recall_ls.append(0.)
            avg_recall = torch.mean(torch.as_tensor(recall_ls)).item()
        else:
            n_tp = pred_correct[label_pred > 0.5].sum().item()
            n_actual_true = torch.sum(label_gt > 0.5).item()
            if n_actual_true > 0:
                avg_recall = n_tp / n_actual_true
            else:
                avg_recall = 0.
        return avg_recall



# def accuracy(label_true: torch.Tensor, label_pred: torch.Tensor):
#     label_true = label_true.detach().reshape(-1)
#     label_pred = label_pred.detach().reshape(-1)
#     n_correct = (label_true == label_pred).sum().item()
#     n_total = label_true.shape[0]
#     if n_total > 0:
#         return n_correct / n_total
#     else:
#         return 0


# def precision(label_true: torch.Tensor, label_pred: torch.Tensor):
#     label_true = label_true.detach().reshape(-1)
#     label_pred = label_pred.detach().reshape(-1)
#     n_tp = (label_true & label_pred).sum().item()
#     n_pred_true = label_pred.sum().item()
#     if n_pred_true > 0:
#         return n_tp / n_pred_true
#     else:
#         return 0


# def recall(label_true: torch.Tensor, label_pred: torch.Tensor):
#     label_true = label_true.detach().reshape(-1)
#     label_pred = label_pred.detach().reshape(-1)
#     n_tp = (label_true & label_pred).sum().item()
#     n_act_true = label_true.sum().item()
#     if n_act_true > 0:
#         return n_tp / n_act_true
#     else:
#         return 0


if __name__ == '__main__':
    _accuracy = Accuracy(n_class=3)
    _precision = Precision(n_class=3)
    _recall = Recall(n_class=3)

    _n = 10
    raw_p = torch.rand((_n, 3))
    _, p = torch.max(raw_p, -1)
    t = torch.randint(0, 3, (1, _n))

    print(f"prediction: \n{p} \nground truth: \n{t}")
    print(f"Accuracy: {_accuracy(t, p)}")
    print(f"Precision: {_precision(t, p)}")
    print(f"Recall: {_recall(t, p)}")

