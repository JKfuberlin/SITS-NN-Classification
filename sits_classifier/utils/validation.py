import torch
from torch import Tensor
from torchmetrics import R2Score
from torchmetrics.classification import MultilabelAccuracy


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def is_valid(labels: Tensor, outputs:Tensor) -> Tensor:
    """Return boolean of prediction for each polygon based on followed rules"""
    # prediction >=50% is main species
    # rule_1 = (labels >= 0.5) & (outputs >= 0.5)
    # prediction <=10% is regarded as not exist
    rule_2 = (labels == 0) & (outputs < 0.1)
    # prediction between >5% is regarded as exist
    rule_3 = (labels > 0) & (outputs >= 0.1)
    res = rule_2 | rule_3
    return res.sum(dim=0)


def valid_pred_num(labels: Tensor, outputs:Tensor) -> int:
    """Return valid prediction number of regression for each batch"""
    assert outputs.size() == labels.size(), "Size of outputs and labels should be equal"
    labels = labels.t()
    outputs = outputs.t()
    tgt_len = labels.size(0)
    res = is_valid(labels, outputs)
    num = (res == tgt_len).sum().item()
    return num


def avg_r2_score(labels: Tensor, outputs:Tensor) -> int:
    """Return average r2 of each prediction in the batch"""
    assert outputs.size() == labels.size(), "Size of outputs and labels should be equal"
    labels = labels.t()
    outputs = outputs.t()
    batch_sz = labels.size(1)
    r2score = R2Score(num_outputs=batch_sz, multioutput='uniform_average').to(device)
    r2:Tensor = r2score(outputs, labels)
    return r2.item()


def true_pred_num(labels: Tensor, outputs:Tensor) -> int:
    """Return true prediction number of classification for each batch"""
    assert outputs.size(0) == labels.size(0), "Size of outputs and labels should be equal"
    _, predicted = torch.max(outputs.data, 1)
    num = (predicted == labels).sum().item()
    return num


def multi_label_acc(y_true:Tensor, y_pred:Tensor) -> float:
    """Return total correct prediction number of multi-label for each batch"""
    assert y_true.size() == y_pred.size(), "Size of outputs and labels should be equal"
    mla = MultilabelAccuracy(num_labels=y_true.size(1)).to(device)
    acc = mla(y_pred, y_true).item()
    return acc