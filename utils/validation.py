import torch
from torch import Tensor
from torchmetrics import R2Score


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def is_valid(labels: Tensor, outputs:Tensor) -> Tensor:
    """Return boolean of prediction for each polygon based on followed rules"""
    # prediction >=50% is main species
    # rule_1 = (labels >= 0.5) & (outputs >= 0.5)
    # prediction <=10% is regarded as not exist
    rule_2 = (labels == 0) & (outputs < 0.05)
    # prediction between >5% is regarded as exist
    rule_3 = (labels > 0) & (outputs >= 0.05)
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


def valid_r2_num(labels: Tensor, outputs:Tensor, based_r2:float=0.5) -> int:
    """Return number of valid prediction whose r2 is over given value"""
    assert outputs.size() == labels.size(), "Size of outputs and labels should be equal"
    labels = labels.t()
    outputs = outputs.t()
    batch_sz = labels.size(1)
    r2score = R2Score(num_outputs=batch_sz, multioutput='raw_values').to(device)
    r2:Tensor = r2score(outputs, labels)
    num = (r2 > based_r2).sum().item()
    return num


def true_pred_num(labels: Tensor, outputs:Tensor) -> int:
    """Return true prediction number of classification for each batch"""
    assert outputs.size(0) == labels.size(0), "Size of outputs and labels should be equal"
    _, predicted = torch.max(outputs.data, 1)
    num = (predicted == labels).sum().item()
    return num