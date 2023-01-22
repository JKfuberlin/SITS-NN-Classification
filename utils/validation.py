import torch
from torch import Tensor
from torchmetrics import R2Score

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def is_valid(labels: Tensor, outputs:Tensor) -> Tensor:
    """Return boolean of prediction for each polygon based on followed rules"""
    # prediction >=50% is main species
    rule_1 = (labels >= 0.5) & (outputs >= 0.5)
    # prediction <=10% is regarded as not exist
    rule_2 = (labels == 0) & (outputs <= 0.1)
    # prediction between 10%-50% is regarded as exist but is not main species
    rule_3 = ((labels > 0) & (labels < 0.5)) & ((outputs > 0) & (outputs < 0.5))
    res = rule_1 | rule_2 | rule_3
    return res.sum(dim=0)

def valid_num(labels: Tensor, outputs:Tensor) -> int:
    """Return number of valid prediction for each batch"""
    assert outputs.size != labels.size(), "Size of outputs and labels should be equal"
    tgt_len = labels.size(0)
    res = is_valid(labels, outputs)
    num = (res == tgt_len).sum().item()
    return num


def valid_r2_num(labels: Tensor, outputs:Tensor, based_r2:float=0.5) -> int:
    """Return number of valid prediction whose r2 is over given value"""
    assert outputs.size != labels.size(), "Size of outputs and labels should be equal"
    batch_sz = labels.size(1)
    r2score = R2Score(num_outputs=batch_sz, multioutput='raw_values').to(device)
    r2:Tensor = r2score(labels/100., outputs)
    num = (r2 > based_r2).sum().item()
    return num
    