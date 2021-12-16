import torch
import torch.nn.functional as F
import numpy as np


def soft_cross_entropy(pred, soft_targets):
    logsoftmax = torch.nn.LogSoftmax(dim=1)
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))


def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)

    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index, :]
    if isinstance(y, dict):
        y_a = {}
        y_b = {}
        y_a['verb'], y_b['verb'] = y['verb'], y['verb'][index]
        y_a['noun'], y_b['noun'] = y['noun'], y['noun'][index]
    else:
        y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_data_and_targets(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)

    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index, :]
    if isinstance(y, dict):
        mixed_y = {}
        y['verb'] = F.one_hot(y['verb'], num_classes=97).float()
        y['noun'] = F.one_hot(y['noun'], num_classes=300).float()
        y['verb'] = (1 - 0.05) * y['verb'] + (0.05 / y['verb'].shape[1])
        y['noun'] = (1 - 0.05) * y['noun'] + (0.05 / y['noun'].shape[1])
        mixed_y['verb'] = lam * y['verb'] + (1 - lam) * y['verb'][index]
        mixed_y['noun'] = lam * y['noun'] + (1 - lam) * y['noun'][index]
    else:
        y = F.one_hot(y, num_classes=10).float()
        mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y


def mixup_criterion(criterion, pred, y_a, y_b, lam, weights=None):
    loss_a = criterion(pred, y_a)
    if weights is not None:
        loss_a = loss_a * weights
        loss_a = loss_a.sum(1)
        loss_a = loss_a.mean()
    loss_b = criterion(pred, y_b)
    if weights is not None:
        loss_b = loss_b * weights
        loss_b = loss_b.sum(1)
        loss_b = loss_b.mean()
    return lam * loss_a + (1 - lam) * loss_b


def mixup_accuracy(output, target_a, target_b, lam, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target_a.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = lam * pred.eq(target_a.view(1, -1).expand_as(pred)) \
        + (1 - lam) * pred.eq(target_b.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).to(torch.float32).sum(0)
        res.append(float(correct_k.mul_(100.0 / batch_size)))
    return tuple(res)
