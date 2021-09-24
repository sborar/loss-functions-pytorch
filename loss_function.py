import numpy as np
import robust_loss_pytorch
import torch
import torch.nn
import torch.nn.functional as F


class BaikalLoss(torch.nn.Module):

    def __init__(self):
        super(BaikalLoss, self).__init__()

    def forward(self, output, target):
        target = F.one_hot(target, num_classes=25)
        loss = torch.mean(torch.sum(- (torch.log(output) - (target / output)), -1))
        return loss


class HingeLoss(torch.nn.Module):

    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, output, target):
        y_onehot = -1 * torch.ones(target.size()[0], 25).long().to(device='cuda')
        y_onehot.scatter_(1, target.unsqueeze(1).to(device='cuda'), 1)
        loss = torch.mean(torch.sum((F.relu(1 / 2 - y_onehot * output)), -1))
        return loss


class SquaredHingeLoss(torch.nn.Module):

    def __init__(self):
        super(SquaredHingeLoss, self).__init__()

    def forward(self, output, target):
        y_onehot = -1 * torch.ones(target.size()[0], 25).long().to(device='cuda')
        y_onehot.scatter_(1, target.unsqueeze(1).to(device='cuda'), 1)
        loss = torch.mean(torch.sum(torch.square(F.relu(1 / 2 - y_onehot * output)), -1))
        return loss


class CubeHingeLoss(torch.nn.Module):

    def __init__(self):
        super(CubeHingeLoss, self).__init__()

    def forward(self, output, target):
        y_onehot = -1 * torch.ones(target.size()[0], 25).long().to(device='cuda')
        y_onehot.scatter_(1, target.unsqueeze(1).to(device='cuda'), 1)
        loss = torch.mean(torch.sum(torch.pow(F.relu(1 / 2 - y_onehot * output), 3), -1))
        return loss


class L1Loss(torch.nn.Module):

    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, output, target):
        target = F.one_hot(target, num_classes=25)
        loss = torch.mean(torch.sum(torch.abs(target - output), dim=1), dim=0)
        return loss


class L2Loss(torch.nn.Module):

    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, output, target):
        target = F.one_hot(target, num_classes=25)
        loss = torch.mean(torch.sum(torch.norm(target - output, 2), dim=1), dim=0)
        return loss


class CSDLoss(torch.nn.Module):

    def __init__(self):
        super(CSDLoss, self).__init__()

    def forward(self, output, target):
        output = output.float()
        target = F.one_hot(target, num_classes=25).float()
        numerator = torch.sum(output * target)
        denominator = torch.norm(output, 2) * torch.norm(target, 2)

        loss = - torch.mean(torch.log(numerator / denominator))
        return loss


class RobustAdaptiveLoss():

    def __init__(self):
        super(RobustAdaptiveLoss, self).__init__()
        self.adaptive = robust_loss_pytorch.adaptive.AdaptiveLossFunction(
            num_dims=1, float_dtype=np.float32,  device='cuda:0')

    def forward(self, output, target):

        target = F.one_hot(target, num_classes=25)
        loss = torch.mean(self.adaptive.lossfun(torch.sum(torch.abs(target - output), dim=1)[:, None]))
        return loss
