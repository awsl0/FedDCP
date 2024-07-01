import math
from turtle import forward
from pyparsing import alphas
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class NC1Loss(nn.Module):
    '''
    Modified Center loss, 1 / n_k ||h-miu||
    '''
    def __init__(self, num_classes=10, feat_dim=128, use_gpu=True):
        super(NC1Loss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.means = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.means = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        #这两行代码计算了每个样本与每个类别中心之间的欧氏距离
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.means, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.means.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long()#这行代码创建了一个包含类别索引的张量
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)#变形成【bs，num】大小的矩阵，每一行都包含了一个样本的真实类别标签。
        mask = labels.eq(classes.expand(batch_size, self.num_classes))#变形成【bs，num】大小的矩阵，这个布尔张量中的每个元素表示对应样本的真实类别是否等于当前类别（classes 张量中的值）。如果相等，则相应位置的布尔值为 True，否则为 False。

        dist = distmat * mask.float()#对于每个样本，只有其真实类别对应的列的距离值保留，其他列的距离值被置零。这是为了在计算损失时只考虑每个样本与其真实类别中心的距离。
        D = torch.sum(dist, dim=0)#这一行代码计算了一个batchsize中每个类别中心的距离和
        N = mask.float().sum(dim=0) + 1e-10 #得到一个batchsize中包含了每个类别中的样本数量。
        # loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        loss = (D / N).clamp(min=1e-12, max=1e+12).sum() / self.num_classes
        # (D / N): 这一部分计算了每个类别中心的平均距离，
        # .clamp(min=1e-12, max=1e+12): 这一部分使用 clamp 函数将每个元素限制在一个范围内，以防止在后续计算中产生数值不稳定性


        return loss, self.means

    
class NC1Loss_1(nn.Module):
    '''
    Modified Center loss, 1 / n_k**2 ||h-miu||
    '''
    def __init__(self, num_classes=10, feat_dim=128, use_gpu=True):
        super(NC1Loss_1, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.means = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.means = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.means, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.means.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        D = torch.sum(dist, dim=0)
        N = mask.float().sum(dim=0) + 1e-5
        N = N ** 2
        # print()
        # loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        loss = (D/N).clamp(min=1e-12, max=1e+12).sum() / self.num_classes

        return loss, self.means


def NC2Loss(means):
    '''
    NC2 loss v0: maximize the average minimum angle of each centered class mean
    使各居中类平均值的平均最小角最大化
    '''
    g_mean = means.mean(dim=0)
    centered_mean = means - g_mean
    means_ = F.normalize(centered_mean, p=2, dim=1)
    cosine = torch.matmul(means_, means_.t())
    # make sure that the diagnonal elements cannot be selected
    cosine = cosine - 2. * torch.diag(torch.diag(cosine))
    max_cosine = cosine.max().clamp(-0.99999, 0.99999)
    # print('min angle:', min_angle)
    # maxmize the minimum angle
    # dim=1 means the maximum angle of the other class to each class
    loss = -torch.acos(cosine.max(dim=1)[0].clamp(-0.99999, 0.99999)).mean()
    # loss = cosine.max(dim=1)[0].clamp(-0.99999, 0.99999).mean() + 1. / (means.size(0)-1)

    return loss

def NC2Loss_v1(means):
    '''
    NC2 loss v1: maximize the minimum angle of centered class means
    使中心类的最小角度最大化
    '''
    g_mean = means.mean(dim=0)
    centered_mean = means - g_mean
    means_ = F.normalize(centered_mean, p=2, dim=1)
    cosine = torch.matmul(means_, means_.t())
    # make sure that the diagnonal elements cannot be selected
    cosine = cosine - 2. * torch.diag(torch.diag(cosine))
    max_cosine = cosine.max().clamp(-0.99999, 0.99999)
    # print('min angle:', min_angle)
    # maxmize the minimum angle
    # dim=1 means the maximum angle of the other class to each class
    loss = -torch.acos(max_cosine)
    min_angle = math.degrees(torch.acos(max_cosine.detach()).item())
    # loss = cosine.max(dim=1)[0].clamp(-0.99999, 0.99999).mean() + 1. / (means.size(0)-1)

    return loss, max_cosine

def NC2Loss_v2(means):
    '''
    NC2 loss: make the cosine of any pair of class-means be close to -1/(C-1))
    使任意一对类均值的余弦值接近于-1/(C-1))
    '''
    C = means.size(0)
    g_mean = means.mean(dim=0)
    centered_mean = means - g_mean
    means_ = F.normalize(centered_mean, p=2, dim=1)
    cosine = torch.matmul(means_, means_.t())
    # make sure that the diagnonal elements cannot be selected
    cosine_ = cosine - 2. * torch.diag(torch.diag(cosine))
    max_cosine = cosine_.max().clamp(-0.99999, 0.99999)
    cosine = cosine_ + (1. - 1/(C-1)) * torch.diag(torch.diag(cosine))
    # print('min angle:', min_angle)
    # maxmize the minimum angle
    loss = cosine.norm()
    # loss = -torch.acos(cosine.max(dim=1)[0].clamp(-0.99999, 0.99999)).mean()
    # loss = cosine.max(dim=1)[0].clamp(-0.99999, 0.99999).mean() + 1. / (means.size(0)-1)

    return loss, max_cosine