import os
import torch
from torch import nn
import torch.nn.functional as F


class STNnd(nn.Module):
    def __init__(self,
                 dim_input,
                 act_fn=F.leaky_relu):
        super(STNnd, self).__init__()
        self.dim_input = dim_input
        self.act_fn = act_fn

        self.conv1 = nn.Conv1d(self.dim_input, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.dim_input**2)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.iden_flat = torch.eye(dim_input).view(1, self.dim_input**2)

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.act_fn(self.bn1(self.conv1(x)))
        x = self.act_fn(self.bn2(self.conv2(x)))
        x = self.act_fn(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = self.act_fn(self.bn4(self.fc1(x)))
        x = self.act_fn(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        x = x + self.iden_flat.to(x.device)
        x = x.view(-1, self.dim_input, self.dim_input)
        return x


class PointNetfeat(nn.Module):
    """
    pointnet feature extractor
    """
    def __init__(self,
                 dim_input,
                 global_feat=True,
                 act_fn=F.leaky_relu):
        super(PointNetfeat, self).__init__()
        self.dim_input = dim_input
        self.act_fn = act_fn

        self.stn = STNnd(dim_input=self.dim_input)
        self.conv1 = torch.nn.Conv1d(self.dim_input, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = self.act_fn(self.bn1(self.conv1(x)))
        pointfeat = x
        x = self.act_fn(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans

class PointNetCls(nn.Module):
    def __init__(self,
                 dim_input,
                 num_classes,
                 act_fn=F.leaky_relu):
        super(PointNetCls, self).__init__()
        self.dim_input = dim_input
        self.num_classes = num_classes

        self.feat = PointNetfeat(dim_input=self.dim_input,
                                 global_feat=True)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.num_classes)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

        self.act_fn = act_fn

    def forward(self, x):
        x, trans = self.feat(x)
        x = self.act_fn(self.bn1(self.fc1(x)))
        x = self.act_fn(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return F.log_softmax(x, dim=0), trans


class PointNetDenseCls(nn.Module):
    def __init__(self,
                 dim_input,
                 num_classes,
                 act_fn=F.leaky_relu):
        super().__init__()
        self.dim_input = dim_input
        self.num_classes = num_classes
        self.act_fn = act_fn

        self.feat = PointNetfeat(self.dim_input,
                                 global_feat=False)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.num_classes, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans = self.feat(x)
        x = self.act_fn(self.bn1(self.conv1(x)))
        x = self.act_fn(self.bn2(self.conv2(x)))
        x = self.act_fn(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1, self.num_classes), dim=-1)
        x = x.view(batchsize, n_pts, self.num_classes)
        return x, trans


if __name__ == '__main__':
    DIM_INPUT = 3  # dimension of the input
    NUM_CLASSES = 5  # number of categorical outputs
    NUM_SEG_CLASSES = 3  # number of categories for pixel-wise segmentation

    NUM_SAMPLES = 32
    POINTS_PER_DATAPOINT = 2500

    sim_data = torch.rand(NUM_SAMPLES,
                          DIM_INPUT,
                          POINTS_PER_DATAPOINT)

    trans = STNnd(dim_input=DIM_INPUT)
    out = trans(sim_data)
    print('stn', out.size())

    pointfeat = PointNetfeat(dim_input=DIM_INPUT,
                             global_feat=True)
    out, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(dim_input=DIM_INPUT,
                             global_feat=False)
    out, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(dim_input=DIM_INPUT,
                      num_classes=NUM_CLASSES)
    out, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(dim_input=DIM_INPUT,
                           num_classes=NUM_SEG_CLASSES)
    out, _ = seg(sim_data)
    print('seg', out.size())

