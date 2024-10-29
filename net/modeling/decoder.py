# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
#
# class Decoder(nn.Module):
#     def __init__(self, num_classes, backbone, BatchNorm):
#         super(Decoder, self).__init__()
#         if backbone == 'resnet' or backbone == 'drn':
#             low_level_inplanes = 256
#         elif backbone == 'xception':
#             low_level_inplanes = 128
#         elif backbone == 'mobilenet':
#             low_level_inplanes = 24
#         else:
#             raise NotImplementedError
#
#         self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
#         self.bn1 = nn.GroupNorm(num_groups=2, num_channels=48)
#         self.relu = nn.ReLU()
#         self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
#                                        nn.GroupNorm(num_groups=2, num_channels=256),
#                                        nn.ReLU(),
#                                        nn.Dropout(0.5),
#                                        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
#                                        nn.GroupNorm(num_groups=2, num_channels=256),
#                                        nn.ReLU(),
#                                        nn.Dropout(0.1),
#                                        nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
#         self._init_weight()
#
#
#     def forward(self, x, low_level_feat):
#         low_level_feat = self.conv1(low_level_feat)
#         low_level_feat = self.bn1(low_level_feat)
#         low_level_feat = self.relu(low_level_feat)
#
#         x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
#         x = torch.cat((x, low_level_feat), dim=1)
#         x = self.last_conv(x)
#
#         return x
#
#     def _init_weight(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 torch.nn.init.kaiming_normal_(m.weight)
#             elif isinstance(m, SynchronizedBatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
# def build_decoder(num_classes, backbone, BatchNorm):
#     return Decoder(num_classes, backbone, BatchNorm)

import math
import torch
from torch import nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


import torch
from torch import nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class AttentionBlock(nn.Module):
    def __init__(self, in_channels, key_channels, value_channels):
        super(AttentionBlock, self).__init__()

        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, 1),
            nn.BatchNorm2d(key_channels)
        )

        self.f_query = self.f_key
        self.f_value = nn.Conv2d(in_channels, value_channels, 1)
        self.softmax = nn.Softmax(dim=-1)  # 修改softmax的维度为-1

    def forward(self, x):
        batch, channels, height, width = x.size()
        proj_key = self.f_key(x).view(batch, -1, height*width)  # 将维度调整为BxKxN
        proj_query = self.f_query(x).view(batch, -1, height*width).permute(0, 2, 1)  # 将维度调整为BxNxK
        energy = torch.bmm(proj_query, proj_key)  # 进行批量矩阵乘法，得到BxNxN
        attention = self.softmax(energy)  # 应用softmax

        proj_value = self.f_value(x).view(batch, -1, height*width)  # 将value的维度调整为BxCxN

        out = torch.bmm(proj_value, attention.permute(0, 2, 1)).view(batch, -1, height, width)
        return out

class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
        super(Decoder, self).__init__()
        self.attn = AttentionBlock(304, 152, 304)
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48) # 使用BatchNorm替代GroupNorm
        self.relu = nn.ReLU()

        self.last_conv = nn.Sequential(
                            nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                            BatchNorm(256),
                            nn.ReLU(),
                            nn.Dropout(0.5),
                            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                            BatchNorm(256),
                            nn.ReLU(),
                            nn.Dropout(0.1),
                            nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()


    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.attn(x)

        x = self.last_conv(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_decoder(num_classes, backbone, BatchNorm):
    return Decoder(num_classes, backbone, BatchNorm)