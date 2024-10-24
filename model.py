import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
import sys,os
sys.path.append('/workspace/TFI-ct/')
from models.resnet import *
from models.resnet import resnet18
import cv2 as cv
import numpy as np
from PIL import Image


GlobalAvgPool2D = lambda: nn.AdaptiveAvgPool2d(1)   

class TemporalFeatureInteractionModule(nn.Module):
    def __init__(self, in_d, out_d):
        super(TemporalFeatureInteractionModule, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.conv_sub = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True)
        )
        self.conv_diff_enh1 = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True)
        )
        self.conv_diff_enh2 = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True)
        )
        self.conv_cat = nn.Sequential(
            nn.Conv2d(self.in_d * 2, self.in_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True)
        )
        self.conv_dr = nn.Sequential(
            nn.Conv2d(self.in_d, self.out_d, kernel_size=1, bias=True),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        # difference enhance
        x_sub = self.conv_sub(torch.abs(x1 - x2))

        
        x1 = self.conv_diff_enh1(x1.mul(x_sub) + x1)
        x2 = self.conv_diff_enh2(x2.mul(x_sub) + x2)
        # fusion
        x_f = torch.cat([x1, x2], dim=1)
        x_f = self.conv_cat(x_f)
        x = x_sub + x_f
        x = self.conv_dr(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class ChangeInformationExtractionModule(nn.Module):
    def __init__(self, in_d, out_d):
        super(ChangeInformationExtractionModule, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.ca = ChannelAttention(self.in_d * 4, ratio=16)
        self.conv_dr = nn.Sequential(
            nn.Conv2d(self.in_d * 4, self.in_d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True)
        )
        self.pools_sizes = [2, 4, 8]
        self.conv_pool1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=self.pools_sizes[0], stride=self.pools_sizes[0]),
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.conv_pool2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=self.pools_sizes[0], stride=self.pools_sizes[0]),
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.conv_pool3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=self.pools_sizes[0], stride=self.pools_sizes[0]),
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, d5, d4, d3, d2):
        # upsampling
        d5 = F.interpolate(d5, d2.size()[2:], mode='bilinear', align_corners=True)
        d4 = F.interpolate(d4, d2.size()[2:], mode='bilinear', align_corners=True)
        d3 = F.interpolate(d3, d2.size()[2:], mode='bilinear', align_corners=True)
        # fusion
        x = torch.cat([d5, d4, d3, d2], dim=1)
        x_ca = self.ca(x)
        x = x * x_ca
        x = self.conv_dr(x)

        # feature = x[0:1, 0:64, 0:64, 0:64]
        # vis.visulize_features(feature)

        # pooling
        d2 = x
        d3 = self.conv_pool1(x)
        d4 = self.conv_pool2(x)
        d5 = self.conv_pool3(x)

        return d5, d4, d3, d2


class GuidedRefinementModule(nn.Module):
    def __init__(self, in_d, out_d):
        super(GuidedRefinementModule, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.conv_d5 = nn.Sequential(
            nn.Conv2d(self.in_d, self.out_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(inplace=True)
        )
        self.conv_d4 = nn.Sequential(
            nn.Conv2d(self.in_d, self.out_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(inplace=True)
        )
        self.conv_d3 = nn.Sequential(
            nn.Conv2d(self.in_d, self.out_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(inplace=True)
        )
        self.conv_d2 = nn.Sequential(
            nn.Conv2d(self.in_d, self.out_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(inplace=True)
        )

    def forward(self, d5, d4, d3, d2, d5_p, d4_p, d3_p, d2_p):
        # feature refinement
        d5 = self.conv_d5(d5_p + d5)
        d4 = self.conv_d4(d4_p + d4)
        d3 = self.conv_d3(d3_p + d3)
        d2 = self.conv_d2(d2_p + d2)

        return d5, d4, d3, d2


class Decoder(nn.Module):
    def __init__(self, in_d, out_d):
        super(Decoder, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.conv_sum1 = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True)
        )
        self.conv_sum2 = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True)
        )
        self.conv_sum3 = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True)
        )
        self.cls = nn.Conv2d(self.in_d, self.out_d, kernel_size=1, bias=False)

    def forward(self, d5, d4, d3, d2):

        d5 = F.interpolate(d5, d4.size()[2:], mode='bilinear', align_corners=True)
        d4 = self.conv_sum1(d4 + d5)
        d4 = F.interpolate(d4, d3.size()[2:], mode='bilinear', align_corners=True)
        d3 = self.conv_sum1(d3 + d4)
        d3 = F.interpolate(d3, d2.size()[2:], mode='bilinear', align_corners=True)
        d2 = self.conv_sum1(d2 + d3)

        mask = self.cls(d2)

        return mask


class CrossAtt(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels

        self.query1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size = 1, stride = 1)
        self.key1   = nn.Conv2d(in_channels, in_channels // 4, kernel_size = 1, stride = 1)
        self.value1 = nn.Conv2d(in_channels, in_channels, kernel_size = 1, stride = 1)

        self.query2 = nn.Conv2d(in_channels, in_channels // 8, kernel_size = 1, stride = 1)
        self.key2   = nn.Conv2d(in_channels, in_channels // 4, kernel_size = 1, stride = 1)
        self.value2 = nn.Conv2d(in_channels, in_channels, kernel_size = 1, stride = 1)

        self.gamma = nn.Parameter(torch.zeros(1)) 
        self.softmax = nn.Softmax(dim = -1)

        self.conv_cat = nn.Sequential(nn.Conv2d(in_channels*2, out_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU()) # conv_f

    def forward(self, input1, input2):
        batch_size, channels, height, width = input1.shape
        q1 = self.query1(input1)
        k1 = self.key1(input1).view(batch_size, -1, height * width)
        v1 = self.value1(input1).view(batch_size, -1, height * width)

        q2 = self.query2(input2) 
        k2 = self.key2(input2).view(batch_size, -1, height * width)
        v2 = self.value2(input2).view(batch_size, -1, height * width)

        q = torch.cat([q1,q2],1).view(batch_size, -1, height * width).permute(0, 2, 1)
        attn_matrix1 = torch.bmm(q, k1)  
        attn_matrix1 = self.softmax(attn_matrix1)
        out1 = torch.bmm(v1, attn_matrix1.permute(0, 2, 1)) 
        out1 = out1.view(*input1.shape)
        out1 = self.gamma * out1 + input1

        attn_matrix2 = torch.bmm(q, k2) 
        attn_matrix2 = self.softmax(attn_matrix2)
        out2 = torch.bmm(v2, attn_matrix2.permute(0, 2, 1))  
        out2 = out2.view(*input2.shape)
        out2 = self.gamma * out2 + input2

        feat_sum = self.conv_cat(torch.cat([out1,out2],1))
        return feat_sum, out1, out2

class Cross_transformer(nn.Module):
    def __init__(self, in_channels = 48):
        super(Cross_transformer, self).__init__()
        self.fa = nn.Linear(in_channels , in_channels, bias=False)
        self.fb = nn.Linear(in_channels, in_channels, bias=False)
        self.fc = nn.Linear(in_channels , in_channels, bias=False)
        self.fd = nn.Linear(in_channels, in_channels, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.to_out = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.gamma_cam_lay3 = nn.Parameter(torch.zeros(1))
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
    
    def attention_layer(self, q, k, v, m_batchsize, C, height, width):
        k = k.permute(0, 2, 1)
        energy = torch.bmm(q, k)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        out = torch.bmm(attention, v)
        out = out.view(m_batchsize, C, height, width)
        
        return out
        
        
    def forward(self, input_feature, features):    
        fa = input_feature
        fb = features[0]
        fc = features[1]
        fd = features[2]
        

        m_batchsize, C, height, width = fa.size()
        fa = self.fa(fa.view(m_batchsize, C, -1).permute(0, 2, 1)).permute(0, 2, 1)
        fb = self.fb(fb.view(m_batchsize, C, -1).permute(0, 2, 1)).permute(0, 2, 1)
        fc = self.fc(fc.view(m_batchsize, C, -1).permute(0, 2, 1)).permute(0, 2, 1)
        fd = self.fd(fd.view(m_batchsize, C, -1).permute(0, 2, 1)).permute(0, 2, 1)
        
        
        qkv_1 = self.attention_layer(fa, fa, fa, m_batchsize, C, height, width)
        qkv_2 = self.attention_layer(fa, fb, fb, m_batchsize, C, height, width)  
        qkv_3 = self.attention_layer(fa, fc, fc, m_batchsize, C, height, width)
        qkv_4 = self.attention_layer(fa, fd, fd, m_batchsize, C, height, width)
        
        atten = self.fuse(torch.cat((qkv_1, qkv_2, qkv_3, qkv_4), dim = 1))
              

        out = self.gamma_cam_lay3 * atten + input_feature

        out = self.to_out(out)
        
        return out


class SceneRelation(nn.Module):
    def __init__(self,
                 in_channels,
                 channel_list,
                 out_channels,
                 scale_aware_proj=True):
        super(SceneRelation, self).__init__()
        self.scale_aware_proj = scale_aware_proj

        if scale_aware_proj:
            self.scene_encoder = nn.ModuleList(
                [nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.ReLU(True),
                    nn.Conv2d(out_channels, out_channels, 1),
                ) for _ in range(len(channel_list))]
            )
        else:
            # 2mlp
            self.scene_encoder = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 1),
            )
        self.content_encoders = nn.ModuleList()
        self.feature_reencoders = nn.ModuleList()
        for c in channel_list:
            self.content_encoders.append(
                nn.Sequential(
                    nn.Conv2d(c, out_channels, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True)
                )
            )
            self.feature_reencoders.append(
                nn.Sequential(
                    nn.Conv2d(c, out_channels, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True)
                )
            )

        self.normalizer = nn.Sigmoid()
        
        

    def forward(self, scene_feature, features: list):           #传入参数 1、全局平均池化后的本层的特征（1，64，1，1）   2、所有特征层的特征（原尺寸8，16，32，64）
        content_feats = [c_en(p_feat) for c_en, p_feat in zip(self.content_encoders, features)]

        scene_feats = [op(scene_feature) for op in self.scene_encoder]
        relations = [self.normalizer(sf) * cf for sf, cf in
                         zip(scene_feats, content_feats)]

        
        return relations

class PSPModule(nn.Module):
    def __init__(self, in_channels, bin_sizes=[1, 2, 4, 6]):
        super(PSPModule, self).__init__()
        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s) 
                                                        for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+(out_channels * len(bin_sizes)), in_channels, 
                                    kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(conv, bn, relu)
    
    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]

        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear', 
                                        align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output




class BaseNet(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(BaseNet, self).__init__()
    def __init__(self, input_nc, output_nc):
        super(BaseNet, self).__init__()
        # f_channels = [64, 128, 256, 512]
        f_channels = [256, 512, 1024, 2048]
        self.backbone = resnet101(pretrained=True)
        self.mid_d = 64
        # self.TFIM5 = TemporalFeatureInteractionModule(512, self.mid_d)
        # self.TFIM4 = TemporalFeatureInteractionModule(256, self.mid_d)
        # self.TFIM3 = TemporalFeatureInteractionModule(128, self.mid_d)
        # self.TFIM2 = TemporalFeatureInteractionModule(64, self.mid_d)

        self.TFIM5 = TemporalFeatureInteractionModule(f_channels[3], f_channels[3])
        self.TFIM4 = TemporalFeatureInteractionModule(f_channels[2], f_channels[2])
        self.TFIM3 = TemporalFeatureInteractionModule(f_channels[1], f_channels[1])
        self.TFIM2 = TemporalFeatureInteractionModule(f_channels[0], f_channels[0])

        self.CIEM1 = ChangeInformationExtractionModule(self.mid_d, output_nc)
        self.GRM1 = GuidedRefinementModule(self.mid_d, self.mid_d)

        self.CIEM2 = ChangeInformationExtractionModule(self.mid_d, output_nc)
        self.GRM2 = GuidedRefinementModule(self.mid_d, self.mid_d)

        self.conv_pool_size_5 = nn.Sequential(
            nn.AvgPool2d(kernel_size=4, stride=4),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, bias=False))
        
        self.conv_pool_size_4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, bias=False))
        
        self.conv_pool_size_3 = nn.Sequential(
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, bias=False))
        
        # self.conv_pool_size_2 = nn.Sequential(
        #     F.interpolate(input, scale_factor=2, mode='nearest'),
        #     nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, bias=False))


        self.cross5 = CrossAtt(f_channels[3], f_channels[3]) 
        self.cross4 = CrossAtt(f_channels[2], f_channels[2]) 
        self.cross3 = CrossAtt(f_channels[1], f_channels[1]) 
        self.cross2 = CrossAtt(f_channels[0], f_channels[0])


        self.PPN = PSPModule(f_channels[-1])
        # Scale-aware
        self.sig = nn.Sigmoid()
        self.gap = GlobalAvgPool2D()
        self.sr1 = SceneRelation(in_channels = f_channels[3], channel_list = f_channels, out_channels = f_channels[3], scale_aware_proj=True)
        self.sr2 = SceneRelation(in_channels = f_channels[2], channel_list = f_channels, out_channels = f_channels[2], scale_aware_proj=True)
        self.sr3 = SceneRelation(in_channels = f_channels[1], channel_list = f_channels, out_channels = f_channels[1], scale_aware_proj=True)
        self.sr4 = SceneRelation(in_channels = f_channels[0], channel_list =f_channels, out_channels = f_channels[0], scale_aware_proj=True)

        # Cross transformer
        self.Cross_transformer1 =  Cross_transformer(in_channels = f_channels[3])
        self.Cross_transformer2 =  Cross_transformer(in_channels = f_channels[2])
        self.Cross_transformer3 =  Cross_transformer(in_channels = f_channels[1])
        self.Cross_transformer4 =  Cross_transformer(in_channels = f_channels[0])

        # self.conv64_128 =  nn.Conv2d(64, 128, kernel_size=1)
        # self.conv64_128 =  nn.Conv2d(64, 128, kernel_size=1)
        # self.conv64_256 =  nn.Conv2d(64, 256, kernel_size=1)
        # self.conv64_512 =  nn.Conv2d(64, 512, kernel_size=1)

        self.conv512_64 = nn.Conv2d(f_channels[3],64,kernel_size =1)
        self.conv256_64 = nn.Conv2d(f_channels[2],64,kernel_size =1)
        self.conv128_64 = nn.Conv2d(f_channels[1],64,kernel_size =1)
        self.conv64_64_s = nn.Conv2d(f_channels[0],64,kernel_size =1)

        self.conv64_64 = nn.Conv2d(f_channels[0],64, kernel_size = 3,stride = 2,padding = 1)   #下采样

        self.decoder = Decoder(self.mid_d, output_nc)




    def forward(self, x1, x2):
        # forward backbone resnet
        x1_1, x1_2, x1_3, x1_4, x1_5 = self.backbone.base_forward(x1)
        x2_1, x2_2, x2_3, x2_4, x2_5 = self.backbone.base_forward(x2)

        #交叉注意力
        cross_result2, cur1_2, cur2_2 = self.cross2(x1_2, x2_2)    #256
        cross_result3, cur1_3, cur2_3 = self.cross3(x1_3, x2_3)    #128
        cross_result4, cur1_4, cur2_4 = self.cross4(x1_4, x2_4)    #64
        cross_result5, cur1_5, cur2_5 = self.cross5(x1_5, x2_5)    #32

        # print('cr',cur1_5.size())
        # print('cr',cur1_4.size())
        # print('cr',cur1_3.size())
        # print('cr',cur1_2.size())
        
        
        # feature difference
        d5 = self.TFIM5(cur1_5, cur2_5)  # 1/32 512-64
        d4 = self.TFIM4(cur1_4, cur2_4)  # 1/16 256-64
        d3 = self.TFIM3(cur1_3, cur2_3)  # 1/8  128-64
        d2 = self.TFIM2(cur1_2, cur2_2)  # 1/4  64-64

        # print('dd',d5.size())
        # print('dd',d4.size())
        # print('dd',d3.size())
        # print('dd',d2.size())

        # dd torch.Size([1, 64, 8, 8])
        # dd torch.Size([1, 64, 16, 16])
        # dd torch.Size([1, 64, 32, 32])
        # dd torch.Size([1, 64, 64, 64])

        # d5 = self.TFIM5(x1_5, x2_5)  # 1/32 512-64
        # d4 = self.TFIM4(x1_4, x2_4)  # 1/16 256-64
        # d3 = self.TFIM3(x1_3, x2_3)  # 1/8  128-64
        # d2 = self.TFIM2(x1_2, x2_2)  # 1/4  64-64
        features = [d2,d3,d4,d5] #[64, 128, 256, 512]
        # features = [d5,d4,d3,d2]
        # features[-1] = self.PPN(features[-1])
        c6 = self.gap(features[-1])   #(1,64,1,1)
        c7 = self.gap(features[-2])    
        c8 = self.gap(features[-3])    
        c9 = self.gap(features[-4])   
        
        features1, features2, features3, features4 = [], [], [], []

        # print('++',c6.size())
        # print('++',c7.size())
        # print('++',c8.size())
        # print('++',c9.size())

        features1[:] = [F.interpolate(feature, size=(64, 64), mode='nearest') for feature in features[:]]

        # print(len(features1))

        # print('++',features1[0].size())
        # print('++',features1[1].size())
        # print('++',features1[2].size())
        # print('++',features1[3].size())
        # print('-'*100)
        
        # features1[1] = self.conv64_128(features1[1])
        # features1[2] = self.conv64_256(features1[2])
        # features1[3] = self.conv64_512(features1[3])
        # c6 = self.conv64_512(c6)
        # print('++',features1[0].size())
        # print('++',features1[1].size())
        # print('++',features1[2].size())
        # print('++',features1[3].size())

        list_3 = self.sr1(c6, features1) 
        fe3 = self.Cross_transformer1(list_3[-1], [list_3[-2], list_3[-3], list_3[-4]]) 
        
        features2[:] = [F.interpolate(feature, size=(64, 64), mode='nearest') for feature in features[:]]




        list_2 = self.sr2(c7, features2) 
        fe2 = self.Cross_transformer2(list_2[-2], [list_2[-1], list_2[-3], list_2[-4]]) 
        
        features3[:] = [F.interpolate(feature, size=(64, 64), mode='nearest') for feature in features[:]]
        list_1 = self.sr3(c8, features3) 
        fe1 = self.Cross_transformer3(list_1[-3], [list_1[-1], list_1[-2], list_1[-4]]) 
        
        features4[:] = [F.interpolate(feature, size=(128, 128), mode='nearest') for feature in features[:]]
        list_0 = self.sr4(c9, features4) 
        fe0 = self.Cross_transformer4(list_0[-4], [list_0[-1], list_0[-2], list_0[-3]]) 

        # print('fe',fe3.size())
        # print('fe',fe2.size())
        # print('fe',fe1.size())
        # print('fe',fe0.size())


        fe3 = self.conv512_64(fe3)
        fe2 = self.conv256_64(fe2)
        fe1 = self.conv128_64(fe1)
        fe0 = self.conv64_64(fe0)
        

        d5 = self.conv512_64(d5)
        d4 = self.conv256_64(d4)
        d3 = self.conv128_64(d3)
        d2 = self.conv64_64_s(d2)

        # change information guided refinement 1
        d5_p, d4_p, d3_p, d2_p = self.CIEM1(fe3, fe2, fe1, fe0)

        # print('dd',d5.size())
        # print('dd',d4.size())
        # print('dd',d3.size())
        # print('dd',d2.size())

        # print('dp',d5_p.size())
        # print('dp',d4_p.size())
        # print('dp',d3_p.size())
        # print('dp',d2_p.size())
        # exit()

        d5, d4, d3, d2 = self.GRM1(d5, d4, d3, d2, d5_p, d4_p, d3_p, d2_p)

        # change information guided refinement 2
        d5_p, d4_p, d3_p, d2_p = self.CIEM2(d5, d4, d3, d2)
        d5, d4, d3, d2 = self.GRM2(d5, d4, d3, d2, d5_p, d4_p, d3_p, d2_p)

        # decoder
        mask = self.decoder(d5, d4, d3, d2)
        mask = F.interpolate(mask, x1.size()[2:], mode='bilinear', align_corners=True)
        mask = torch.sigmoid(mask)

        # print("---------Done!----------------")
        # print(mask.shape)
        
        return mask



def vis_tensor(tensor):
    pred = tensor
    # pred = torch.where(tensor > 0.5, torch.ones_like(tensor), torch.zeros_like(tensor)).long()
    array1=pred.detach().numpy()#将tensor数据转为numpy数据
    maxValue=array1.max()
    array1=array1*255/maxValue#normalize，将图像数据扩展到[0,255]
    mat=np.uint8(array1)#float32-->uint8
    print('mat_shape:',mat.shape)
    mat.resize(1,256,256)
    mat=mat.transpose(1,2,0)#mat_shape: (982, 814，3)
    
    mat = 0.4 * cv.applyColorMap(mat, cv.COLORMAP_JET)
    # mat = 0.4 * cv.applyColorMap(mat, cv.COLORMAP_RAINBOW)
    
    pa = "/workspace/vis-cam/1.png"
    cv.imwrite(pa,mat)


# def visualize_heatmap(image, mask):
#     '''
#     Save the heatmap of ones
#     '''
#     # masks = 0
#     # (mask).astype(np.uint8)
#     # mask->heatmap
#     heatmap = cv.applyColorMap(mask, cv.COLORMAP_JET)
#     heatmap = np.float32(heatmap)

#     heatmap = cv.resize(heatmap, (image.shape[1], image.shape[0]))    # same shape

#     # merge heatmap to original image
#     cam = 0.4*heatmap + 0.6*np.float32(image)
#     print(cam)
#     return cam

if __name__ == "__main__" :

    # print('==='*10)
    model = BaseNet(3, 1)
    img1 = np.array(Image.open("/workspace/TFI-ct/samples/test/A/test_102_0512_0000.png"))
    img2 = np.array(Image.open("/workspace/TFI-ct/samples/test/B/test_102_0512_0000.png"))
    
    transf = transforms.ToTensor()
    x1 = transf(img1)
    x2 = transf(img2)
    x1 = x1.unsqueeze(0)
    x2 = x2.unsqueeze(0)

    print(x1.shape)
    mask= model.forward(x1,x2)
    # cam1 = visualize_heatmap(img1,mask)
    # print(cam1.shape)
    # vi = vis_tensor(mask)