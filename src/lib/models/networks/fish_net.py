from __future__ import division

import torch.nn as nn
import torch
import math

class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, mode='NORM', k=1, dilation=1):
        """
        Pre-act residual block, the middle transformations are bottle-necked
        :param inplanes:
        :param planes:
        :param stride:
        :param downsample:
        :param mode: NORM | UP
        :param k: times of additive
        """

        super(Bottleneck, self).__init__()
        self.mode = mode
        self.relu = nn.ReLU(inplace=True)
        self.k = k

        btnk_ch = planes // 2
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, btnk_ch, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm2d(btnk_ch)
        self.conv2 = nn.Conv2d(btnk_ch, btnk_ch, kernel_size=3, stride=stride, padding=dilation,
                               dilation=dilation, bias=False)

        self.bn3 = nn.BatchNorm2d(btnk_ch)
        self.conv3 = nn.Conv2d(btnk_ch, planes, kernel_size=1, bias=False)

        if mode == 'UP':
            self.shortcut = None
        elif inplanes != planes or stride > 1:
            self.shortcut = nn.Sequential(
                nn.BatchNorm2d(inplanes),
                self.relu,
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
            )
        else:
            self.shortcut = None

    def _pre_act_forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.mode == 'UP':
            residual = self.squeeze_idt(x)
        elif self.shortcut is not None:
            residual = self.shortcut(residual)

        out += residual

        return out

    def squeeze_idt(self, idt):
        n, c, h, w = idt.size()
        return idt.view(n, c // self.k, self.k, h, w).sum(2)

    def forward(self, x):
        out = self._pre_act_forward(x)
        return out



__all__ = ['fish']
def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :] 

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class Fish(nn.Module):
    def __init__(self, block, num_cls=1000, num_down_sample=5, num_up_sample=3, trans_map=(2, 1, 0, 6, 5, 4),
                 network_planes=None, num_res_blks=None, num_trans_blks=None, heads=None, head_conv=0):
        super(Fish, self).__init__()
        self.block = block
        self.trans_map = trans_map
        self.upsample = nn.Upsample(scale_factor=2)
        self.down_sample = nn.MaxPool2d(2, stride=2)
        self.num_cls = num_cls
        self.num_down = num_down_sample
        self.num_up = num_up_sample
        self.network_planes = network_planes[1:]
        self.depth = len(self.network_planes)
        self.num_trans_blks = num_trans_blks
        self.num_res_blks = num_res_blks
        self.heads = heads
        self.head_conv = head_conv
        self.fish = self._make_fish(network_planes[0])
        
    def _make_score(self, in_ch, out_ch=1000, has_pool=False):
        bn = nn.BatchNorm2d(in_ch)
        relu = nn.ReLU(inplace=True)
        conv_trans = nn.Conv2d(in_ch, in_ch // 2, kernel_size=1, bias=False)
        bn_out = nn.BatchNorm2d(in_ch // 2)
        conv = nn.Sequential(bn, relu, conv_trans, bn_out, relu)
        if has_pool:
            fc = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_ch // 2, out_ch, kernel_size=1, bias=True))
        else:
            fc = nn.Conv2d(in_ch // 2, out_ch, kernel_size=1, bias=True)
        return [conv, fc]

    def _make_se_block(self, in_ch, out_ch):
        bn = nn.BatchNorm2d(in_ch)
        sq_conv = nn.Conv2d(in_ch, out_ch // 16, kernel_size=1)
        ex_conv = nn.Conv2d(out_ch // 16, out_ch, kernel_size=1)
        return nn.Sequential(bn,
                             nn.ReLU(inplace=True),
                             nn.AdaptiveAvgPool2d(1),
                             sq_conv,
                             nn.ReLU(inplace=True),
                             ex_conv,
                             nn.Sigmoid())

    def _make_residual_block(self, inplanes, outplanes, nstage, is_up=False, k=1, dilation=1):
        layers = []

        if is_up:
            layers.append(self.block(inplanes, outplanes, mode='UP', dilation=dilation, k=k))
        else:
            layers.append(self.block(inplanes, outplanes, stride=1))
        for i in range(1, nstage):
            layers.append(self.block(outplanes, outplanes, stride=1, dilation=dilation))
        return nn.Sequential(*layers)

    def _make_stage(self, is_down_sample, inplanes, outplanes, n_blk, has_trans=True,
                    has_score=False, trans_planes=0, no_sampling=False, num_trans=2, **kwargs):
        sample_block = []
        if has_score:
            sample_block.extend(self._make_score(outplanes, outplanes * 2, has_pool=False))

        if no_sampling or is_down_sample:
            res_block = self._make_residual_block(inplanes, outplanes, n_blk, **kwargs)
        else:
            res_block = self._make_residual_block(inplanes, outplanes, n_blk, is_up=True, **kwargs)

        sample_block.append(res_block)

        if has_trans:
            trans_in_planes = self.in_planes if trans_planes == 0 else trans_planes
            sample_block.append(self._make_residual_block(trans_in_planes, trans_in_planes, num_trans))

        if not no_sampling and is_down_sample:
            sample_block.append(self.down_sample)
        elif not no_sampling:  # Up-Sample
            sample_block.append(self.upsample)

        return nn.ModuleList(sample_block)

    def _make_fish(self, in_planes):
        def get_trans_planes(index):
            map_id = self.trans_map[index-self.num_down-1] - 1
            p = in_planes if map_id == -1 else cated_planes[map_id]
            return p

        def get_trans_blk(index):
            return self.num_trans_blks[index-self.num_down-1]

        def get_cur_planes(index):
            return self.network_planes[index]

        def get_blk_num(index):
            return self.num_res_blks[index]

        cated_planes, fish = [in_planes] * self.depth, []
        for i in range(self.depth):
            # even num for down-sample, odd for up-sample
            is_down, has_trans, no_sampling = i not in range(self.num_down, self.num_down+self.num_up+1),\
                                              i > self.num_down, i == self.num_down
            cur_planes, trans_planes, cur_blocks, num_trans =\
                get_cur_planes(i), get_trans_planes(i), get_blk_num(i), get_trans_blk(i)

            stg_args = [is_down, cated_planes[i - 1], cur_planes, cur_blocks]

            if is_down or no_sampling:
                k, dilation = 1, 1
            else:
                k, dilation = cated_planes[i - 1] // cur_planes, 2 ** (i-self.num_down-1)

            sample_block = self._make_stage(*stg_args, has_trans=has_trans, trans_planes=trans_planes,
                                            has_score=(i==self.num_down), num_trans=num_trans, k=k, dilation=dilation,
                                            no_sampling=no_sampling)
            if i == self.depth - 1:
                sample_block.extend(self._make_score(cur_planes + trans_planes, out_ch=self.num_cls, has_pool=True))
            elif i == self.num_down:
                sample_block.append(nn.Sequential(self._make_se_block(cur_planes*2, cur_planes)))

            if i == self.num_down-1:
                cated_planes[i] = cur_planes * 2
            elif has_trans:
                cated_planes[i] = cur_planes + trans_planes
            else:
                cated_planes[i] = cur_planes
            fish.append(sample_block)
        return nn.ModuleList(fish)

    def _fish_forward(self, all_feat):
        def _concat(a, b):
            return torch.cat([a, b], dim=1)

        def stage_factory(*blks):
            def stage_forward(*inputs):
                if stg_id < self.num_down:  # tail
                    tail_blk = nn.Sequential(*blks[:2])
                    return tail_blk(*inputs)
                elif stg_id == self.num_down:
                    score_blks = nn.Sequential(*blks[:2])
                    score_feat = score_blks(inputs[0])
                    att_feat = blks[3](score_feat)
                    return blks[2](score_feat) * att_feat + att_feat
                else:  # refine
                    feat_trunk = blks[2](blks[0](inputs[0]))
                    feat_branch = blks[1](inputs[1])
                return _concat(feat_trunk, feat_branch)
            return stage_forward

        stg_id = 0
        # tail:
        while stg_id < self.depth:
            stg_blk = stage_factory(*self.fish[stg_id])
            if stg_id <= self.num_down:
                in_feat = [all_feat[stg_id]]
            else:
                trans_id = self.trans_map[stg_id-self.num_down-1]
                in_feat = [all_feat[stg_id], all_feat[trans_id]]

            all_feat[stg_id + 1] = stg_blk(*in_feat)
            stg_id += 1
            # loop exit
            if stg_id == self.depth:
                score_feat = self.fish[self.depth-1][-2](all_feat[-1])
                score = self.fish[self.depth-1][-1](score_feat)
                return score

    def forward(self, x):
        all_feat = [None] * (self.depth + 1)
        all_feat[0] = x
        return self._fish_forward(all_feat)


class FishNet(nn.Module):
    def __init__(self, block, **kwargs):
        super(FishNet, self).__init__()

        inplanes = kwargs['network_planes'][0]
        outplanes = inplanes
        # resolution: 224x224
        self.conv1 = self._conv_bn_relu(3, inplanes // 2, stride=2)
        self.conv2 = self._conv_bn_relu(inplanes // 2, inplanes // 2)
        self.conv3 = self._conv_bn_relu(inplanes //2 , outplanes)
        # self.conv4 = self._conv_bn_relu(outplanes, outplanes)
        self.pool1 = nn.MaxPool2d(3, padding=1, stride=2)
        # construct fish, resolution 56x56
        self.fish = Fish(block, **kwargs)
        self._init_weights()
        
        self.heads = kwargs['heads']
        self.head_conv = kwargs['head_conv']
        # self.head_conv = 32
        for head in self.heads:
            classes = self.heads[head]
            if self.head_conv > 0:
              fc = nn.Sequential(
                  nn.Conv2d(inplanes, self.head_conv,
                    kernel_size=3, padding=1, bias=True),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(self.head_conv, classes, 
                    kernel_size=1, stride=1, 
                    padding=1 // 2, bias=True))
              if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
              else:
                fill_fc_weights(fc)
            else:
              fc = nn.Conv2d(inplanes, classes, 
                  kernel_size=1, stride=1, 
                  padding=1 // 2, bias=True)
              if 'hm' in head:
                fc.bias.data.fill_(-2.19)
              else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def _conv_bn_relu(self, in_ch, out_ch, stride=1):
        return nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=stride, bias=False),
                             nn.BatchNorm2d(out_ch),
                             nn.ReLU(inplace=True))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        print(x.shape)
        x = self.conv1(x)
        print(x.shape)
        x = self.conv2(x)
        print(x.shape)
        x = self.conv3(x)
        print(x.shape)
        # x = self.conv4(x)
        # print(x.shape)
        x = self.pool1(x)
        print(x.shape)
        score = self.fish(x)
        print(score.shape)
        # 1*1 output
        out = score.view(x.size(0), -1)
        print(out.shape)
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(score)
        return [ret]

def fish(**kwargs):
    return FishNet(Bottleneck, **kwargs)

def fishnet99(**kwargs):
    """

    :return:
    """
    net_cfg = {
        #  input size:   [224, 56, 28,  14  |  7,   7,  14,  28 | 56,   28,  14]
        # output size:   [56,  28, 14,   7  |  7,  14,  28,  56 | 28,   14,   7]
        #                  |    |    |   |     |    |    |    |    |     |    |
        'network_planes': [64, 128, 256, 512, 512, 512, 384, 256, 320, 832, 1600],
        'num_res_blks': [2, 2, 6, 2, 1, 1, 1, 1, 2, 2],
        'num_trans_blks': [1, 1, 1, 1, 1, 4],
        'num_cls': kwargs['head_conv'],
        'num_down_sample': 3,
        'num_up_sample': 3,
    }
    cfg = {**net_cfg, **kwargs}
    return fish(**cfg)


def fishnet150(**kwargs):
    """

    :return:
    """
    net_cfg = {
        #  input size:   [224, 56, 28,  14  |  7,   7,  14,  28 | 56,   28,  14]
        # output size:   [56,  28, 14,   7  |  7,  14,  28,  56 | 28,   14,   7]
        #                  |    |    |   |     |    |    |    |    |     |    |
        'network_planes': [64, 128, 256, 512, 512, 512, 384, 256, 320, 832, 1600],
        'num_res_blks': [2, 4, 8, 4, 2, 2, 2, 2, 2, 4],
        'num_trans_blks': [2, 2, 2, 2, 2, 4],
        'num_cls': kwargs['head_conv'],
        'num_down_sample': 3,
        'num_up_sample': 3,
    }
    cfg = {**net_cfg, **kwargs}
    return fish(**cfg)


def fishnet201(**kwargs):
    """

    :return:
    """
    net_cfg = {
        #  input size:   [224, 56, 28,  14  |  7,   7,  14,  28 | 56,   28,  14]
        # output size:   [56,  28, 14,   7  |  7,  14,  28,  56 | 28,   14,   7]
        #                  |    |    |   |     |    |    |    |    |     |    |
        'network_planes': [64, 128, 256, 512, 512, 512, 384, 256, 320, 832, 1600],
        'num_res_blks': [3, 4, 12, 4, 2, 2, 2, 2, 3, 10],
        'num_trans_blks': [2, 2, 2, 2, 2, 9],
        'num_cls': kwargs['head_conv'],
        'num_down_sample': 3,
        'num_up_sample': 3,
    }
    cfg = {**net_cfg, **kwargs}
    return fish(**cfg)

fish_models = {
    'fishnet99': fishnet99,
    'fishnet150': fishnet150,
    'fishnet201': fishnet201
}
def get_fish(num_layers, heads, head_conv):
    fish_cfg = {
        'heads': heads,
        'head_conv': 64,
    }
    fish_net = 'fishnet{}'.format(num_layers)
    return fish_models[fish_net](**fish_cfg)