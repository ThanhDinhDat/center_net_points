import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from .DCNv2.dcn_v2 import DCN
import torch.utils.model_zoo as model_zoo

BN_MOMENTUM = 0.1

__all__ = ['VoVNet', 'vovnet27_slim', 'vovnet39', 'vovnet57']


model_urls = {
    'vovnet39': 'https://www.dropbox.com/s/1lnzsgnixd8gjra/vovnet39_torchvision.pth',
    'vovnet57': 'https://www.dropbox.com/s/6bfu9gstbwfw31m/vovnet57_torchvision.pth'
}

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def conv3x3(in_channels, out_channels, module_name, postfix,
            stride=1, groups=1, kernel_size=3, padding=1):
    """3x3 convolution with padding"""
    return [
        ('{}_{}/conv'.format(module_name, postfix),
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      groups=groups,
                      bias=False)),
        ('{}_{}/norm'.format(module_name, postfix),
            nn.BatchNorm2d(out_channels)),
        ('{}_{}/relu'.format(module_name, postfix),
            nn.ReLU(inplace=True)),
    ]


def conv1x1(in_channels, out_channels, module_name, postfix,
            stride=1, groups=1, kernel_size=1, padding=0):
    """1x1 convolution"""
    return [
        ('{}_{}/conv'.format(module_name, postfix),
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      groups=groups,
                      bias=False)),
        ('{}_{}/norm'.format(module_name, postfix),
            nn.BatchNorm2d(out_channels)),
        ('{}_{}/relu'.format(module_name, postfix),
            nn.ReLU(inplace=True)),
    ]


class _OSA_module(nn.Module):
    def __init__(self,
                 in_ch,
                 stage_ch,
                 concat_ch,
                 layer_per_block,
                 module_name,
                 identity=False):
        super(_OSA_module, self).__init__()

        self.identity = identity
        self.layers = nn.ModuleList()
        in_channel = in_ch
        for i in range(layer_per_block):
            self.layers.append(nn.Sequential(
                OrderedDict(conv3x3(in_channel, stage_ch, module_name, i))))
            in_channel = stage_ch

        # feature aggregation
        in_channel = in_ch + layer_per_block * stage_ch
        self.concat = nn.Sequential(
            OrderedDict(conv1x1(in_channel, concat_ch, module_name, 'concat')))

    def forward(self, x):
        identity_feat = x
        pool_idx=[]
        output = []
        output.append(x)
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        
        x = torch.cat(output, dim=1)
        xt = self.concat(x)
        
        if self.identity:
            xt = xt + identity_feat
        
        return xt

class _OSA_stage(nn.Sequential):
    def __init__(self,
                 in_ch,
                 stage_ch,
                 concat_ch,
                 block_per_stage,
                 layer_per_block,
                 stage_num):
        super(_OSA_stage, self).__init__()

        if not stage_num == 2:
            self.add_module('Pooling',
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))

        module_name = 'OSA{}_1'.format(stage_num)
        self.add_module(module_name,
            _OSA_module(in_ch,
                        stage_ch,
                        concat_ch,
                        layer_per_block,
                        module_name))
        for i in range(block_per_stage-1):
            module_name = 'OSA{}_{}'.format(stage_num, i+2)
            self.add_module(module_name,
                _OSA_module(concat_ch,
                            stage_ch,
                            concat_ch,
                            layer_per_block,
                            module_name,
                            identity=True))

        # if not stage_num == 2:
        #     self.add_module('UnPooling',
        #         nn.MaxUnpool2d(kernel_size=3, stride=2 ))


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

class VoVNet(nn.Module):
    def __init__(self, 
                 config_stage_ch,
                 config_concat_ch,
                 block_per_stage,
                 layer_per_block,
                 heads, head_conv,
                 num_classes=1000):
        super(VoVNet, self).__init__()

        # Final channel size after vov net
        self.inplanes = config_concat_ch[-1]

        # Stem module
        stem = conv3x3(3,   64, 'stem', '1', 2)
        stem += conv3x3(64,  64, 'stem', '2', 1)
        stem += conv3x3(64, 128, 'stem', '3', 2)
        self.add_module('stem', nn.Sequential(OrderedDict(stem)))

        stem_out_ch = [128]
        in_ch_list = stem_out_ch + config_concat_ch[:-1]
        self.stage_names = []
        for i in range(4): #num_stages
            name = 'stage%d' % (i+2)
            self.stage_names.append(name)
            self.add_module(name,
                            _OSA_stage(in_ch_list[i],
                                       config_stage_ch[i],
                                       config_concat_ch[i],
                                       block_per_stage[i],
                                       layer_per_block,
                                       i+2))

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            3,
            [256, 128, 64],
            [4, 4, 4],
        )

        self.heads = heads
        self.head_conv = head_conv
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                  nn.Conv2d(64, head_conv,
                    kernel_size=3, padding=1, bias=True),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(head_conv, classes, 
                    kernel_size=1, stride=1, 
                    padding=0, bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(64, classes, 
                  kernel_size=1, stride=1, 
                  padding=0, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            fc = DCN(self.inplanes, planes, 
                    kernel_size=(3,3), stride=1,
                    padding=1, dilation=1, deformable_groups=1)
            # fc = nn.Conv2d(self.inplanes, planes,
            #         kernel_size=3, stride=1, 
            #         padding=1, dilation=1, bias=False)
            # fill_fc_weights(fc)
            up = nn.ConvTranspose2d(
                    in_channels=planes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False)
            fill_up_weights(up)

            layers.append(fc)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            layers.append(up)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        pool_idx=[]
        outputs={}
        for name in self.stage_names:
            x = getattr(self, name)(x)

        x = self.deconv_layers(x)
        for head in self.heads:
            outputs[head] = self.__getattr__(head)(x)

        return [outputs]


def _vovnet(arch,
            config_stage_ch,
            config_concat_ch,
            block_per_stage,
            layer_per_block,
            pretrained,
            progress,
            **kwargs):
    model = VoVNet(config_stage_ch, config_concat_ch,
                   block_per_stage, layer_per_block,
                   **kwargs)
    pretrained_path = {
        'vovnet57': "../models/vovnet57_torchvision.pth",
        'vovnet39': "../models/vovnet57_torchvision.pth"
    }
    if pretrained:
        path = pretrained_path[arch]
        pretrained_state_dict = torch.load(path)
        print("==> Loaded pretrained model from {}".format(path))

        # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/3
        new_state_dict = OrderedDict()
        for k, v in pretrained_state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v

        new_dict = OrderedDict()
        non_load = []
        for k, v in model.state_dict().items():
            if k in new_state_dict and v.size() == new_state_dict[k].size():
                new_dict[k] = new_state_dict[k]
            else:
                new_dict[k] = v
                non_load.append(k)

        print("This weight is not loaded from checkpoint: {}".format(non_load))
        model.load_state_dict(new_dict)
    return model


def vovnet57(pretrained=False, progress=True, **kwargs):
    r"""Constructs a VoVNet-57 model as described in 
    `"An Energy and GPU-Computation Efficient Backbone Networks"
    <https://arxiv.org/abs/1904.09730>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vovnet('vovnet57', [128, 160, 192, 224], [256, 512, 768, 1024],
                    [1,1,4,3], 5, pretrained, progress, **kwargs)


def vovnet39(pretrained=False, progress=True, **kwargs):
    r"""Constructs a VoVNet-39 model as described in
    `"An Energy and GPU-Computation Efficient Backbone Networks"
    <https://arxiv.org/abs/1904.09730>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vovnet('vovnet39', [128, 160, 192, 224], [256, 512, 768, 1024],
                    [1,1,2,2], 5, pretrained, progress, **kwargs)


def vovnet27_slim(pretrained=False, progress=True, **kwargs):
    r"""Constructs a VoVNet-39 model as described in
    `"An Energy and GPU-Computation Efficient Backbone Networks"
    <https://arxiv.org/abs/1904.09730>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vovnet('vovnet27_slim', [64, 80, 96, 112], [128, 256, 384, 512],
                    [1,1,1,1], 5, pretrained, progress, **kwargs)

arch_setting = {
    57: ('vovnet57', [128, 160, 192, 224], [256, 512, 768, 1024],
                    [1,1,4,3], 5, True, True),
    39: ('vovnet39', [128, 160, 192, 224], [256, 512, 768, 1024],
                    [1,1,2,2], 5, True, True),
    27: ('vovnet27_slim', [64, 80, 96, 112], [128, 256, 384, 512],
                    [1,1,1,1], 5, False, True)
}

def vov_net(num_layers, heads, head_conv):
    arch, config_stage_ch, config_concat_ch, block_per_stage, \
    layer_per_block, pretrained, progress = arch_setting[57]
    print('{}'.format(heads))
    cfg = {
        'heads': heads,
        'head_conv': 256,
    }
    return _vovnet(arch, config_stage_ch, config_concat_ch, block_per_stage,
    layer_per_block, pretrained, progress, **cfg)