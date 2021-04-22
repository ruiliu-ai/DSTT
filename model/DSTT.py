''' Spatial-Temporal Transformer Networks
'''
import numpy as np
import time
import math
from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from core.spectral_norm import spectral_norm as _spectral_norm

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).' % (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)

class HierarchyEncoder(nn.Module):
    def __init__(self, channel):
        super(HierarchyEncoder, self).__init__()
        assert channel == 256
        self.group = [1, 2, 4, 8, 1]
        self.layers = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, groups=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, groups=4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, groups=8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1, groups=1),
            nn.LeakyReLU(0.2, inplace=True)
        ])

    def forward(self, x):
        bt, c, h, w = x.size()
        out = x
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and i != 0:
                g = self.group[i//2]
                x0 = x.view(bt, g, -1, h, w)
                out0 = out.view(bt, g, -1, h, w)
                out = torch.cat([x0, out0], 2).view(bt, -1, h, w)
            out = layer(out)
        return out


class InpaintGenerator(BaseNetwork):
    def __init__(self, init_weights=True):
        super(InpaintGenerator, self).__init__()
        channel = 256
        hidden = 512
        stack_num = 8
        num_head = 4
        kernel_size = (7, 7)
        padding = (3, 3)
        stride = (3, 3)
        output_size = (60, 108)
        token_size = tuple(map(lambda x,y:x//y, output_size, stride))
        blocks = []
        dropout = 0.

        for _ in range(stack_num//2):
            blocks.append(TransformerBlock(token_size, hidden=hidden, num_head=num_head, mode='t', dropout=dropout))
            blocks.append(TransformerBlock(token_size, hidden=hidden, num_head=num_head, mode='s', dropout=dropout))
        self.transformer = nn.Sequential(*blocks)
        self.patch2vec = nn.Conv2d(channel//2, hidden, kernel_size=kernel_size, stride=stride, padding=padding)
        self.vec2patch = Vec2Patch(channel//2, hidden, output_size, kernel_size, stride, padding)

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, channel, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.hier_enc = HierarchyEncoder(channel)
        self.decoder = nn.Sequential(
            deconv(channel // 2, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            deconv(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        )

        if init_weights:
            self.init_weights()

    def forward(self, masked_frames):
        b, t, c, h, w = masked_frames.size()
        enc_feat = self.encoder(masked_frames.view(b*t, c, h, w))
        enc_feat = self.hier_enc(enc_feat)

        trans_feat = self.patch2vec(enc_feat)
        _, c, h, w = trans_feat.size()
        trans_feat = trans_feat.view(b*t, c, -1).permute(0, 2, 1)
        trans_feat = self.transformer({'x': trans_feat, 't': t})['x']
        trans_feat = self.vec2patch(trans_feat)
        enc_feat = enc_feat + trans_feat
        
        output = self.decoder(enc_feat)
        output = torch.tanh(output)
        return output


class deconv(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, padding=0, scale_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(input_channel, output_channel,
                              kernel_size=kernel_size, stride=1, padding=padding)
        self.s = scale_factor

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.s, mode='bilinear',
                          align_corners=True)
        return self.conv(x)


# #############################################################################
# ############################# Transformer  ##################################
# #############################################################################


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """
    def __init__(self, p=0.1):
        super(Attention, self).__init__()
        self.dropout = nn.Dropout(p=p)

    def forward(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1)
                              ) / math.sqrt(query.size(-1))
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        p_val = torch.matmul(p_attn, value)
        return p_val, p_attn


class Vec2Patch(nn.Module):
    def __init__(self, channel, hidden, output_size, kernel_size, stride, padding):
        super(Vec2Patch, self).__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        c_out = reduce((lambda x, y: x * y), kernel_size) * channel
        self.embedding = nn.Linear(hidden, c_out)
        self.to_patch = torch.nn.Fold(output_size=output_size, kernel_size=kernel_size, stride=stride, padding=padding)
        h, w = output_size

    def forward(self, x):
        feat = self.embedding(x)
        b, n, c = feat.size()
        feat = feat.permute(0, 2, 1)
        feat = self.to_patch(feat)
        return feat


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, tokensize, d_model, head, mode, p=0.1):
        super().__init__()
        self.mode = mode
        self.query_embedding = nn.Linear(d_model, d_model)
        self.value_embedding = nn.Linear(d_model, d_model)
        self.key_embedding = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention(p=p)
        self.head = head
        self.h, self.w = tokensize

    def forward(self, x, t):
        bt, n, c = x.size() 
        b = bt // t
        c_h = c // self.head
        key = self.key_embedding(x)
        query = self.query_embedding(x)
        value = self.value_embedding(x)
        if self.mode == 's':
            key = key.view(b, t, n, self.head, c_h).permute(0, 1, 3, 2, 4)
            query = query.view(b, t, n, self.head, c_h).permute(0, 1, 3, 2, 4)
            value = value.view(b, t, n, self.head, c_h).permute(0, 1, 3, 2, 4)
            att, _ = self.attention(query, key, value)
            att = att.permute(0, 1, 3, 2, 4).contiguous().view(bt, n, c)
        elif self.mode == 't':
            key = key.view(b, t, 2, self.h//2, 2, self.w//2, self.head, c_h)
            key = key.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous().view(b, 4, self.head, -1, c_h)
            query = query.view(b, t, 2, self.h//2, 2, self.w//2, self.head, c_h)
            query = query.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous().view(b, 4, self.head, -1, c_h)
            value = value.view(b, t, 2, self.h//2, 2, self.w//2, self.head, c_h)
            value = value.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous().view(b, 4, self.head, -1, c_h)
            att, _ = self.attention(query, key, value)
            att = att.view(b, 2, 2, self.head, t, self.h//2, self.w//2, c_h)
            att = att.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous().view(bt, n, c)
        output = self.output_linear(att)
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, p=0.1):
        super(FeedForward, self).__init__()
        # We set d_ff as a default to 2048
        self.conv = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(p=p))

    def forward(self, x):
        x = self.conv(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, tokensize, hidden=128, num_head=4, mode='s', dropout=0.1):
        super().__init__()
        self.attention = MultiHeadedAttention(tokensize, d_model=hidden, head=num_head, mode=mode, p=dropout)
        self.ffn = FeedForward(hidden, p=dropout)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, input):
        x, t = input['x'], input['t']
        x = self.norm1(x)
        x = x + self.dropout(self.attention(x, t))
        y = self.norm2(x)
        x = x + self.ffn(y)
        return {'x': x, 't': t}

# ######################################################################
# ######################################################################

class Discriminator(BaseNetwork):
    def __init__(self, in_channels=3, use_sigmoid=False, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        nf = 32

        self.conv = nn.Sequential(
            spectral_norm(nn.Conv3d(in_channels=in_channels, out_channels=nf*1, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                                    padding=1, bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(64, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(nf*1, nf*2, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                                    padding=(1, 2, 2), bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(128, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(nf * 2, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                                    padding=(1, 2, 2), bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(nf * 4, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                                    padding=(1, 2, 2), bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(nf * 4, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                                    padding=(1, 2, 2), bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(nf * 4, nf * 4, kernel_size=(3, 5, 5),
                      stride=(1, 2, 2), padding=(1, 2, 2))
        )

        if init_weights:
            self.init_weights()

    def forward(self, xs):
        # T, C, H, W = xs.shape
        xs_t = torch.transpose(xs, 0, 1)
        xs_t = xs_t.unsqueeze(0)  # B, C, T, H, W
        feat = self.conv(xs_t)
        if self.use_sigmoid:
            feat = torch.sigmoid(feat)
        out = torch.transpose(feat, 1, 2)  # B, T, C, H, W
        return out


def spectral_norm(module, mode=True):
    if mode:
        return _spectral_norm(module)
    return module
