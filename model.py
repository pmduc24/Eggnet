import torch
import torch.nn as nn
from torch.nn import Softmax

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization."""
        return self.act(self.conv(x))
    
def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).repeat(H),0).unsqueeze(0).repeat(B*W,1,1).to('cpu')

class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim):
        super(CrissCrossAttention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
  
        return self.gamma*(out_H + out_W) + x
    
class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 7, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.a1 = nn.AvgPool2d(kernel_size=13, stride=1, padding=6)
        self.a2 = nn.AvgPool2d(kernel_size=9, stride=1, padding=4)
        self.a3 = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        self.crisscrossattn = CrissCrossAttention(c_ * 7)  # Match concatenated channel size

    def forward(self, x):
        x = self.cv1(x)  # [B, c_, H, W]
        y1 = self.m(x)   
        y2 = self.m(y1)  
        a1 = self.a1(x)
        a2 = self.a2(x)
        a3 = self.a3(x)
        
        concat = torch.cat((x, y1, y2, self.m(y2), a1, a2, a3), 1)  # [B, c_*7, H, W]
        b, c, h, w = concat.shape
        concat = concat.permute(0, 2, 3, 1)  # [B, H, W, c_*7]
        attended = self.crisscrossattn(concat.permute(0, 3, 1, 2))  # Back to [B, c_*7, H, W]
        return self.cv2(attended)  # [B, c2, H, W
    
class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    
class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )

class C3k2CC(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else CCBottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )

class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    
def yolo_params(version):
    #  [depth, width, max_channels]
    if version=='n':
        return 0.50, 0.25, 1024
    elif version=='s':
        return 0.50, 0.50, 1024
    elif version=='m':
        return 0.50, 1.00, 512
    elif version=='l':
        return 1.00, 1.00, 512
    elif version=='x':
        return 1.00, 1.50, 512
    
class CCBottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, recurrence = 2):
        super().__init__()
        self.recurrence = recurrence
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
 
        self.in_channels = c1
        self.channels = c1 // 8
        self.ConvQuery = nn.Conv2d(self.in_channels, self.channels, kernel_size=1)
        self.ConvKey = nn.Conv2d(self.in_channels, self.channels, kernel_size=1)
        self.ConvValue = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1)
 
        self.SoftMax = nn.Softmax(dim=3)
        self.gamma = nn.Parameter(torch.zeros(1))
 
    def forward(self, x):
        x0 = self.cv2(self.cv1(x))
        x1 = x0
        # print('x1 is:',x1)
 
        for i in range(self.recurrence):
            b, _, h, w = x1.size()
 
            # [b, c', h, w]
            query = self.ConvQuery(x1)
            # [b, w, c', h] -> [b*w, c', h] -> [b*w, h, c']
            query_H = query.permute(0, 3, 1, 2).contiguous().view(b*w, -1, h).permute(0, 2, 1)
            # [b, h, c', w] -> [b*h, c', w] -> [b*h, w, c']
            query_W = query.permute(0, 2, 1, 3).contiguous().view(b*h, -1, w).permute(0, 2, 1)
            
            # [b, c', h, w]
            key = self.ConvKey(x1)
            # [b, w, c', h] -> [b*w, c', h]
            key_H = key.permute(0, 3, 1, 2).contiguous().view(b*w, -1, h)
            # [b, h, c', w] -> [b*h, c', w]
            key_W = key.permute(0, 2, 1, 3).contiguous().view(b*h, -1, w)
            
            # [b, c, h, w]
            value = self.ConvValue(x1)
            # [b, w, c, h] -> [b*w, c, h]
            value_H = value.permute(0, 3, 1, 2).contiguous().view(b*w, -1, h).float()
            # [b, h, c, w] -> [b*h, c, w]
            value_W = value.permute(0, 2, 1, 3).contiguous().view(b*h, -1, w).float()
 
            if query_H.is_cuda:
                inf = -1 * torch.diag(torch.tensor(float("inf")).cuda().repeat(h),0).unsqueeze(0).repeat(b*w,1,1)
            else:
                inf = -1 * torch.diag(torch.tensor(float("inf")).repeat(h),0).unsqueeze(0).repeat(b*w,1,1)
            # print('inf is ', inf)
            # print(query_H.is_cuda, inf.is_cuda)
 
            # [b*w, h, c']* [b*w, c', h] -> [b*w, h, h] -> [b, h, w, h]
            energy_H = (torch.bmm(query_H, key_H)  + inf).view(b, w, h, h).permute(0, 2, 1, 3)
            # energy_H = torch.bmm(query_H, key_H).view(b, w, h, h).permute(0, 2, 1, 3)
            # [b*h, w, c']*[b*h, c', w] -> [b*h, w, w] -> [b, h, w, w]
            energy_W = torch.bmm(query_W, key_W).view(b, h, w, w)
            # [b, h, w, h+w]  concate channels in axis=3
            energy_total = torch.cat([energy_H, energy_W], 3)
            # print('energy_total is ', energy_total)
            concate = self.SoftMax(energy_total)
            # print('concate is ', concate)
            # [b, h, w, h] -> [b, w, h, h] -> [b*w, h, h]
            attention_H = concate[:,:,:, 0:h].permute(0, 2, 1, 3).contiguous().view(b*w, h, h)
            attention_W = concate[:,:,:, h:h+w].contiguous().view(b*h, w, w)
    
            # [b*w, h, c]*[b*w, h, h] -> [b, w, c, h]
            out_H = torch.bmm(value_H, attention_H.permute(0, 2, 1)).view(b, w, -1, h).permute(0, 2, 3, 1)
            out_W = torch.bmm(value_W, attention_W.permute(0, 2, 1)).view(b, h, -1, w).permute(0, 2, 1, 3)
 
            x1 = self.gamma*(out_H + out_W) + x1
            # print('In cc x1 is:', x1)
 
        # out = self.conv_out(x1)
        out = x1.expand_as(x0)
 
        # print('out x1 is:', x1)
        
        return x + out if self.add else out

class Focus(nn.Module):
    """Focus wh information into c-space."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Applies convolution to concatenated tensor and returns the output.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))

class CARAFE(nn.Module):
    def __init__(self, c, k_enc=3, k_up=5, c_mid=64, scale=2):
        """ The unofficial implementation of the CARAFE module.
        The details are in "https://arxiv.org/abs/1905.02188".
        Args:
            c: The channel number of the input and the output.
            c_mid: The channel number after compression.
            scale: The expected upsample scale.
            k_up: The size of the reassembly kernel.
            k_enc: The kernel size of the encoder.
        Returns:
            X: The upsampled feature map.
        """
        super(CARAFE, self).__init__()
        self.scale = scale

        self.comp = Conv(c, c_mid)
        self.enc = Conv(c_mid, (scale*k_up)**2, k=k_enc, act=False)
        self.pix_shf = nn.PixelShuffle(scale)

        self.upsmp = nn.Upsample(scale_factor=scale, mode='nearest')
        self.unfold = nn.Unfold(kernel_size=k_up, dilation=scale, 
                                padding=k_up//2*scale)

    def forward(self, X):
        b, c, h, w = X.size()
        h_, w_ = h * self.scale, w * self.scale
        
        W = self.comp(X)                                # b * m * h * w
        W = self.enc(W)                                 # b * 100 * h * w
        W = self.pix_shf(W)                             # b * 25 * h_ * w_
        W = torch.softmax(W, dim=1)                         # b * 25 * h_ * w_

        X = self.upsmp(X)                               # b * c * h_ * w_
        X = self.unfold(X)                              # b * 25c * h_ * w_
        X = X.view(b, c, -1, h_, w_)                    # b * 25 * c * h_ * w_

        X = torch.einsum('bkhw,bckhw->bchw', [W, X])    # b * c * h_ * w_
        return X
    
class Yolov11Backbone(nn.Module):
    def __init__(self, in_channels=3, version='n'):
        super(Yolov11Backbone, self).__init__()
        d, w, mc = yolo_params(version)
        self.conv1 = Conv(in_channels, int(min(64, mc) * w), 3, 2, 1)
        self.conv2 = Conv(int(min(64, mc) * w), int(min(128, mc) * w), 3, 2, 1)
        self.c3k2_3 = C3k2CC(int(min(128, mc) * w), int(min(128, mc) * w), n = int(3 * d), shortcut=False)
        self.conv4 = Conv(int(min(128, mc) * w), int(min(256, mc) * w), 3, 2, 1)
        self.focus4 = Focus(int(min(256, mc) * w), int(min(256, mc) * w))
        self.c3k2_5 = C3k2CC(int(min(256, mc) * w), int(min(256, mc) * w), n= int(6 * d), shortcut=False)
        self.conv6 = Conv(int(min(256, mc) * w), int(min(512, mc) * w), 3, 2, 1)
        self.focus6 = Focus(int(min(512, mc) * w), int(min(512, mc) * w))
        self.c3k2_7 = C3k2CC(int(min(512, mc) * w), int(min(512, mc) * w), n= int(6 * d), shortcut=True)
        self.conv8 = Conv(int(min(512, mc) * w), int(min(1024, mc) * w), 3, 2, 1)
        self.focus8 = Focus(int(min(1024, mc) * w), int(min(1024, mc) * w))
        self.c3k2_9 = C3k2CC(int(min(1024, mc) * w), int(min(1024, mc) * w), n= int(3 * d), shortcut=True)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        out1 = self.c3k2_3(x)
        x = self.conv4(out1)
        x = self.focus4(x)
        out2 = self.c3k2_5(x)
        x = self.conv6(out2)
        x = self.focus6(x)
        out3 = self.c3k2_7(x)
        x = self.conv8(out3)
        x = self.focus8(x)
        out4 = self.c3k2_9(x)

        return out1,out2,out3, out4


import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_rate=16):
        super(ChannelAttention, self).__init__()
        self.squeeze = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1),
            nn.AdaptiveMaxPool2d(1)
        ])
        self.excitation = nn.Sequential(
            nn.Conv2d(in_channels=channels,
                      out_channels=channels // reduction_rate,
                      kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels // reduction_rate,
                      out_channels=channels,
                      kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # perform squeeze with independent Pooling
        avg_feat = self.squeeze[0](x)
        max_feat = self.squeeze[1](x)
        # perform excitation with the same excitation sub-net
        avg_out = self.excitation(avg_feat)
        max_out = self.excitation(max_feat)
        # attention
        attention = self.sigmoid(avg_out + max_out)
        return attention * x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # mean on spatial dim
        avg_feat    = torch.mean(x, dim=1, keepdim=True)
        # max on spatial dim
        max_feat, _ = torch.max(x, dim=1, keepdim=True)
        feat = torch.cat([avg_feat, max_feat], dim=1)
        out_feat = self.conv(feat)
        attention = self.sigmoid(out_feat)
        return attention * x

class UpsampleConnection4x(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleConnection4x, self).__init__()
        
        self.conv1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.dilated_conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=6, padding=6)
        self.upsample1 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)

        self.relu = nn.ReLU()
        self.conv1x1_2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.dilated_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=12, padding=12)
        self.upsample2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.conv1x1_1(x)
        x = self.dilated_conv1(x)
        x = self.upsample1(x)

        x = self.relu(x)
        x = self.conv1x1_2(x)
        x = self.dilated_conv2(x)
        x = self.upsample2(x)
        
        return x

class UpsampleConnection2x(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleConnection2x, self).__init__()

        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.dilated_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=6, padding=6)
        self.upsample = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1x1(x)
        x = self.dilated_conv(x)
        x = self.upsample(x)
        x = self.relu(x)
        return x

class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))

class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x

class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x

class PSA(nn.Module):
    """
    PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1, c2, e=0.5):
        """Initializes the PSA module with input/output channels and attention mechanism for feature extraction."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x):
        """Executes forward pass in PSA module, applying attention and feed-forward layers to the input tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))

class DownsampleConnection(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsampleConnection, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.conv2(x)

        return x

class Decoder(nn.Module):
    def __init__(self, version='n'):
        super(Decoder, self).__init__()
        d, w, mc = yolo_params(version)

        self.down_scale1 = DownsampleConnection(int(min(128, mc) * w),int(min(256, mc) * w))
        self.down_scale2 = DownsampleConnection(int(min(256, mc) * w),int(min(512, mc) * w))

        self.down_ca = DownsampleConnection(int(min(512, mc) * w),int(min(256, mc) * w))
        self.down_sa = DownsampleConnection(int(min(1024, mc) * w),int(min(512, mc) * w))

        self.ca = ChannelAttention(int(min(256, mc) * w))
        self.sa = SpatialAttention()

        self.up_scalefm1 = UpsampleConnection4x(int(min(1024, mc) * w) + int(min(512, mc) * w), int(min(256, mc) * w))
        self.up_scalefm2 = UpsampleConnection4x(int(min(1024, mc) * w) + int(min(256, mc) * w), int(min(512, mc) * w))
        self.up_scalefm3 = UpsampleConnection2x(int(min(512, mc) * w) + int(min(256, mc) * w), int(min(512, mc) * w))
        self.sppf = SPPF(int(min(1024, mc) * w), int(min(1024, mc) * w))
        self.c2psa = C2PSA(int(min(1024, mc) * w), int(min(1024, mc) * w), n=3, e=0.5)

    def forward(self, out1, out2, out3, out4):
        down_scale_out1 = self.down_scale1(out1)    
        out_ca = torch.concat([down_scale_out1, out2], dim=1)
        
        down_scale_out2 = self.down_scale2(out2)
        out_sa = torch.concat([down_scale_out2, out3],dim=1)
    
        down_scale_ca = self.down_ca(out_ca)
        down_scale_sa = self.down_sa(out_sa)

        sppf_out = self.sppf(out4)
        out4 = self.c2psa(out4)

        fm1 = torch.concat([sppf_out, down_scale_sa],dim=1)
        fm1_up_scale = self.up_scalefm1(fm1)

        fm2 = torch.concat([down_scale_ca, out_sa],dim=1)
        fm2_up_scale = self.up_scalefm2(fm2)

    
        fm3 = torch.concat([out_ca, fm1_up_scale],dim=1)
        fm3_up_scale = self.up_scalefm3(fm3)

        fm4 = torch.concat([out1, fm3_up_scale, fm2_up_scale], dim=1)
        print(fm4.shape)
        return fm4

class Segnet(nn.Module):
    def __init__(self, num_classes, version='n'):
        super(Segnet, self).__init__()
        d, w, mc = yolo_params(version)
        self.encoder = Yolov11Backbone(version=version)
        self.decoder = Decoder(version=version)
        self.segmetation = UpsampleConnection4x(int(min(128, mc) * w) + int(min(512, mc) * w) + int(min(512, mc) * w), num_classes)

    def forward(self, x):
        out1,out2,out3,out4 = self.encoder(x)
        out = self.decoder(out1,out2,out3,out4)
        out = self.segmetation(out)

        return x

dummy_input = torch.randn(1, 3, 224, 224)  # Example shape, adjust as needed

# Initialize the Segnet model
model = Segnet(num_classes=21, version='n')  # Adjust num_classes and version as needed

# Pass the dummy input through the model
output = model(dummy_input)

# Print the output shape
print(output.shape)