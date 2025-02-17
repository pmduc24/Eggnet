import torch
import torch.nn as nn
from torch.nn import Softmax
from snake import DySnakeConv
from common import *

class Yolov11Backbone(nn.Module):
    def __init__(self, in_channels=3, version='n'):
        super(Yolov11Backbone, self).__init__()
        d, w, mc = yolo_params(version)
        self.conv1 = Conv(in_channels, int(min(64, mc) * w), 3, 2, 1)
        self.conv2 = Conv(int(min(64, mc) * w), int(min(128, mc) * w), 3, 2, 1)
        self.c3k2_3 = C3k2CC(int(min(128, mc) * w), int(min(128, mc) * w), n = int(3 * d), shortcut=False)
        self.conv4 = DySnakeConv(int(min(128, mc) * w), int(min(256, mc) * w), 3)
        self.focus4 = Focus(int(min(256, mc) * w), int(min(256, mc) * w))
        self.c3k2_5 = C3k2CC(int(min(256, mc) * w), int(min(256, mc) * w), n= int(6 * d), shortcut=False)
        self.conv6 = DySnakeConv(int(min(256, mc) * w), int(min(512, mc) * w), 3)
        self.focus6 = Focus(int(min(512, mc) * w), int(min(512, mc) * w))
        self.c3k2_7 = C3k2CC(int(min(512, mc) * w), int(min(512, mc) * w), n= int(6 * d), shortcut=True)
        self.focus8 = Focus(int(min(1024, mc) * w), int(min(1024, mc) * w))
        self.conv8 = DySnakeConv(int(min(512, mc) * w), int(min(1024, mc) * w), 3)
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

        self.up_scalefm1 = CCUpsampleConnection4x(int(min(1024, mc) * w) + int(min(512, mc) * w), int(min(256, mc) * w))
        self.up_scalefm2 = CCUpsampleConnection4x(int(min(1024, mc) * w) + int(min(256, mc) * w), int(min(512, mc) * w))
        self.up_scalefm3 = CCUpsampleConnection2x(int(min(512, mc) * w) + int(min(256, mc) * w), int(min(512, mc) * w))
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
        self.segmetation = CCUpsampleConnection4x(int(min(128, mc) * w) + int(min(512, mc) * w) + int(min(512, mc) * w), num_classes)

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
print("wwo", output.shape)