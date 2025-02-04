
# Tải MobileNetV3 Small hoặc Large
backbone = models.mobilenet_v3_small(pretrained=True) 

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
    
class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))

class SEModule(nn.Module):
    def __init__(self, channels: int, ratio: int = 8) -> None:
        super(SEModule, self).__init__()

        # Average Pooling for Squeeze
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # Excitation Operation
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels // ratio, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Squeeze & Excite Forward Pass
        b, c, _, _ = x.size()

        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        return x * y


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilations: list[int] = [6, 12, 18, 24]) -> None:
        super(ASPP, self).__init__()

        # Atrous Convolutions
        self.atrous_convs = nn.ModuleList()
        for d in dilations:
            at_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, dilation=d, padding="same", bias=False
            )
            self.atrous_convs.append(at_conv)

        self.batch_norm = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()
        
        self.squeeze_excite = SEModule(channels=out_channels)

        self.dropout = nn.Dropout(p=0.5)

        # Upsampling by Bilinear Interpolation
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=16)

        # Global Average Pooling
        self.avgpool = nn.AvgPool2d(kernel_size=(16, 16))

        # 1x1 Convolution
        self.conv1x1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, padding="same", bias=False
        )

        # Final 1x1 Convolution
        self.final_conv = nn.Conv2d(
            in_channels=out_channels * (len(dilations) + 2),
            out_channels=out_channels,
            kernel_size=1,
            padding="same",
            bias=False,
        )

    def forward(self, x):
        # ASPP Forward Pass

        # 1x1 Convolution
        x1 = self.conv1x1(x)
        x1 = self.batch_norm(x1)
        x1 = self.dropout(x1)
        x1 = self.relu(x1)
        x1 = self.squeeze_excite(x1)

        # Atrous Convolutions
        atrous_outputs = []
        for at_conv in self.atrous_convs:
            at_output = at_conv(x)
            at_output = self.batch_norm(at_output)
            at_output = self.relu(at_output)
            at_output = self.squeeze_excite(at_output)
            atrous_outputs.append(at_output)

        # Global Average Pooling and 1x1 Convolution for global context
        avg_pool = self.avgpool(x)
        avg_pool = self.conv1x1(avg_pool)
        avg_pool = self.batch_norm(avg_pool)
        avg_pool = self.relu(avg_pool)
        avg_pool = self.upsample(avg_pool)
        avg_pool = self.squeeze_excite(avg_pool)

        # Concatenating Dilated Convolutions and Global Average Pooling
        combined_output = torch.cat((x1, *atrous_outputs, avg_pool), dim=1)

        # Final 1x1 Convolution for ASPP Output
        aspp_output = self.final_conv(combined_output)
        aspp_output = self.batch_norm(aspp_output)
        aspp_output = self.relu(aspp_output)
        aspp_output = self.squeeze_excite(aspp_output)

        return aspp_output

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

class Upsample(nn.Module):
    def __init__(self, scale_factor=2, mode='nearest'):
        super(Upsample, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)
    
    def forward(self, x):
        x = self.upsample(x)
        return x
    