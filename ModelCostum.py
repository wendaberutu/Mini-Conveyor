import torch
import torch.nn as nn

# Fungsi padding otomatis
def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

# SiLU (Swish)
class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

# Conv Block (Conv + BN + Activation)
class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, activation, k=3, s=1, p=None, g=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, autopad(k, p), groups=g, bias=False)
        self.norm = nn.BatchNorm2d(out_ch)
        self.relu = activation
    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

# Bottleneck Residual Block
class Residual(nn.Module):
    def __init__(self, ch, e=0.5):
        super().__init__()
        hidden_ch = int(ch * e)
        self.conv1 = Conv(ch, hidden_ch, SiLU(), k=3)
        self.conv2 = Conv(hidden_ch, ch, SiLU(), k=3)
    def forward(self, x):
        return x + self.conv2(self.conv1(x))

# C3K2 (1 Residual block)
class C3K2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = Conv(in_ch, out_ch, SiLU())
        self.res = Residual(out_ch, e=0.5)
    def forward(self, x):
        return self.res(self.conv1(x))

# SPPF (tetap dipertahankan)
class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, SiLU(), k=1)
        self.cv2 = Conv(c_ * 4, c2, SiLU(), k=1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat((x, y1, y2, y3), dim=1))

# Head sangat sederhana (output: 4 box + kelas)
class HeadLite(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, num_classes + 4, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

# Model utama YOLOv11Lite
class YOLOv11Lite(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.backbone = nn.Sequential(
            Conv(3, 16, SiLU(), k=3),
            C3K2(16, 32),
            C3K2(32, 64),
            SPPF(64, 128)
        )
        self.head = HeadLite(128, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)

# Test jalankan model
if __name__ == "__main__":
    model = YOLOv11Lite(num_classes=1)
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    print("Output shape:", out.shape)  # [1, num_classes + 4, H, W]
