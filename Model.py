import torch
import torch.nn as nn

#--------------------
#basic Block 
#-----------------------
def autopad (k, p=None , d=1):#(k=kernel , p=padding , d= dilation)
    if d > 1: 
        #actual Kernel size 
        k= d* (k-1) + 1 if isinstance(k, int) else [d* (x-1)+1 for x in k ]
    if p is None: 
        #auto-pad 
        p= k // 2 if isinstance(k,int) else [x// 2 for x in k]
    return p

#activasi fungsi dengan Yolov11 

class SiLU(nn.Module):#sigmoid Linier Unit
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)
    
# code base ConV Block 
class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, activation, k=1, s=1, p=None, g=1):
        # in_ch = jumlah chanel input
        # out_ch = jumlah chanel output 
        # activation =  function dari activation function (SiLU atau Identity)
        # k = kernel size
        # s = stride
        # p = padding
        # g = groups
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, autopad(k, p, d=1), groups=g, bias=False)
        self.norm = torch.nn.BatchNorm2d(out_ch, eps=0.001, momentum=0.03)
        self.relu = activation
        # momentum = keceptan update moving average saat trining 
    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))
        
    def fuse_forward(self, x):
        return self.relu(self.conv(x))


#Bottleneck Block 
class Residual(torch.nn.Module):
    def __init__(self, ch, e =0.5):
        super().__init__()
        self.conv1 = Conv(ch, int(ch * e), torch.nn.SiLU(), k=3, p=1)
        self.conv2 = Conv(int(ch * e),ch, torch.nn.SiLU(), k=3, p=1)
    def forward(self ,x ):
        return x + self.conv2(self.conv1(x))

#----------------------------------
#            BackBone 
#----------------------------------

#Modul C3K  Cross Stage Partial Bottleneck
class C3K( torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        #turunkan channel → setengahnya
        self.conv1 = Conv(in_ch, out_ch // 2, activation=torch.nn.SiLU())
        self.conv2 = Conv(in_ch, out_ch // 2,activation=torch.nn.SiLU())
        self.conv3 = Conv(2*(out_ch // 2),out_ch, activation=torch.nn.SiLU())
        self.res_m = torch.nn.Sequential(Residual(out_ch // 2, e = 1.0),
                                         Residual(out_ch // 2, e =1.0))
    def forward(self,x):
        y= self.res_m(self.conv1(x))

        return self.conv3(torch.cat((y, self.conv2(x)),dim=1))

#Modul Blok C3K2 
class C3K2 (torch.nn.Module):
    def __init__(self, in_ch, out_ch, n, csp, r):
        super().__init__()
        self.conv1 = Conv(in_ch, 2 * (out_ch // r), activation=torch.nn.SiLU())
        self.conv2 = Conv((2+n)* (out_ch // r), out_ch, activation=torch.nn.SiLU())

        if not csp:
            self.res_m = torch.nn.ModuleList(Residual(out_ch // r)for _ in range(n))
        else : 
            self.res_m = torch.nn.ModuleList(C3K(out_ch // r, out_ch // r)for _ in range(n))
    
    def forward(self, x):
        y = list(self.conv1(x).chunk(2,1))
        y.extend(m(y[-1]) for m in self.res_m)
        return self.conv2(torch.cat(y, dim=1))

#Spatial Pyramid Pooling and Fusion (SPPF) Layer
class SPPF (nn.Module):
    def __init__(self, c1, c2, k=5 ):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, activation=torch.nn.SiLU(), k=1, s=1)
        self.cv2 = Conv(c_ * 4, c2, activation=torch.nn.SiLU(), k=1, s=1)
        self.m = nn.MaxPool2d(kernel_size = k, stride= 1, padding = k // 2)
    
    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x,y1,y2, self.m(y2)),1))
    

#----------------------------------
#            NECK 
#----------------------------------

# Pengganti PSABlock tanpa Attention (pakai Residual)
class PSABlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = Residual(ch)

    def forward(self, x):
        return self.block(x)

# Pengganti PSA yang ringan
class PSA(nn.Module):
    def __init__(self, ch, n):
        super().__init__()
        self.conv1 = Conv(ch, 2 * (ch // 2), SiLU())
        self.conv2 = Conv(2 * (ch // 2), ch, SiLU())
        self.res_m = nn.Sequential(*[PSABlock(ch // 2) for _ in range(n)])

    def forward(self, x):
        x, y = self.conv1(x).chunk(2, 1)
        y = self.res_m(y)
        return self.conv2(torch.cat((x, y), dim=1))

#----------------------------------
#            Head 

#----------------------------------

class DFL(nn.Module):
    # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    def __init__(self, c1=16):
        super().__init__()
        self.conv   = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x           = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1     = c1

    def forward(self, x):
        b, c, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
    
def make_anchors(x, strides, offset=0.5):
    assert x is not None
    anchor_tensor, stride_tensor = [], []
    dtype, device = x[0].dtype, x[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = x[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + offset  # shift y
        sy, sx = torch.meshgrid(sy, sx)
        anchor_tensor.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_tensor), torch.cat(stride_tensor)

def fuse_conv(conv, norm):
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 kernel_size=conv.kernel_size,
                                 stride=conv.stride,
                                 padding=conv.padding,
                                 groups=conv.groups,
                                 bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv


class Head(torch.nn.Module):
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, filters=()):
        super().__init__()
        self.ch = 16  # DFL channels
        self.nc = nc  # number of classes
        self.nl = len(filters)  # number of detection layers
        self.no = nc + self.ch * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build

        box = max(64, filters[0] // 4)
        cls = max(80, filters[0], self.nc)

        self.dfl = DFL(self.ch)
        
        self.box = torch.nn.ModuleList(
           torch.nn.Sequential(Conv(x, box,activation=torch.nn.SiLU(), k=3, p=1),
           Conv(box, box,activation=torch.nn.SiLU(), k=3, p=1),
           torch.nn.Conv2d(box, out_channels=4 * self.ch,kernel_size=1)) for x in filters)
        
        self.cls = torch.nn.ModuleList(
            torch.nn.Sequential(Conv(x, x, torch.nn.SiLU(), k=3, p=1, g=x),
            Conv(x, cls, torch.nn.SiLU()),
            Conv(cls, cls, torch.nn.SiLU(), k=3, p=1, g=cls),
            Conv(cls, cls, torch.nn.SiLU()),
            torch.nn.Conv2d(cls, out_channels=self.nc,kernel_size=1)) for x in filters)

    def forward(self, x):
        for i, (box, cls) in enumerate(zip(self.box, self.cls)):
            x[i] = torch.cat(tensors=(box(x[i]), cls(x[i])), dim=1)
        if self.training:
            return x

        self.anchors, self.strides = (i.transpose(0, 1) for i in make_anchors(x, self.stride))
        x = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], dim=2)
        box, cls = x.split(split_size=(4 * self.ch, self.nc), dim=1)

        a, b = self.dfl(box).chunk(2, 1)
        a = self.anchors.unsqueeze(0) - a
        b = self.anchors.unsqueeze(0) + b
        box = torch.cat(tensors=((a + b) / 2, b - a), dim=1)

        return torch.cat(tensors=(box * self.strides, cls.sigmoid()), dim=1)

class YOLOv11(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        self.backbone = nn.Sequential(
            Conv(3, 8, SiLU(), k=3, p=1),
            C3K2(8, 16, n=1, csp=False, r=2),
            C3K2(16, 32, n=2, csp=True, r=2),
            SPPF(32, 64)
        )
        self.neck = PSA(64, n=1)
        self.head = Head(nc=num_classes, filters=[64])

    def forward(self, x):
        x = [self.backbone(x)]
        x = [self.neck(x[0])]
        return self.head(x)

if __name__ == '__main__':
    model = YOLOv11(num_classes=1)
    model.eval()  # ⬅️ Tambahkan ini
    x = torch.randn(1, 3, 128, 128)
    out = model(x)
    print(out.shape)
