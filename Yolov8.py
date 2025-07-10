import torch
import torch.nn as nn
import torchvision

# --- AUTO PADDING ---
def autopad(k, p=None, d=1):
    #"""Menghitung padding otomatis berdasarkan kernel dan dilation."""
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p

# --- AKTIVASI SiLU ---
class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

# --- CONV BLOCK: Conv + BatchNorm + SiLU ---
class Conv(nn.Module):
    #k= kernel size 
    #s= stride
    def __init__(self, in_ch, out_ch, k=1, s=1, p=None, g=1 , d=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, autopad(k, p, d), groups=g, bias=False)
        self.norm = nn.BatchNorm2d(out_ch, eps=0.001, momentum=0.03)
        self.act = SiLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

    def fuse_forward(self, x): 
       # """Digunakan saat inference setelah fusing Conv+BN."""
        return self.act(self.conv(x))
#--------------------
#--- BOTTLENECK -----
#--------------------

class Bottleneck(nn.Module):
    def __init__(self, in_ch, out_ch, shortcut=True, e=0.5):
        super().__init__()
        hidden_ch = int(out_ch * e)
        self.conv1 = Conv(in_ch, hidden_ch, k=1)
        self.conv2 = Conv(hidden_ch, out_ch, k=3)
        self.use_shortcut = shortcut and in_ch == out_ch

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        return x + out if self.use_shortcut else out

#-----------------------
# -----------c2f--------
#----------------------
class c2f(nn.Module):
    def __init__(self, in_ch, out_ch, n=1, shortcut =True, e=0.5 ):
        super().__init__()
        hidden_ch = int(out_ch * e) #channel per branch

        #conv awal: dinaikkan 2 karena split 
        self.conv1 = Conv(in_ch, 2 * hidden_ch, k = 1)

        #Bottleneck layers 
        self.bottlenecks = nn.ModuleList([
            Bottleneck(hidden_ch, hidden_ch, shortcut, e = 1.0)
            for _ in range (n)
        ])

        #conv akhir setelah concat 
        self.conv2 = Conv((n + 1) * hidden_ch, out_ch, k =1)

    def forward (self, x):
        y = self.conv1(x)
        x1, x2 = y.chunk(2, dim=1) #split channel ke 2 bagian 
        outs= [x1]

        for b in self.bottlenecks:
            x2 = b(x2)
            outs.append(x2) # menambah output dari setiap Bottleneck 
        
        out = torch.cat(outs, dim=1)
        return self.conv2(out)

#--------------------
#------Backbone-----
#------------------

