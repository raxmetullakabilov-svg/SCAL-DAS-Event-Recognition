import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


def _gn(ch):
    g=8
    while ch % g != 0 and g > 1:
        g //= 2
    return nn.GroupNorm(g, ch)

class ResidualBlock(nn.Module):
    def __init__(self, cin, cout, stride=(1,1)):
        super().__init__()
        self.c1=nn.Conv2d(cin,cout,3,stride=stride,padding=1,bias=False)
        self.n1=_gn(cout)
        self.c2=nn.Conv2d(cout,cout,3,padding=1,bias=False)
        self.n2=_gn(cout)
        self.skip=nn.Identity() if cin==cout and stride==(1,1) else nn.Sequential(
            nn.Conv2d(cin,cout,1,stride=stride,bias=False), _gn(cout))
    def forward(self,x):
        z=F.gelu(self.n1(self.c1(x)))
        z=self.n2(self.c2(z))
        return F.gelu(z+self.skip(x))

class TSResNetSmall(nn.Module):
    def __init__(self, embedding_dim=192):
        super().__init__()
        self.stem=nn.Sequential(
            nn.Conv2d(1,32,kernel_size=(31,7),stride=(8,2),padding=(15,3),bias=False),
            _gn(32), nn.GELU(),
            nn.MaxPool2d(kernel_size=(4,2),stride=(4,2)),
        )
        self.body=nn.Sequential(
            ResidualBlock(32,32),
            ResidualBlock(32,64,(2,2)),
            ResidualBlock(64,64),
            ResidualBlock(64,128,(2,2)),
            ResidualBlock(128,128),
            ResidualBlock(128,embedding_dim,(2,2)),
            ResidualBlock(embedding_dim,embedding_dim),
        )
        self.pool=nn.AdaptiveAvgPool2d(1)
    def forward(self,x):
        return self.pool(self.body(self.stem(x))).flatten(1)

class _GRL(Function):
    @staticmethod
    def forward(ctx,x,lambd):
        ctx.lambd=float(lambd)
        return x.view_as(x)
    @staticmethod
    def backward(ctx,g):
        return -ctx.lambd*g, None

def grl(x,lambd):
    return _GRL.apply(x,lambd)

def grl_schedule(progress,lambda_max=0.15,gamma=10.0):
    p=max(0.0,min(1.0,float(progress)))
    return float(lambda_max*(2.0/(1.0+math.exp(-gamma*p))-1.0))

class RDFTPreprocessor(nn.Module):
    def __init__(self,time_len=8192,redundancy=3,keep_bins=2048):
        super().__init__()
        self.time_len=int(time_len); self.redundancy=int(redundancy); self.keep_bins=int(keep_bins)
        self.n_fft=self.time_len*self.redundancy
        self.register_buffer('hann',torch.hann_window(self.time_len,periodic=False,dtype=torch.float32),persistent=True)
    def forward(self,x):
        if x.ndim!=4 or x.shape[1]!=1 or x.shape[2]!=self.time_len:
            raise ValueError(f'Expected [B,1,{self.time_len},C], got {tuple(x.shape)}')
        s=x[:,0].float()*self.hann[None,:,None]
        mag=torch.fft.rfft(s,n=self.n_fft,dim=1).abs()
        mag=mag[:,:self.keep_bins,:].to(torch.float32)
        logmag=torch.log10(torch.clamp(mag,min=1e-6))
        logmag=logmag-logmag.mean(dim=1,keepdim=True)
        return logmag.unsqueeze(1)

class EventClassifier(nn.Module):
    def __init__(self,n_classes=9,embedding_dim=192):
        super().__init__(); self.backbone=TSResNetSmall(embedding_dim); self.head=nn.Linear(embedding_dim,n_classes)
    def forward(self,x,return_embedding=False):
        z=self.backbone(x); logits=self.head(z)
        return (logits,z) if return_embedding else logits

class SCALDAS(nn.Module):
    def __init__(self,n_classes=9,embedding_dim=192):
        super().__init__(); self.n_classes=n_classes
        self.backbone=TSResNetSmall(embedding_dim); self.event_head=nn.Linear(embedding_dim,n_classes)
        self.loc_head=nn.Sequential(nn.Linear(embedding_dim+n_classes,128),nn.GELU(),nn.Dropout(0.1),
                                    nn.Linear(128,64),nn.GELU(),nn.Linear(64,1),nn.Sigmoid())
    def forward(self,x,y=None,grl_lambda=0.0,return_embedding=False):
        z=self.backbone(x); logits=self.event_head(z); loc_pred=None
        if y is not None:
            cond=F.one_hot(y,num_classes=self.n_classes).float().detach()
            loc_pred=self.loc_head(torch.cat([grl(z,grl_lambda),cond],dim=1)).squeeze(1)
        if return_embedding: return logits,loc_pred,z
        return logits,loc_pred

class RDFTSpectrumClassifier(nn.Module):
    def __init__(self,n_classes=9,embedding_dim=192):
        super().__init__(); self.pre=RDFTPreprocessor(); self.backbone=TSResNetSmall(embedding_dim); self.head=nn.Linear(embedding_dim,n_classes)
    def forward(self,x,return_embedding=False):
        sx=self.pre(x); z=self.backbone(sx); logits=self.head(z)
        return (logits,z) if return_embedding else logits

class SCALRDFT(nn.Module):
    def __init__(self,n_classes=9,embedding_dim=192):
        super().__init__(); self.n_classes=n_classes; self.pre=RDFTPreprocessor()
        self.backbone=TSResNetSmall(embedding_dim); self.event_head=nn.Linear(embedding_dim,n_classes)
        self.loc_head=nn.Sequential(nn.Linear(embedding_dim+n_classes,128),nn.GELU(),nn.Dropout(0.1),
                                    nn.Linear(128,64),nn.GELU(),nn.Linear(64,1),nn.Sigmoid())
    def forward(self,x,y=None,grl_lambda=0.0,return_embedding=False):
        sx=self.pre(x); z=self.backbone(sx); logits=self.event_head(z); loc_pred=None
        if y is not None:
            cond=F.one_hot(y,num_classes=self.n_classes).float().detach()
            loc_pred=self.loc_head(torch.cat([grl(z,grl_lambda),cond],dim=1)).squeeze(1)
        if return_embedding: return logits,loc_pred,z
        return logits,loc_pred

def build_model(model_name,n_classes=9,embedding_dim=192):
    m=model_name.lower()
    if m=='raw_ts_resnet': return EventClassifier(n_classes,embedding_dim)
    if m=='raw_scal_das': return SCALDAS(n_classes,embedding_dim)
    if m=='spectrum_rdft': return RDFTSpectrumClassifier(n_classes,embedding_dim)
    if m=='scal_rdft': return SCALRDFT(n_classes,embedding_dim)
    raise ValueError(model_name)

def is_scal(model_name): return model_name.lower() in {'raw_scal_das','scal_rdft'}
def is_spectral(model_name): return model_name.lower() in {'spectrum_rdft','scal_rdft'}
