from fastai.vision import *
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import optim
from torchsummary import summary
import gc
from fastai.callbacks import *
from fastai.utils.mem import *

from torchvision.models import vgg16_bn

# Perceptual Loss:
# original code from: fast.ai lesson ???
# modifications:
# - Instance Normalization of low-level features to remove influence of "style"
def gram_matrix(x):
    n,c,h,w = x.size()
    x = x.view(n, c, -1)
    return (x @ x.transpose(1,2))/(c*h*w)

vgg_m = vgg16_bn(True).features.cuda().eval()
requires_grad(vgg_m, False)
blocks = [i-1 for i,o in enumerate(children(vgg_m)) if isinstance(o,nn.MaxPool2d)]
base_loss = F.mse_loss

class FeatureLoss(nn.Module):
    def __init__(self, layer_ids, layer_wgts,without_instancenorm=1):
        super().__init__()
        self.m_feat = vgg_m
        
        # how many layers are not subjected to instance norm (starting from high-level, i.e. later layers)
        self.without_instancenorm = without_instancenorm
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = [f'feat_{i}' for i in range(len(layer_ids))
                                         ]+ [f'gram_{i}' for i in range(len(layer_ids))]
              

    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]
    
    def forward(self, input, target):
#         self.feat_losses = [dice_bce_loss(input, target)]
        
        input = torch.cat([input]*3,dim=1)*2-1#4-1.6
        target = torch.cat([target]*3,dim=1)*2-1#4-1.6
        
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)
        try:
            # instance normalization for all but last layer
            for l in range(len(in_feat)-self.without_instancenorm):
                in_feat[l] = nn.InstanceNorm2d(in_feat[l][1],momentum=0)(in_feat[l])
                out_feat[l] = nn.InstanceNorm2d(out_feat[l][1],momentum=0)(out_feat[l])
            
            
            self.feat_losses = [base_loss(f_in, f_out)*w
                                 for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
            self.feat_losses += [base_loss(gram_matrix(f_in), gram_matrix(f_out))*w**2 * 5e3
                                 for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        except IndexError: # No idea why this tends to happen (only during validation)
            self.feat_losses = [torch.tensor(1).float().to(device)]
            for k in range(6):
                self.feat_losses += [torch.tensor(1).float().to(device)]
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)
    
    def __del__(self): self.hooks.remove()
        
        
