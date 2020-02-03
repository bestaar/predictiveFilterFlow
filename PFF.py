import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F

# predictive filter flow layer
# https://arxiv.org/abs/1811.11482
# Kong, S., & Fowlkes, C. (2018). Image reconstruction with predictive filter flow. arXiv preprint arXiv:1811.11482.
# - learn and apply individual filters (ksize x ksize) for each spatial position in the input
# - i.e. when using softmax activation it is basically image warping, but instead of offsets, we learn filters
class pFF(nn.Module):
    def __init__(self,ni, ksize=3,dilation=1,softmax = True,upsample=1):
        super(pFF, self).__init__()
        # size of the learned filter: ksize x ksize
        self.ksize=ksize
        # use softmax or tanh
        self.softmax = softmax
        # upsampling of the learned filters (gives smoother result)
        self.upsample = upsample
        # train conv layer to output filter flow and use reflection padding
        self.get_filter = nn.Conv2d(ni,ksize**2,3,padding=1,stride=upsample,padding_mode='reflect')
        self.pad = nn.ReflectionPad2d(padding=int((ksize-1)/2)*stride)
        # apply learned filters
        self.uf1 = nn.Unfold(ksize, dilation=dilation, padding=0, stride=1)
        self.uf2 = nn.Unfold(1, dilation=1, padding=0, stride=1)
        if upsample>1:
            self.us = nn.UpsamplingBilinear2d(scale_factor=upsample)
        
    def forward(self, features,inpt):
        # features: features learned by CNN, inpt: input that filters should be applied to
        # 1: get filter
        ff = self.get_filter(features)
        # 2: apply activation function
        if self.softmax:
#             ff = F.gumbel_softmax(ff,dim=1)
            ff = F.softmax(ff,dim=1)
        else:
            ff = torch.tanh(ff)
        if self.upsample>1:
            ff = self.us(ff)
            
        # apply learned filters
        inp_pad = self.pad(inpt)
        
        # use filter on each channel/feature of the input
        ff = torch.cat([ff]*inpt.shape[1],dim=1)
        out = self.uf1(inp_pad) * self.uf2(ff)
        out = out.view(-1,inpt.shape[1],self.ksize**2,inpt.shape[2],inpt.shape[3])
        return out.sum(dim=2)



class pFF_channelwise(nn.Module):
    # allow different deformations for each channel
    #   -> different filters for each channel can slow things down a lot
    #   to allow for some flexibility I implemented the option to split the channels into groups,
    #   each with their own deformation as a way to trade-off variety in deformations for performance
    def __init__(self,ni, ksize=3,dilation=1,softmax = True,groups=2,upsample=1):
        super(pFF_channelwise, self).__init__()
        # size of the learned filter: ksize x ksize
        self.ksize=ksize
        # use softmax or tanh
        self.softmax = softmax
        # upsampling of the learned filters (gives smoother result)
        self.upsample = upsample
        # different filters for each channel can slow things down a lot
        # to allow for some flexibility I implemted the option to split the channels into groups,
        # each with their own deformation
        if ni%groups!=0:
            raise ValueError('The number of input features must be divisible by the number of groups')
        self.groups = groups
        # train conv layer to output filter flow and use reflection padding
        self.get_filter = nn.Conv2d(ni,ksize**2*groups,3,padding=1,stride=upsample,padding_mode='reflect')
        self.pad = nn.ReflectionPad2d(padding=int((ksize-1)/2)*dilation)
        # apply learned filters
        self.uf1 = nn.Unfold(ksize, dilation=dilation, padding=0, stride=1)
        self.uf2 = nn.Unfold(1, dilation=1, padding=0, stride=1)
        if upsample>1:
            self.us = nn.UpsamplingBilinear2d(scale_factor=upsample)
        
    def forward(self, features,inpt):
        # features: features learned by CNN, inpt: input that filters should be applied to
        # 1: get filter
        ff = self.get_filter(features)
        # 2: apply activation function
        if self.softmax:
#             ff = F.gumbel_softmax(ff,dim=1)
            ff = F.softmax(ff.view(-1,self.ksize**2,self.groups,ff.shape[2],ff.shape[3]),dim=1)
            ff = ff.view(-1,self.ksize**2*self.groups,ff.shape[3],ff.shape[4])
        else:
            ff = torch.tanh(ff)
        if self.upsample>1:
            ff = self.us(ff)
            
        # apply learned filters
        inp_pad = self.pad(inpt)
        
        # use filter on each channel/feature of the input
        ff = torch.cat([ff]*int(inpt.shape[1]/self.groups),dim=1)
        out = self.uf1(inp_pad) * self.uf2(ff)
        out = out.view(-1,inpt.shape[1],self.ksize**2,inpt.shape[2],inpt.shape[3])
        return out.sum(dim=2)
