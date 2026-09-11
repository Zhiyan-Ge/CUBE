import math
import torch
import torch.nn as nn


class GlobalCoordinateEmbedding(nn.Module):
    def __init__(self, dim=256, num_freqs=64):
        super().__init__()
        self.register_buffer("frequencies", 2.0 ** torch.linspace(0,6,num_freqs))
        self.proj = nn.Linear(num_freqs*4, dim)

    def forward(self, coord):
        x = coord[:,0:1] * self.frequencies[None,:] * (2*math.pi)
        y = coord[:,1:2] * self.frequencies[None,:] * (2*math.pi)
        return self.proj(torch.cat([x.sin(),x.cos(),y.sin(),y.cos()], dim=-1)).unsqueeze(1)


class UR2Embedding(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1,16*16,dim))
        self.summary = nn.Parameter(torch.zeros(1,1,dim))
        self.global_coord = GlobalCoordinateEmbedding(dim)
        nn.init.trunc_normal_(self.pos,std=0.02); nn.init.trunc_normal_(self.summary,std=0.02)

    def forward(self, ur, coord):
        b = ur.size(0)
        x = ur.flatten(2).transpose(1,2) + self.pos
        x = torch.cat([self.summary.expand(b,-1,-1),x], dim=1)
        return x + self.global_coord(coord)
