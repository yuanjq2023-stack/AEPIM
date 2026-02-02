import math
import torch
from torch import nn
from torch.nn import functional as F

class AttentionLayer(nn.Module):
    def __init__(self, feature_dim=192):
        super().__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.scale = feature_dim ** -0.5

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        attn_score = torch.sigmoid((q * k) * self.scale)
        v = self.value(x)
        output = v * attn_score
        return output


class AttentionFusion(nn.Module):
    def __init__(self, feature_dim=192):
        super().__init__()
        self.attention1 = AttentionLayer()
        self.attention2 = AttentionLayer()
        self.fc = nn.Linear(feature_dim*2, 128)
        self.gelu = nn.GELU()
        
    def forward(self, x, embedding):
        feature1 = self.attention1(x)
        feature2 = self.attention2(embedding)
        fusion_feature = torch.cat((feature1, feature2),dim=1)
        fusion_feature = self.gelu(self.fc(fusion_feature))
        return fusion_feature