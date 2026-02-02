import torch
from torch import nn
from .DIEM import DepressionInformationExtractionModule
from .SAFM import AttentionFusion

class AEPIM(nn.Module):
    def __init__(self):
        super().__init__()
        self.depressionFeature = DepressionInformationExtractionModule()
        self.fusion_module = AttentionFusion()
        self.regression = nn.Linear(128,1)
       
    def forward(self, x, embedding):
        """
        Personalized information (embedding) is extracted from the frozen ECAPA-TDNN model during the training phase.
        """
        depression_feature = self.depressionFeature(x)
        fused_representation = self.fusion_module(depression_feature, embedding)
        out = self.regression(fused_representation)
        return out

