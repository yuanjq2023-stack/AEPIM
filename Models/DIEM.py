import math
import torch
from torch import nn
from torch.nn import functional as F

class SE_Connect(nn.Module):
    def __init__(self, channels, s=16):
        super().__init__()
        assert channels % s == 0, "{} % {} != 0".format(channels, s)
        self.linear1 = nn.Linear(channels, channels // s)
        self.linear2 = nn.Linear(channels // s, channels)

    def forward(self, x):
        out = x.mean(dim=2)   
        out = torch.tanh(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        out = x * out.unsqueeze(2)
        return out

class CNNBlocks(nn.Module):
    def __init__(self, in_features=80):
        super().__init__()
        self.conv2d1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 3), padding=(0, 1))
        self.conv2d2 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 1), padding=(1, 0))
        self.conv2d3 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=(1, 1))
        self.conv2d4 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.GELU()
        )
        self.flatten_dim = 32 * in_features
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=self.flatten_dim, out_channels=512, kernel_size=5),
            nn.GELU(),
            nn.Conv1d(in_channels=512, out_channels=64, kernel_size=5),
            nn.GELU()
        )
        self.gelu = nn.GELU()
        self.se = SE_Connect(64)

    def forward(self,x):
        x = x.unsqueeze(1)
        conv1_out = self.gelu(self.conv2d1(x))
        conv2_out = self.gelu(self.conv2d2(x))
        conv3_out = self.gelu(self.conv2d3(x))
        conv_out = torch.cat((conv1_out, conv2_out, conv3_out), dim=1)
        conv4_out = self.conv2d4(conv_out)
        b, c, f, t = conv4_out.shape
        conv4_out = conv4_out.view(b, c * f, t)
        conv5_out = self.conv1d(conv4_out)
        res = conv5_out
        out = self.se(conv5_out) + res
        return out
    
class BiLSTMTransformer(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64, nhead=4, num_layers=1):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=1, 
            bidirectional=True, 
            batch_first=True
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim * 2, 
            nhead=nhead, 
            dim_feedforward=hidden_dim * 8, 
            dropout=0.1, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.layer_norm1 = nn.LayerNorm(hidden_dim * 2)
        self.layer_norm2 = nn.LayerNorm(192)
        self.dense = nn.Linear(hidden_dim * 2, 192)
        self.dropout = nn.Dropout(0.5)
        self.gelu = nn.GELU()

    def forward(self, x):
        """Input x: [B, C, T]"""
        x = x.transpose(1, 2)
        lstm_out, _ = self.bilstm(x)
        trans_out = self.transformer(lstm_out)
        x = self.layer_norm1(trans_out + lstm_out)
        x = torch.mean(x, dim=1) 
    
        x = self.dropout(x)
        x = self.dense(x)
        x = self.gelu(self.layer_norm2(x))
        return x


class DepressionInformationExtractionModule(nn.Module):
    """Depression Information Extraction Module (DIEM)"""
    def __init__(self, in_features=80):
        super().__init__()
        self.cnn_extractor = CNNBlocks(in_features=in_features)
        self.temporal_encoder = BiLSTMTransformer(input_dim=64)

    def forward(self, x):
        features = self.cnn_extractor(x)
        representation = self.temporal_encoder(features)
        return representation

    
