import torch
import torch.nn as nn
from models.climate_bert import ClimateTextEncoder

class SequenceEncoder(nn.Module):
    """LSTM-based time-series spatial encoder for structured sequences."""
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return h_n[-1] # Output of the last layer's final time step

class MultimodalAgriFuturesNet(nn.Module):
    def __init__(self, price_dim, climate_dim, text_embed_dim=128, hidden_dim=64):
        super().__init__()
        
        # Modality Encoders
        self.price_net = SequenceEncoder(input_dim=price_dim, hidden_dim=hidden_dim)
        self.climate_net = SequenceEncoder(input_dim=climate_dim, hidden_dim=hidden_dim)
        self.text_net = ClimateTextEncoder(hidden_dim=text_embed_dim)
        
        # Fusion Layer
        fused_dim = hidden_dim * 2 + text_embed_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.GELU(),
        )
        
        # Trading Prediction Head (Continuous expected return)
        self.regressor = nn.Linear(64, 1)

    def forward(self, price_seq, climate_seq, text_ids, text_mask):
        # 1. Extract Modality representations
        z_price = self.price_net(price_seq)
        z_climate = self.climate_net(climate_seq)
        
        if text_ids is not None:
             z_text = self.text_net(text_ids, text_mask)
        else:
             # Handle days with no reports (impute zero tensor)
             z_text = torch.zeros(price_seq.size(0), self.text_net.fc[0].out_features).to(price_seq.device)
             
        # 2. Concat Fusion
        z_fused = torch.cat([z_price, z_climate, z_text], dim=1)
        
        # 3. Predict Return
        dense_repr = self.fusion_mlp(z_fused)
        expected_return = self.regressor(dense_repr)
        
        return expected_return
