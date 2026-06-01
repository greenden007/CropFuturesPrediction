"""
Multi-Commodity Neural Network Architectures

Models that train jointly across corn, soybeans, and wheat:
- Shared encoder with commodity-specific heads
- Multi-task learning with shared representations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiCommodityLSTM(nn.Module):
    """
    Shared LSTM encoder with commodity-specific prediction heads.
    
    Learns common patterns across all commodities while having
    separate output layers for each.
    
    Args:
        input_dim: Number of input features (same for all commodities)
        hidden_dim: LSTM hidden dimension
        num_layers: Number of LSTM layers
        num_commodities: Number of commodities (3 for corn, soybeans, wheat)
        dropout: Dropout rate
    """
    def __init__(self, input_dim=30, hidden_dim=128, num_layers=2, 
                 num_commodities=3, dropout=0.2):
        super(MultiCommodityLSTM, self).__init__()
        
        self.num_commodities = num_commodities
        self.hidden_dim = hidden_dim
        
        # Shared LSTM encoder
        self.shared_lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Commodity-specific prediction heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(num_commodities)
        ])
        
    def forward(self, x, commodity_idx=None):
        """
        Args:
            x: (batch, seq_len, input_dim) - input sequence
            commodity_idx: (batch,) - which commodity each sample belongs to
                          If None, returns predictions for all commodities
        
        Returns:
            If commodity_idx is provided: (batch, 1) - prediction for that commodity
            If commodity_idx is None: (batch, num_commodities) - predictions for all
        """
        # Shared encoding
        out, _ = self.shared_lstm(x)
        out = out[:, -1, :]  # (batch, hidden_dim)
        
        if commodity_idx is not None:
            # Return prediction for specific commodity per sample
            batch_size = x.size(0)
            predictions = torch.zeros(batch_size, 1, device=x.device)
            
            for i in range(self.num_commodities):
                mask = (commodity_idx == i)
                if mask.any():
                    head_out = self.heads[i](out[mask])
                    predictions[mask] = head_out
            
            return predictions
        else:
            # Return predictions for all commodities
            all_preds = []
            for head in self.heads:
                pred = head(out)
                all_preds.append(pred)
            
            return torch.cat(all_preds, dim=1)  # (batch, num_commodities)


class MultiCommodityTransformer(nn.Module):
    """
    Transformer with shared encoder and multi-commodity heads.
    
    Args:
        input_dim: Input feature dimension
        d_model: Transformer model dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        num_commodities: Number of commodities
        dropout: Dropout rate
    """
    def __init__(self, input_dim=30, d_model=128, nhead=8, num_layers=4,
                 dim_feedforward=256, num_commodities=3, dropout=0.2):
        super(MultiCommodityTransformer, self).__init__()
        
        self.num_commodities = num_commodities
        self.d_model = d_model
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Shared transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        
        # Commodity-specific heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1)
            ) for _ in range(num_commodities)
        ])
    
    def forward(self, x, commodity_idx=None):
        """
        Args:
            x: (batch, seq_len, input_dim)
            commodity_idx: (batch,) - commodity indices, or None for all
        
        Returns:
            Predictions: (batch, 1) or (batch, num_commodities)
        """
        # Embed and add positional encoding
        out = self.input_embedding(x) * math.sqrt(self.d_model)
        out = self.pos_encoder(out)
        
        # Transformer encoding
        out = self.transformer_encoder(out)
        out = out.mean(dim=1)  # (batch, d_model)
        
        if commodity_idx is not None:
            # Specific commodity per sample
            batch_size = x.size(0)
            predictions = torch.zeros(batch_size, 1, device=x.device)
            
            for i in range(self.num_commodities):
                mask = (commodity_idx == i)
                if mask.any():
                    head_out = self.heads[i](out[mask])
                    predictions[mask] = head_out
            
            return predictions
        else:
            # All commodities
            all_preds = [head(out) for head in self.heads]
            return torch.cat(all_preds, dim=1)


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class CrossCommodityAttention(nn.Module):
    """
    Cross-commodity attention layer that allows information sharing
    between commodities.
    
    Takes features from all commodities and computes attention weights
    to aggregate relevant information.
    """
    def __init__(self, feature_dim, num_commodities=3):
        super(CrossCommodityAttention, self).__init__()
        
        self.num_commodities = num_commodities
        self.feature_dim = feature_dim
        
        # Query, Key, Value projections
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim * num_commodities, feature_dim)
        self.value = nn.Linear(feature_dim * num_commodities, feature_dim)
        
    def forward(self, commodity_features):
        """
        Args:
            commodity_features: list of (batch, feature_dim) tensors
                               one for each commodity
        
        Returns:
            List of (batch, feature_dim) - attended features for each commodity
        """
        batch_size = commodity_features[0].size(0)
        
        # Concatenate all commodity features
        all_features = torch.cat(commodity_features, dim=1)  # (batch, feat_dim * num_commodities)
        
        outputs = []
        for i, feat in enumerate(commodity_features):
            # Query from current commodity
            q = self.query(feat)  # (batch, feature_dim)
            
            # Key and Value from all commodities
            k = self.key(all_features)  # (batch, feature_dim)
            v = self.value(all_features)  # (batch, feature_dim)
            
            # Attention
            scores = torch.matmul(q.unsqueeze(1), k.unsqueeze(2)) / math.sqrt(self.feature_dim)
            attn = F.softmax(scores, dim=-1)
            
            # Weighted sum
            attended = torch.matmul(attn, v.unsqueeze(1)).squeeze(1)
            outputs.append(attended)
        
        return outputs


class MultiCommodityWithCrossAttention(nn.Module):
    """
    Multi-commodity model with cross-commodity attention.
    
    Learns to share information between commodities during encoding.
    """
    def __init__(self, input_dim=30, hidden_dim=128, num_layers=2,
                 num_commodities=3, dropout=0.2, use_cross_attention=True):
        super(MultiCommodityWithCrossAttention, self).__init__()
        
        self.num_commodities = num_commodities
        self.hidden_dim = hidden_dim
        self.use_cross_attention = use_cross_attention

        # Separate encoders for each commodity
        self.encoders = nn.ModuleList([
            nn.LSTM(input_dim, hidden_dim, num_layers,
                   batch_first=True, dropout=dropout if num_layers > 1 else 0)
            for _ in range(num_commodities)
        ])
        
        # Cross-commodity attention
        if use_cross_attention:
            self.cross_attn = CrossCommodityAttention(hidden_dim, num_commodities)
        
        # Fusion and prediction layers
        fusion_dim = hidden_dim * (2 if use_cross_attention else 1)
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.output = nn.Linear(hidden_dim, 1)
    
    def forward(self, x, commodity_idx):
        """
        Args:
            x: (batch, seq_len, input_dim) - input for specific commodity
            commodity_idx: (batch,) - which commodity this is
        
        Returns:
            predictions: (batch, 1)
        """
        batch_size = x.size(0)
        
        # Encode each sample with its commodity-specific encoder
        encoded = []
        for i in range(batch_size):
            c_idx = commodity_idx[i].item()
            enc_out, _ = self.encoders[c_idx](x[i:i+1])
            enc_out = enc_out[:, -1, :]  # (1, hidden_dim)
            encoded.append(enc_out)
        
        encoded = torch.cat(encoded, dim=0)  # (batch, hidden_dim)
        
        if self.use_cross_attention:
            # Group by commodity for cross-attention
            commodity_groups = [[] for _ in range(self.num_commodities)]
            group_indices = [[] for _ in range(self.num_commodities)]
            for i, c_idx in enumerate(commodity_idx):
                commodity_groups[c_idx.item()].append(encoded[i])
                group_indices[c_idx.item()].append(i)
            
            # Pad and stack for cross-attention
            max_samples = max((len(g) for g in commodity_groups), default=0)
            if max_samples > 0:
                padded_features = []
                for group in commodity_groups:
                    if group:
                        stacked = torch.stack(group)
                        # Pad if needed
                        if len(group) < max_samples:
                            padding = torch.zeros(max_samples - len(group),
                                                self.hidden_dim,
                                                device=x.device)
                            stacked = torch.cat([stacked, padding], dim=0)
                    else:
                        stacked = torch.zeros(max_samples,
                                            self.hidden_dim,
                                            device=x.device)
                    padded_features.append(stacked)

                # Apply cross-attention
                attended = self.cross_attn(padded_features)  # List of (max_samples, hidden_dim)

                # Extract attended features for each sample in original order
                attended_flat = torch.zeros_like(encoded)
                for c_idx, indices in enumerate(group_indices):
                    for i, orig_idx in enumerate(indices):
                        attended_flat[orig_idx] = attended[c_idx][i]

                # Concatenate original with attended
                combined = torch.cat([encoded, attended_flat], dim=1)
            else:
                combined = encoded
        else:
            combined = encoded
        
        # Fusion and prediction
        fused = self.fusion(combined)
        predictions = self.output(fused)
        
        return predictions


def create_multi_commodity_model(model_type='lstm', **kwargs):
    """
    Factory function for multi-commodity models.
    
    Args:
        model_type: 'lstm', 'transformer', or 'cross_attn'
        **kwargs: Model-specific arguments
    """
    if model_type == 'lstm':
        return MultiCommodityLSTM(**kwargs)
    elif model_type == 'transformer':
        return MultiCommodityTransformer(**kwargs)
    elif model_type == 'cross_attn':
        return MultiCommodityWithCrossAttention(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test multi-commodity models
    batch_size = 4
    seq_len = 20
    input_dim = 30
    
    print("Testing multi-commodity models...")
    
    # Test MultiCommodityLSTM
    print("\n1. MultiCommodityLSTM")
    model = MultiCommodityLSTM(input_dim=input_dim, num_commodities=3)
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Test with commodity indices
    commodity_idx = torch.tensor([0, 1, 2, 0])
    out = model(x, commodity_idx)
    print(f"   Input: {x.shape}, commodity_idx: {commodity_idx}")
    print(f"   Output (specific): {out.shape}")
    
    # Test all commodities
    out_all = model(x, None)
    print(f"   Output (all): {out_all.shape}")
    
    # Test MultiCommodityTransformer
    print("\n2. MultiCommodityTransformer")
    model = MultiCommodityTransformer(input_dim=input_dim, num_commodities=3)
    out = model(x, commodity_idx)
    print(f"   Output (specific): {out.shape}")
    out_all = model(x, None)
    print(f"   Output (all): {out_all.shape}")
    
    print("\nTests passed!")
