"""
Attention-Based Vision-Enabled Model for Crop Futures Prediction

This model makes vision (weather outlook images) a mandatory input.
Features:
- Three-stream architecture: Price, Fundamentals, Vision
- Cross-stream attention between all three inputs
- WeatherOutlookEncoder for processing CPC outlook images
- Gating mechanism to learn stream importance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from image_encoder import WeatherOutlookEncoder


class CrossStreamAttention3Way(nn.Module):
    """
    Cross-attention between three streams: price, fundamentals, vision.
    Each stream attends to the other two.
    """
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Three separate attention modules for each pair
        self.attn_price_fund = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.attn_price_vision = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.attn_fund_vision = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Layer norms
        self.norm_price = nn.LayerNorm(hidden_dim)
        self.norm_fund = nn.LayerNorm(hidden_dim)
        self.norm_vision = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, price_feat, fund_feat, vision_feat):
        """
        Args:
            price_feat: (batch, hidden_dim)
            fund_feat: (batch, hidden_dim)
            vision_feat: (batch, hidden_dim)
        Returns:
            enhanced features for each stream
        """
        # Stack as sequences for attention
        price_query = price_feat.unsqueeze(1)  # (batch, 1, hidden)
        fund_query = fund_feat.unsqueeze(1)
        vision_query = vision_feat.unsqueeze(1)
        
        # Concatenate for keys/values
        fund_vision_kv = torch.stack([fund_feat, vision_feat], dim=1)  # (batch, 2, hidden)
        price_vision_kv = torch.stack([price_feat, vision_feat], dim=1)
        price_fund_kv = torch.stack([price_feat, fund_feat], dim=1)
        
        # Cross-attention: each stream attends to the other two
        price_attended, _ = self.attn_price_fund(price_query, fund_vision_kv, fund_vision_kv)
        fund_attended, _ = self.attn_fund_vision(fund_query, price_vision_kv, price_vision_kv)
        vision_attended, _ = self.attn_price_vision(vision_query, price_fund_kv, price_fund_kv)
        
        # Residual connections
        price_enhanced = self.norm_price(price_feat + self.dropout(price_attended.squeeze(1)))
        fund_enhanced = self.norm_fund(fund_feat + self.dropout(fund_attended.squeeze(1)))
        vision_enhanced = self.norm_vision(vision_feat + self.dropout(vision_attended.squeeze(1)))
        
        return price_enhanced, fund_enhanced, vision_enhanced


class AttentionVisionDualStreamLSTM(nn.Module):
    """
    Three-Stream LSTM with Vision (MANDATORY)
    
    Architecture:
    1. Price LSTM - processes OHLCV data
    2. Fundamentals LSTM - processes WASDE, crop progress, weather indices
    3. Vision Encoder + LSTM - processes CPC weather outlook images
    
    Cross-stream attention fuses information from all three sources.
    """
    def __init__(self, 
                 price_input_dim=10, 
                 fund_input_dim=20,
                 vision_feature_dim=64,
                 hidden_dim=128,
                 num_layers=2,
                 output_dim=1,
                 dropout=0.2,
                 num_images=6):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_images = num_images
        
        # === Stream 1: Price LSTM ===
        self.price_lstm = nn.LSTM(
            price_input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # === Stream 2: Fundamentals LSTM ===
        self.fund_lstm = nn.LSTM(
            fund_input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # === Stream 3: Vision Encoder + LSTM ===
        # Weather image encoder (outputs feature_dim per image)
        self.vision_encoder = WeatherOutlookEncoder(feature_dim=vision_feature_dim)
        
        # Aggregate multiple images per timestep
        self.vision_aggregator = nn.Sequential(
            nn.Linear(vision_feature_dim * num_images, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Temporal processing for vision features
        self.vision_lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # === Cross-Stream Attention ===
        self.cross_attention = CrossStreamAttention3Way(hidden_dim, num_heads=4, dropout=dropout)
        
        # === Gating Mechanism ===
        # Learns how much to weight each of the 3 streams
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, 3),  # Output 3 weights
            nn.Softmax(dim=1)  # Normalize to sum to 1
        )
        
        # === Merge Network ===
        self.merge_fc = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # === Output Layer ===
        self.output_fc = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, output_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Xavier/orthogonal initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name and param.dim() >= 1:
                nn.init.zeros_(param)
    
    def forward(self, price_x, fund_x, vision_x):
        """
        Args:
            price_x: (batch, seq_len, price_input_dim)
            fund_x: (batch, seq_len, fund_input_dim)
            vision_x: (batch, seq_len, num_images, 1, 64, 64) - weather images
        
        Returns:
            output: (batch, output_dim)
        """
        batch_size, seq_len = price_x.size(0), price_x.size(1)
        
        # === Process Price Stream ===
        price_out, _ = self.price_lstm(price_x)  # (batch, seq_len, hidden)
        price_last = price_out[:, -1, :]  # (batch, hidden)
        
        # === Process Fundamentals Stream ===
        fund_out, _ = self.fund_lstm(fund_x)  # (batch, seq_len, hidden)
        fund_last = fund_out[:, -1, :]  # (batch, hidden)
        
        # === Process Vision Stream ===
        # Encode all images: (batch*seq_len*num_images, 1, 64, 64)
        num_images = vision_x.size(2)
        vision_flat = vision_x.view(batch_size * seq_len * num_images, 1, 64, 64)
        vision_features = self.vision_encoder(vision_flat)  # (batch*seq_len*num_images, vision_feature_dim)
        
        # Reshape back: (batch, seq_len, num_images, vision_feature_dim)
        vision_features = vision_features.view(batch_size, seq_len, num_images, -1)
        
        # Flatten images per timestep: (batch, seq_len, num_images*vision_feature_dim)
        vision_features = vision_features.view(batch_size, seq_len, -1)
        
        # Aggregate and process temporally
        vision_aggregated = self.vision_aggregator(vision_features)  # (batch, seq_len, hidden)
        vision_temporal, _ = self.vision_lstm(vision_aggregated)
        vision_last = vision_temporal[:, -1, :]  # (batch, hidden)
        
        # === Cross-Stream Attention ===
        price_enhanced, fund_enhanced, vision_enhanced = self.cross_attention(
            price_last, fund_last, vision_last
        )
        
        # === Gating ===
        combined = torch.cat([price_enhanced, fund_enhanced, vision_enhanced], dim=1)
        gate_weights = self.gate(combined)  # (batch, 3)
        
        # Apply gates
        price_gated = price_enhanced * gate_weights[:, 0:1]
        fund_gated = fund_enhanced * gate_weights[:, 1:2]
        vision_gated = vision_enhanced * gate_weights[:, 2:3]
        
        # === Merge and Predict ===
        merged = torch.cat([price_gated, fund_gated, vision_gated], dim=1)
        features = self.merge_fc(merged)
        output = self.output_fc(features)
        
        return output
    
    def get_attention_weights(self, price_x, fund_x, vision_x):
        """
        Get the gating weights to interpret stream importance.
        Useful for analysis.
        
        Returns:
            gate_weights: (batch, 3) - [price_weight, fund_weight, vision_weight]
        """
        self.eval()
        with torch.no_grad():
            batch_size, seq_len = price_x.size(0), price_x.size(1)
            
            # Forward through LSTMs
            price_out, _ = self.price_lstm(price_x)
            price_last = price_out[:, -1, :]
            
            fund_out, _ = self.fund_lstm(fund_x)
            fund_last = fund_out[:, -1, :]
            
            # Vision processing
            num_images = vision_x.size(2)
            vision_flat = vision_x.view(batch_size * seq_len * num_images, 1, 64, 64)
            vision_features = self.vision_encoder(vision_flat)
            vision_features = vision_features.view(batch_size, seq_len, num_images, -1)
            vision_features = vision_features.view(batch_size, seq_len, -1)
            vision_aggregated = self.vision_aggregator(vision_features)
            vision_temporal, _ = self.vision_lstm(vision_aggregated)
            vision_last = vision_temporal[:, -1, :]
            
            # Get gate weights
            combined = torch.cat([price_last, fund_last, vision_last], dim=1)
            gate_weights = self.gate(combined)
            
        return gate_weights


def test_model():
    """Test the vision-enabled model."""
    batch_size = 4
    seq_len = 20
    price_dim = 10
    fund_dim = 20
    num_images = 6
    
    print("Testing AttentionVisionDualStreamLSTM...")
    print(f"Batch size: {batch_size}, Sequence length: {seq_len}")
    
    model = AttentionVisionDualStreamLSTM(
        price_input_dim=price_dim,
        fund_input_dim=fund_dim,
        vision_feature_dim=64,
        hidden_dim=128,
        num_layers=2,
        output_dim=1,
        dropout=0.2,
        num_images=num_images
    )
    
    # Create dummy inputs
    price_x = torch.randn(batch_size, seq_len, price_dim)
    fund_x = torch.randn(batch_size, seq_len, fund_dim)
    vision_x = torch.randn(batch_size, seq_len, num_images, 1, 64, 64)
    
    print(f"\nInputs:")
    print(f"  Price: {price_x.shape}")
    print(f"  Fund: {fund_x.shape}")
    print(f"  Vision: {vision_x.shape}")
    
    # Forward pass
    output = model(price_x, fund_x, vision_x)
    print(f"\nOutput: {output.shape}")
    
    # Get attention weights
    gate_weights = model.get_attention_weights(price_x, fund_x, vision_x)
    print(f"\nGate weights (avg): Price={gate_weights[:, 0].mean():.3f}, "
          f"Fund={gate_weights[:, 1].mean():.3f}, "
          f"Vision={gate_weights[:, 2].mean():.3f}")
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {n_params:,}")
    
    print("\nTest passed!")


if __name__ == "__main__":
    test_model()
