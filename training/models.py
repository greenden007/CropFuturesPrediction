"""
Neural Network Architectures for Crop Futures Prediction

Implements 5 architectures:
1. DualStreamLSTM - Parallel LSTM streams for different feature types
2. VisionDualStreamLSTM - Triple-stream with vision encoder for weather images
3. ResNet1D - 1D Residual CNN for time series
4. TransformerModel - Multi-head attention transformer
5. TemporalFusionTransformer - TFT-style temporal architecture
6. GRUModel - Standard GRU baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from image_encoder import WeatherOutlookEncoder
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False


class DualStreamLSTM(nn.Module):
    """
    Dual-Stream LSTM Architecture
    
    Two parallel LSTM streams that process different feature groups,
    then merge for final prediction. Useful for handling heterogeneous
    data (e.g., price stream + fundamentals stream).
    
    Args:
        price_input_dim: Dimension of price/volume features
        fund_input_dim: Dimension of fundamental features (WASDE, crop progress)
        hidden_dim: LSTM hidden dimension
        num_layers: Number of LSTM layers per stream
        output_dim: Output dimension (1 for regression)
        dropout: Dropout rate
    """
    def __init__(self, price_input_dim=10, fund_input_dim=20, hidden_dim=64,
                 num_layers=2, output_dim=1, dropout=0.2):
        super(DualStreamLSTM, self).__init__()
        
        # Price stream LSTM
        self.price_lstm = nn.LSTM(
            price_input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Fundamentals stream LSTM
        self.fund_lstm = nn.LSTM(
            fund_input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Merge layer
        self.merge_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output layer
        self.output_fc = nn.Linear(hidden_dim // 2, output_dim)
        
    def forward(self, price_x, fund_x):
        """
        Args:
            price_x: (batch, seq_len, price_input_dim)
            fund_x: (batch, seq_len, fund_input_dim)
        Returns:
            output: (batch, output_dim)
        """
        # Both streams must have the same sequence length
        # (price data is daily, fundamentals are monthly - resampling required)
        assert price_x.size(1) == fund_x.size(1), \
            f"Sequence length mismatch: price={price_x.size(1)}, fund={fund_x.size(1)}"

        # Process price stream
        price_out, _ = self.price_lstm(price_x)
        price_out = price_out[:, -1, :]  # Take last timestep
        
        # Process fundamentals stream
        fund_out, _ = self.fund_lstm(fund_x)
        fund_out = fund_out[:, -1, :]  # Take last timestep
        
        # Concatenate streams
        merged = torch.cat([price_out, fund_out], dim=1)
        
        # Merge and predict
        features = self.merge_fc(merged)
        output = self.output_fc(features)
        
        return output


class VisionDualStreamLSTM(nn.Module):
    """
    Triple-Stream LSTM with Vision Encoder for Weather Images
    
    Extends DualStreamLSTM with a third stream that processes CPC weather
    outlook images (6-10 day and 8-14 day forecasts) using a CNN encoder.
    
    Args:
        price_input_dim: Dimension of price/volume features
        fund_input_dim: Dimension of fundamental features (WASDE, crop progress)
        vision_feature_dim: Output dimension of image encoder (default: 64)
        hidden_dim: LSTM hidden dimension
        num_layers: Number of LSTM layers per stream
        output_dim: Output dimension (1 for regression)
        dropout: Dropout rate
        num_images: Number of weather images per sample (default: 6)
    """
    def __init__(self, price_input_dim=10, fund_input_dim=20, vision_feature_dim=64,
                 hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2, num_images=6):
        super(VisionDualStreamLSTM, self).__init__()
        
        if not VISION_AVAILABLE:
            raise ImportError("WeatherOutlookEncoder not available. Check image_encoder.py exists.")
        
        # LSTM streams (same as DualStreamLSTM)
        self.price_lstm = nn.LSTM(
            price_input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        self.fund_lstm = nn.LSTM(
            fund_input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Vision encoder for weather images
        self.vision_encoder = WeatherOutlookEncoder(feature_dim=vision_feature_dim)
        
        # Vision aggregation: combine multiple images per timestep
        self.num_images = num_images
        self.vision_aggregator = nn.Sequential(
            nn.Linear(vision_feature_dim * num_images, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Temporal processing for vision features
        self.vision_temporal = nn.LSTM(
            hidden_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Merge layer: price + fundamentals + vision
        self.merge_fc = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output layer
        self.output_fc = nn.Linear(hidden_dim, output_dim)
        
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
        
        # Validate sequence lengths
        assert price_x.size(1) == fund_x.size(1), \
            f"Sequence length mismatch: price={price_x.size(1)}, fund={fund_x.size(1)}"
        assert vision_x.size(1) == seq_len, \
            f"Sequence length mismatch: vision={vision_x.size(1)}, expected={seq_len}"
        
        # Process price stream
        price_out, _ = self.price_lstm(price_x)
        price_out = price_out[:, -1, :]  # (batch, hidden_dim)
        
        # Process fundamentals stream
        fund_out, _ = self.fund_lstm(fund_x)
        fund_out = fund_out[:, -1, :]  # (batch, hidden_dim)
        
        # Process vision stream: encode each image
        # Reshape vision_x for batch processing: (batch*seq_len*num_images, 1, 64, 64)
        num_images = vision_x.size(2)
        vision_flat = vision_x.view(batch_size * seq_len * num_images, 1, 64, 64)
        
        # Encode all images
        vision_features = self.vision_encoder(vision_flat)  # (batch*seq_len*num_images, vision_feature_dim)
        
        # Reshape back: (batch, seq_len, num_images, vision_feature_dim)
        vision_features = vision_features.view(batch_size, seq_len, num_images, -1)
        
        # Flatten images per timestep and aggregate
        vision_features = vision_features.view(batch_size, seq_len, -1)  # (batch, seq_len, num_images*vision_feature_dim)
        vision_aggregated = self.vision_aggregator(vision_features)  # (batch, seq_len, hidden_dim)
        
        # Temporal processing for vision
        vision_temporal_out, _ = self.vision_temporal(vision_aggregated)
        vision_out = vision_temporal_out[:, -1, :]  # (batch, hidden_dim)
        
        # Concatenate all three streams
        merged = torch.cat([price_out, fund_out, vision_out], dim=1)  # (batch, hidden_dim*3)
        
        # Merge and predict
        features = self.merge_fc(merged)
        output = self.output_fc(features)
        
        return output


class ResidualBlock(nn.Module):
    """1D Residual Block for ResNet1D"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout=0.2):
        super(ResidualBlock, self).__init__()
        
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        out += self.shortcut(x)  # Residual connection
        out = F.relu(out)
        return out


class ResNet1D(nn.Module):
    """
    1D ResNet Architecture for Time Series
    
    Applies ResNet-style residual convolutions to time series data.
    Good for capturing local patterns and trends.
    
    Args:
        input_dim: Number of input features
        seq_len: Length of input sequence
        num_blocks: Number of residual blocks
        hidden_dim: Base number of channels
        output_dim: Output dimension
        dropout: Dropout rate
    """
    def __init__(self, input_dim=30, seq_len=20, num_blocks=3, hidden_dim=64,
                 output_dim=1, dropout=0.2):
        super(ResNet1D, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        # Residual blocks with progressive channel doubling
        self.blocks = nn.ModuleList()
        channels = [hidden_dim] + [hidden_dim * (2 ** (i+1)) for i in range(num_blocks)]
        for i in range(num_blocks):
            self.blocks.append(
                ResidualBlock(channels[i], channels[i+1], dropout=dropout)
            )

        # Global average pooling and FC
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        final_dim = channels[-1]
        self.fc = nn.Sequential(
            nn.Linear(final_dim, final_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim // 2, output_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            output: (batch, output_dim)
        """
        # Transpose to (batch, input_dim, seq_len) for Conv1d
        x = x.transpose(1, 2)
        
        # Initial conv
        out = F.relu(self.bn1(self.conv1(x)))
        
        # Residual blocks
        for block in self.blocks:
            out = block(out)
        
        # Global pooling
        out = self.avg_pool(out).squeeze(-1)
        
        # FC layers
        output = self.fc(out)
        
        return output


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""
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


class TransformerModel(nn.Module):
    """
    Transformer Architecture for Time Series
    
    Multi-head attention model with positional encoding.
    Good for capturing long-range dependencies.
    
    Args:
        input_dim: Number of input features
        d_model: Model dimension (embedding size)
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        dim_feedforward: FFN hidden dimension
        output_dim: Output dimension
        dropout: Dropout rate
        max_seq_len: Maximum sequence length
    """
    def __init__(self, input_dim=30, d_model=128, nhead=8, num_layers=4,
                 dim_feedforward=256, output_dim=1, dropout=0.2, max_seq_len=100):
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer encoder
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
        
        # Output layers
        self.output_fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            output: (batch, output_dim)
        """
        # Embed and add positional encoding
        out = self.input_embedding(x) * math.sqrt(self.d_model)
        out = self.pos_encoder(out)
        
        # Transformer encoding
        out = self.transformer_encoder(out)

        # Take last timestep (consistent with LSTM/GRU models)
        out = out[:, -1, :]
        
        # Output
        output = self.output_fc(out)
        
        return output


class PatchTSTModel(nn.Module):
    """PatchTST-style model: patchify then encode with Transformer."""
    def __init__(self, input_dim=30, seq_len=20, d_model=128, nhead=8, num_layers=3,
                 patch_len=4, stride=2, dim_feedforward=256, output_dim=1, dropout=0.2, **kwargs):
        super(PatchTSTModel, self).__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.input_dim = input_dim

        # Number of temporal patches per feature channel.
        self.n_patches = max(1, (seq_len - patch_len) // stride + 1)
        self.patch_proj = nn.Linear(patch_len, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.n_patches)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        patches = x.unfold(dimension=2, size=self.patch_len, step=self.stride)  # (b, c, n_p, patch_len)
        b, c, n_p, p = patches.shape
        tokens = patches.reshape(b * c, n_p, p)
        tokens = self.patch_proj(tokens)
        tokens = self.pos_encoder(tokens)
        encoded = self.encoder(tokens)
        pooled = encoded.mean(dim=1).reshape(b, c, -1).mean(dim=1)  # channel-independence style pooling
        return self.head(pooled)


class NBeatsBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, theta_dim):
        super(NBeatsBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, theta_dim)
        )
        self.backcast = nn.Linear(theta_dim, input_dim)
        self.forecast = nn.Linear(theta_dim, 1)

    def forward(self, x):
        theta = self.fc(x)
        return self.backcast(theta), self.forecast(theta)


class NBeatsModel(nn.Module):
    """Simple N-BEATS style MLP stack for point forecasting."""
    def __init__(self, input_dim=30, seq_len=20, hidden_dim=256, num_blocks=4, output_dim=1, dropout=0.0, **kwargs):
        super(NBeatsModel, self).__init__()
        self.flat_dim = input_dim * seq_len
        self.blocks = nn.ModuleList(
            [NBeatsBlock(self.flat_dim, hidden_dim, hidden_dim // 2) for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.output_dim = output_dim

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        residual = x.reshape(x.size(0), -1)
        forecast = torch.zeros((x.size(0), 1), dtype=x.dtype, device=x.device)
        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = self.dropout(residual - backcast)
            forecast = forecast + block_forecast
        return forecast


class GatedResidualNetwork(nn.Module):
    """Core TFT block: non-linear transform with skip-connection gating."""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super(GatedResidualNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Linear(output_dim, output_dim)
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        h = F.elu(self.fc1(x))
        h = self.dropout(self.fc2(h))
        gated = torch.sigmoid(self.gate(h)) * h
        return self.norm(gated + self.skip(x))


class TemporalFusionTransformer(nn.Module):
    """Lightweight TFT-style model for temporal tabular forecasting."""
    def __init__(self, input_dim=30, hidden_dim=128, num_layers=2, nhead=4,
                 output_dim=1, dropout=0.2, max_seq_len=100):
        super(TemporalFusionTransformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=max_seq_len)

        self.variable_grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.temporal_grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=True)
        self.attn_grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        h = self.input_proj(x)
        h = self.pos_encoder(h)
        h = self.variable_grn(h)
        h, _ = self.lstm(h)
        h = self.temporal_grn(h)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        h = self.attn_grn(attn_out + h)
        return self.out(h[:, -1, :])


class GRUModel(nn.Module):
    """
    GRU (Gated Recurrent Unit) Baseline
    
    Simpler than LSTM with fewer parameters. Good baseline for
    time series forecasting.
    
    Args:
        input_dim: Number of input features
        hidden_dim: GRU hidden dimension
        num_layers: Number of GRU layers
        output_dim: Output dimension
        dropout: Dropout rate
        bidirectional: Whether to use bidirectional GRU
    """
    def __init__(self, input_dim=30, hidden_dim=64, num_layers=2,
                 output_dim=1, dropout=0.2, bidirectional=False):
        super(GRUModel, self).__init__()

        if bidirectional:
            import warnings
            warnings.warn("Bidirectional GRU uses future timesteps — only use for classification, not forecasting")

        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Calculate output dimension from GRU
        gru_out_dim = hidden_dim * (2 if bidirectional else 1)
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(gru_out_dim, gru_out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gru_out_dim // 2, output_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            output: (batch, output_dim)
        """
        # GRU forward
        out, _ = self.gru(x)
        
        # Take last timestep
        out = out[:, -1, :]
        
        # FC layers
        output = self.fc(out)
        
        return output


# Factory function for creating models
class SingleStreamLSTM(nn.Module):
    """
    Single-Stream LSTM Ablation Model
    
    Ablation of DualStreamLSTM that concatenates all features
    into a single LSTM stream instead of processing price and
    fundamentals separately. Tests if dual-stream architecture
    provides benefit over simple concatenation.
    
    Args:
        input_dim: Total dimension of all features
        hidden_dim: LSTM hidden dimension
        num_layers: Number of LSTM layers
        output_dim: Output dimension (1 for regression)
        dropout: Dropout rate
    """
    def __init__(self, input_dim=30, hidden_dim=64, num_layers=2,
                 output_dim=1, dropout=0.2):
        super(SingleStreamLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim) concatenated features
        Returns:
            output: (batch, output_dim)
        """
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Last timestep
        output = self.fc(lstm_out)
        return output


class LSTMAblation(nn.Module):
    """
    LSTM with Feature Ablation
    
    Standard LSTM that can exclude specific feature groups
    for ablation studies (e.g., no weather, no WASDE).
    
    Args:
        input_dim: Dimension of included features
        hidden_dim: LSTM hidden dimension
        num_layers: Number of LSTM layers
        output_dim: Output dimension
        dropout: Dropout rate
        ablation_type: String describing what's ablated
    """
    def __init__(self, input_dim=30, hidden_dim=64, num_layers=2,
                 output_dim=1, dropout=0.2, ablation_type='none'):
        super(LSTMAblation, self).__init__()
        
        self.ablation_type = ablation_type
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            output: (batch, output_dim)
        """
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        output = self.fc(lstm_out)
        return output


def create_model(model_name, **kwargs):
    """
    Factory function to create models by name.
    
    Args:
        model_name: One of 'dual_stream_lstm', 'vision_dual_stream_lstm',
                   'resnet', 'transformer', 'tft', 'gru',
                   'single_stream_lstm', 'lstm_ablation'
        **kwargs: Model-specific arguments
    """
    models = {
        'dual_stream_lstm': DualStreamLSTM,
        'vision_dual_stream_lstm': VisionDualStreamLSTM,
        'single_stream_lstm': SingleStreamLSTM,
        'lstm_ablation': LSTMAblation,
        'resnet': ResNet1D,
        'transformer': TransformerModel,
        'patchtst': PatchTSTModel,
        'nbeats': NBeatsModel,
        'tft': TemporalFusionTransformer,
        'gru': GRUModel
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")
    
    return models[model_name](**kwargs)


if __name__ == "__main__":
    # Test model creation and forward pass
    batch_size = 4
    seq_len = 20
    
    print("Testing model architectures...")
    
    # Test Dual-Stream LSTM
    print("\n1. Dual-Stream LSTM")
    model = DualStreamLSTM(price_input_dim=10, fund_input_dim=20)
    price_x = torch.randn(batch_size, seq_len, 10)
    fund_x = torch.randn(batch_size, seq_len, 20)
    out = model(price_x, fund_x)
    print(f"   Input: price={price_x.shape}, fund={fund_x.shape}")
    print(f"   Output: {out.shape}")
    
    # Test ResNet
    print("\n2. ResNet1D")
    model = ResNet1D(input_dim=30, seq_len=seq_len)
    x = torch.randn(batch_size, seq_len, 30)
    out = model(x)
    print(f"   Input: {x.shape}")
    print(f"   Output: {out.shape}")
    
    # Test Transformer
    print("\n3. Transformer")
    model = TransformerModel(input_dim=30)
    x = torch.randn(batch_size, seq_len, 30)
    out = model(x)
    print(f"   Input: {x.shape}")
    print(f"   Output: {out.shape}")
    
    # Test GRU
    print("\n4. GRU")
    model = GRUModel(input_dim=30)
    x = torch.randn(batch_size, seq_len, 30)
    out = model(x)
    print(f"   Input: {x.shape}")
    print(f"   Output: {out.shape}")
    
    # Test Vision-Dual-Stream LSTM (if vision available)
    if VISION_AVAILABLE:
        print("\n5. Vision-Dual-Stream LSTM")
        model = VisionDualStreamLSTM(price_input_dim=10, fund_input_dim=20)
        price_x = torch.randn(batch_size, seq_len, 10)
        fund_x = torch.randn(batch_size, seq_len, 20)
        vision_x = torch.randn(batch_size, seq_len, 6, 1, 64, 64)  # 6 images per timestep
        out = model(price_x, fund_x, vision_x)
        print(f"   Input: price={price_x.shape}, fund={fund_x.shape}, vision={vision_x.shape}")
        print(f"   Output: {out.shape}")
    else:
        print("\n5. Vision-Dual-Stream LSTM (skipped - image_encoder not available)")
    
    print("\nAll tests passed!")
