"""
Vision Encoder for CPC Weather Outlook Images

Extracts features from 6-10 day and 8-14 day outlook GIF images
using a lightweight CNN encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class WeatherOutlookEncoder(nn.Module):
    """
    CNN encoder for weather outlook GIF images.
    
    Architecture: Lightweight ResNet-style CNN
    Input: (B, 1, 64, 64) grayscale weather maps
    Output: (B, feature_dim) feature vectors
    """
    
    def __init__(self, feature_dim: int = 64, pretrained: bool = False):
        super(WeatherOutlookEncoder, self).__init__()
        
        self.feature_dim = feature_dim
        
        # Convolutional backbone
        # Input: 1 x 64 x 64 (grayscale weather map)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)  # 32 x 32 x 32
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 64 x 16 x 16
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # 128 x 8 x 8
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # 256 x 4 x 4
        self.bn4 = nn.BatchNorm2d(256)
        
        # Global average pooling + feature projection
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, feature_dim)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, H, W) grayscale weather map images
        
        Returns:
            features: (B, feature_dim)
        """
        # Conv layers with residual-style connections
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Global pooling
        x = self.avgpool(x)  # (B, 256, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 256)
        
        # Feature projection
        x = self.dropout(x)
        x = self.fc(x)  # (B, feature_dim)
        
        return x


class MultiOutlookImageProcessor:
    """
    Processor for handling multiple outlook images per date.
    
    Handles:
    - 6-10 day outlooks (prcp, temp, hghts)
    - 8-14 day outlooks (prcp, temp, hghts)
    """
    
    def __init__(self, 
                 outlook_index_path: Path,
                 data_dir: Path,
                 feature_dim: int = 64,
                 image_size: int = 64):
        
        if not PIL_AVAILABLE:
            raise ImportError("PIL/Pillow is required for image processing. Install with: pip install Pillow")
        
        self.outlook_index_path = outlook_index_path
        self.data_dir = data_dir
        self.feature_dim = feature_dim
        self.image_size = image_size
        
        # Load outlook index
        self.outlook_df = self._load_outlook_index()
        
        # Create encoder
        self.encoder = WeatherOutlookEncoder(feature_dim=feature_dim)
        
    def _load_outlook_index(self):
        """Load the weather outlook index CSV."""
        import pandas as pd
        df = pd.read_csv(self.outlook_index_path)
        df['outlook_date'] = pd.to_datetime(df['outlook_date'])
        return df
    
    def load_image(self, gif_path: Path) -> Optional[np.ndarray]:
        """
        Load and preprocess a GIF image.
        
        Args:
            gif_path: Path to the GIF file
            
        Returns:
            Preprocessed image as (1, H, W) numpy array, or None if failed
        """
        try:
            if not gif_path.exists():
                return None
            
            # Load GIF (take first frame if animated)
            img = Image.open(gif_path)
            if hasattr(img, 'n_frames') and img.n_frames > 1:
                img.seek(0)  # Get first frame
            
            # Convert to grayscale and resize
            img = img.convert('L')  # Grayscale
            img = img.resize((self.image_size, self.image_size))
            
            # Normalize to [0, 1]
            img_array = np.array(img).astype(np.float32) / 255.0
            
            return img_array
            
        except Exception as e:
            print(f"Error loading image {gif_path}: {e}")
            return None
    
    def get_images_for_date(self, date: str) -> Dict[str, List[np.ndarray]]:
        """
        Get all outlook images for a specific date.
        
        Args:
            date: Date string (YYYY-MM-DD)
            
        Returns:
            Dictionary mapping outlook types to lists of images
        """
        import pandas as pd
        
        date_dt = pd.to_datetime(date)
        day_outlooks = self.outlook_df[self.outlook_df['outlook_date'] == date_dt]
        
        images = {
            '6-10_day': [],
            '8-14_day': []
        }
        
        for _, row in day_outlooks.iterrows():
            gif_path = self.data_dir / row['full_path']
            img = self.load_image(gif_path)
            
            if img is not None:
                forecast_type = row['forecast_type']
                if forecast_type in images:
                    images[forecast_type].append(img)
        
        return images
    
    def extract_features_for_date(self, date: str, device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """
        Extract features from all outlook images for a date.
        
        Args:
            date: Date string (YYYY-MM-DD)
            device: Device to run encoder on
            
        Returns:
            Dictionary mapping outlook types to feature tensors
        """
        self.encoder.to(device)
        self.encoder.eval()
        
        images_dict = self.get_images_for_date(date)
        features = {}
        
        with torch.no_grad():
            for outlook_type, images in images_dict.items():
                if len(images) == 0:
                    # No images available - return zeros
                    features[outlook_type] = torch.zeros(self.feature_dim, device=device)
                    continue
                
                # Stack images into batch
                batch = np.stack(images)  # (N, H, W)
                batch = torch.FloatTensor(batch).unsqueeze(1)  # (N, 1, H, W)
                batch = batch.to(device)
                
                # Extract features
                feat = self.encoder(batch)  # (N, feature_dim)
                
                # Aggregate features (mean pooling)
                features[outlook_type] = feat.mean(dim=0)  # (feature_dim,)
        
        return features
    
    def process_all_dates(self, dates: List[str], device: str = 'cpu') -> np.ndarray:
        """
        Process all outlook images for a list of dates.
        
        Args:
            dates: List of date strings (YYYY-MM-DD)
            device: Device to run encoder on
            
        Returns:
            Array of shape (len(dates), feature_dim * 2) with features for both outlook types
        """
        all_features = []
        
        for date in dates:
            feat_dict = self.extract_features_for_date(date, device)
            
            # Concatenate 6-10 day and 8-14 day features
            combined = torch.cat([
                feat_dict.get('6-10_day', torch.zeros(self.feature_dim)),
                feat_dict.get('8-14_day', torch.zeros(self.feature_dim))
            ], dim=0)
            
            all_features.append(combined.cpu().numpy())
        
        return np.array(all_features)  # (N, feature_dim * 2)


def extract_image_features_to_csv(outlook_index_path: Path,
                                   data_dir: Path,
                                   output_path: Path,
                                   feature_dim: int = 64,
                                   batch_size: int = 32):
    """
    Pre-extract image features for all dates and save to CSV.
    
    This is useful for pre-computing features to avoid loading images during training.
    """
    import pandas as pd
    from tqdm import tqdm
    
    if not PIL_AVAILABLE:
        print("Warning: PIL not available. Cannot extract image features.")
        return
    
    processor = MultiOutlookImageProcessor(
        outlook_index_path=outlook_index_path,
        data_dir=data_dir,
        feature_dim=feature_dim
    )
    
    # Get unique dates
    unique_dates = processor.outlook_df['outlook_date'].unique()
    unique_dates = sorted([pd.to_datetime(d).strftime('%Y-%m-%d') for d in unique_dates])
    
    print(f"Processing {len(unique_dates)} dates...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    all_features = []
    all_dates = []
    
    for i in tqdm(range(0, len(unique_dates), batch_size)):
        batch_dates = unique_dates[i:i + batch_size]
        batch_features = processor.process_all_dates(batch_dates, device=device)
        
        all_features.extend(batch_features)
        all_dates.extend(batch_dates)
    
    # Create dataframe
    feature_cols = [f'img_feat_{i}' for i in range(feature_dim * 2)]
    df = pd.DataFrame(all_features, columns=feature_cols)
    df['date'] = all_dates
    df['date'] = pd.to_datetime(df['date'])
    
    # Save
    df.to_csv(output_path, index=False)
    print(f"Saved image features to {output_path}")


if __name__ == "__main__":
    # Test the encoder
    print("Testing WeatherOutlookEncoder...")
    
    encoder = WeatherOutlookEncoder(feature_dim=64)
    
    # Test with random input
    test_input = torch.randn(2, 1, 64, 64)
    output = encoder(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: (2, 64)")
    
    # Count parameters
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"Total parameters: {n_params:,}")
