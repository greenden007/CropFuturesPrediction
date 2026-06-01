import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class ClimateTextEncoder(nn.Module):
    """Encodes USDA/NOAA text data using a HuggingFace Transformer."""
    def __init__(self, model_name='distilbert-base-uncased', hidden_dim=128):
        super().__init__()
        # We use a smaller model by default or let it be passed in.
        # In a real environment, you'd use 'climatebert/distilroberta-base-climate-f'
        try:
            self.bert = AutoModel.from_pretrained(model_name)
        except Exception:
            # Fallback for dry runs or when offline
            config = AutoConfig.from_pretrained('distilbert-base-uncased')
            self.bert = AutoModel.from_config(config)
            
        # Freeze initial layers to avoid overfitting on limited financial datasets
        for param in self.bert.parameters():
            param.requires_grad = False
            
        self.fc = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # For DistilBERT, we take the first token of the last hidden state
        cls_token = outputs.last_hidden_state[:, 0, :]
        return self.fc(cls_token)
