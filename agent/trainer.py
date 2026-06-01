import torch
import torch.nn as nn
from torch.optim import AdamW
import time

class ModelTrainer:
    """Handles the PyTorch optimization loop for the Multimodal Neural Network."""
    
    def __init__(self, model, train_loader, lr=1e-4):
        self.model = model
        self.train_loader = train_loader
        
        # We use Mean Squared Error because we are regressing Continuous Target Returns
        self.criterion = nn.MSELoss()
        
        # AdamW adds advanced weight decay (L2 regularization) preventing overfitting 
        # on noisy financial data
        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        
        start_time = time.time()
        for batch in self.train_loader:
            self.optimizer.zero_grad()
            
            # 1. Forward Pass
            out = self.model(
                batch['price_seq'], 
                batch['climate_seq'], 
                batch['input_ids'], 
                batch['attention_mask']
            )
            
            # 2. Compute Loss
            loss = self.criterion(out, batch['target'])
            
            # 3. Backpropagation (Calculate gradients)
            loss.backward()
            
            # 4. Gradient Clipping (crucial for LSTMs traversing long temporal sequences)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 5. Update weights
            self.optimizer.step()
            
            total_loss += loss.item()
            
        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(self.train_loader)
        
        return avg_loss, epoch_time
        
    def fit(self, epochs=5):
        print(f"\n[*] Booting GPU/CPU Trainer...")
        print(f"[*] Commencing Optimization over {epochs} Epochs!")
        print("-" * 50)
        
        for epoch in range(1, epochs + 1):
            avg_loss, epoch_time = self.train_epoch()
            print(f"    Epoch {epoch:02d}/{epochs} | MSE Loss: {avg_loss:.6f} | Execution Time: {epoch_time:.2f}s")
            
        print("-" * 50)
        print("[*] Optimization Finished! Neuromorphic pathways updated.")
        return self.model
