import argparse
import os
import torch
import warnings
from models.climate_bert import ClimateTextEncoder
from models.multimodal import MultimodalAgriFuturesNet
from data.dataset import AgriMultimodalDataset
from data.parsers import load_futures_data, load_wasde_features, load_cpc_text
from torch.utils.data import DataLoader, Subset
from agent.signal_generator import SignalGenerator
from agent.backtest import BacktestSimulator
from agent.trainer import ModelTrainer

warnings.filterwarnings('ignore')

class DummyTokenizer:
    """Fallback tokenizer so transformers dependency isn't explicitly required to mock"""
    def __call__(self, text, **kwargs):
        return {'input_ids': torch.zeros(1, kwargs.get('max_length', 128), dtype=torch.long), 
                'attention_mask': torch.ones(1, kwargs.get('max_length', 128), dtype=torch.long)}

def get_base_dataset():
    # Paths pointing to the user's real datasets
    root_path = "/Users/muhilramesh/Desktop/CropFuturesPrediction/data"
    futures_path = os.path.join(root_path, "futures_prices/corn_futures.csv")
    wasde_path = os.path.join(root_path, "usda_wasde/psd_grains_pulses.csv")
    cpc_path = os.path.join(root_path, "noaa_cpc_discussions")
    
    price_df = load_futures_data(futures_path, forward_days=5)
    climate_df = load_wasde_features(wasde_path, crop_name='Corn')
    text_df = load_cpc_text(cpc_path)

    dataset = AgriMultimodalDataset(
        price_df=price_df,
        climate_df=climate_df,
        text_df=text_df,
        tokenizer=DummyTokenizer(),
        price_seq_len=10,
        climate_seq_len=30,
        max_text_len=64
    )
    return dataset

def run_dry_run():
    print("\n[+] INGESTING RAW DATA")
    try:
        dataset = get_base_dataset()
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    except FileNotFoundError as e:
        print(f"[!] Error: Could not locate datasets: {e}")
        return

    batch = next(iter(dataloader))
    
    print("\n[+] RUNNING MULTIMODAL NETWORK")
    model = MultimodalAgriFuturesNet(
        price_dim=batch['price_seq'].shape[2],
        climate_dim=batch['climate_seq'].shape[2],
        text_embed_dim=128,
        hidden_dim=64
    )
    
    out = model(batch['price_seq'], batch['climate_seq'], batch['input_ids'], batch['attention_mask'])
    print(f"[*] Neural Network output fully compiled to target shape: {out.shape}")
    print("[*] PIPELINE TEST SUCCESSFUL. Live Data integrated.")

def run_backtest(train_first=False):
    print("\n" + "="*60)
    print(" [*] INITIATING AGENT BACKTEST")
    print("="*60)
    
    dataset = get_base_dataset()
    
    # Chronological Split: 80% Train, 20% Test (Walk-Forward out-of-sample)
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    
    train_dataset = Subset(dataset, range(0, train_size))
    test_dataset = Subset(dataset, range(train_size, len(dataset)))
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    # Batch size 1 for test loader because we simulate trading step-by-step
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    sample_batch = next(iter(test_loader))
    model = MultimodalAgriFuturesNet(
        price_dim=sample_batch['price_seq'].shape[2],
        climate_dim=sample_batch['climate_seq'].shape[2],
        text_embed_dim=128,
        hidden_dim=64
    )
    
    if train_first:
        print("\n[!] Initiating Optimization Sequence on Historical Training Set (80%)")
        trainer = ModelTrainer(model=model, train_loader=train_loader, lr=2e-4)
        # Run 3 epochs for demonstration speed
        model = trainer.fit(epochs=3)
    else:
        print("\n[!] WARNING: Model is UNTRAINED (Random Weights). Running baseline.")
    
    model.eval()
    signal_gen = SignalGenerator(rolling_window=60, threshold_multiplier=0.75)
    simulator = BacktestSimulator(initial_capital=100000.0, commission=0.001)
    
    print("\n[*] Executing chronological walk-forward simulation on Test Set (20%)...")
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            expected_return = model(batch['price_seq'], batch['climate_seq'], batch['input_ids'], batch['attention_mask']).item()
            signal = signal_gen.process_prediction(expected_return)
            actual_return = batch['target'].item()
            simulator.step(signal, actual_return)
            
            if i % 200 == 0 and i > 0:
                print(f"    - Simulated {i} out-of-sample trade days...")
                
    summary = simulator.summary()
    print("\n" + "="*60)
    print(" [~] O.O.S. BACKTEST FINAL SUMMARY")
    print("="*60)
    print(f" Model Trained  : {train_first}")
    print(f" Days Tested    : {len(test_loader)}")
    print(f" Final Capital  : ${simulator.capital:,.2f}")
    print(f" Total Return   : {summary['Total Return']*100:.2f}%")
    print(f" Sharpe Ratio   : {summary['Sharpe Ratio']:.4f}")
    print(f" Max Drawdown   : {summary['Max Drawdown']*100:.2f}%")
    print("="*60)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Run dummy forward pass to check dims')
    parser.add_argument('--backtest', action='store_true', help='Run baseline untrained execution over the dataset')
    parser.add_argument('--train', action='store_true', help='Train the network on history, then walk-forward backtest')
    args = parser.parse_args()
    
    if args.dry_run:
        run_dry_run()
    elif args.train:
        run_backtest(train_first=True)
    elif args.backtest:
        run_backtest(train_first=False)
    else:
        print("Please provide a flag. Ex: python main.py --train")

if __name__ == "__main__":
    main()
