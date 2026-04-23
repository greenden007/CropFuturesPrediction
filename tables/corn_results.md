# Corn Futures Prediction Results

## Table 1: Test Performance Metrics for Corn Futures Prediction

| Model | Horizon | MSE | MAE | Dir. Acc. | Params |
|:------|:-------:|----:|----:|----------:|-------:|
| Dual-Stream LSTM | H=1 | 0.028709 | 0.1178 | 50.2% | 460,033 |
| Dual-Stream LSTM | H=5 | 0.130910 | 0.2620 | 49.3% | 460,033 |
| Transformer | H=1 | 0.312396 | 0.4423 | 47.0% | 295,169 |
| Transformer | H=5 | 0.201032 | 0.3571 | 52.1% | 295,169 |
| GRU | H=1 | 0.113471 | 0.2733 | 47.0% | 222,593 |
| GRU | H=5 | 0.197604 | 0.3609 | 48.9% | 222,593 |
| ResNet | H=1 | 0.369982 | 0.4788 | 46.0% | 365,889 |
| ResNet | H=5 | 0.449900 | 0.5188 | 49.4% | 365,889 |

## Table 2: Trading Performance for Corn Futures (H=1, Fixed $10k Position)

| Model | Return % | Sharpe | Max DD % | Win Rate | Profit Factor |
|:------|---------:|-------:|---------:|---------:|--------------:|
| Dual-Stream LSTM | 58.73 | 3.850 | -2.43 | 46.2% | 2.10 |
| Transformer | 68.86 | 4.389 | -2.02 | 48.3% | 2.31 |
| GRU | 55.18 | 3.642 | -2.62 | 45.3% | 2.04 |
| ResNet | 61.66 | 4.018 | -1.65 | 46.5% | 2.17 |
