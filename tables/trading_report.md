# Trading Simulation Results

## Summary Statistics

**Total Models Evaluated:** 57

**Models:** dual_stream_lstm, gru, nbeats, patchtst, resnet, tft, transformer

**Commodities:** corn, soybeans, wheat

**Horizons:** [np.int64(1), np.int64(5), np.int64(10)]

## Top 10 Overall by Sharpe Ratio

| Model | Commodity | H | Return % | Sharpe | Max DD % | Win Rate |
|---|---|---|---|---|---|---|
| dual_stream_lstm | wheat | 5 | 80.09 | 5.264 | -2.21 | 50.46 |
| gru | wheat | 5 | 80.09 | 5.264 | -2.21 | 50.46 |
| nbeats | wheat | 5 | 80.09 | 5.264 | -2.21 | 50.46 |
| patchtst | wheat | 5 | 80.09 | 5.264 | -2.21 | 50.46 |
| resnet | wheat | 5 | 80.09 | 5.264 | -2.21 | 50.46 |
| tft | wheat | 5 | 80.09 | 5.264 | -2.21 | 50.46 |
| transformer | wheat | 5 | 80.09 | 5.264 | -2.21 | 50.46 |
| dual_stream_lstm | wheat | 1 | 80.57 | 5.263 | -2.2 | 50.46 |
| gru | wheat | 1 | 80.57 | 5.263 | -2.2 | 50.46 |
| nbeats | wheat | 1 | 80.57 | 5.263 | -2.2 | 50.46 |

## Corn Results

| Model | H | Return % | Sharpe | Max DD % | Win Rate | Profit Factor |
|---|---|---|---|---|---|---|
| patchtst | 5 | 78.42 | 5.112 | -1.44 | 50.39 | 2.54 |
| dual_stream_lstm | 5 | 78.42 | 5.112 | -1.44 | 50.39 | 2.54 |
| transformer | 5 | 78.42 | 5.112 | -1.44 | 50.39 | 2.54 |
| gru | 5 | 78.42 | 5.112 | -1.44 | 50.39 | 2.54 |
| tft | 5 | 78.42 | 5.112 | -1.44 | 50.39 | 2.54 |
| nbeats | 5 | 78.42 | 5.112 | -1.44 | 50.39 | 2.54 |
| resnet | 5 | 78.42 | 5.112 | -1.44 | 50.39 | 2.54 |
| tft | 10 | 77.71 | 5.09 | -1.45 | 50.31 | 2.53 |
| resnet | 10 | 77.71 | 5.09 | -1.45 | 50.31 | 2.53 |
| transformer | 10 | 77.71 | 5.09 | -1.45 | 50.31 | 2.53 |
| gru | 10 | 77.71 | 5.09 | -1.45 | 50.31 | 2.53 |
| dual_stream_lstm | 10 | 77.71 | 5.09 | -1.45 | 50.31 | 2.53 |
| resnet | 1 | 78.19 | 5.072 | -1.44 | 50.31 | 2.53 |
| patchtst | 1 | 78.19 | 5.072 | -1.44 | 50.31 | 2.53 |
| nbeats | 1 | 78.19 | 5.072 | -1.44 | 50.31 | 2.53 |
| tft | 1 | 78.19 | 5.072 | -1.44 | 50.31 | 2.53 |
| gru | 1 | 78.19 | 5.072 | -1.44 | 50.31 | 2.53 |
| transformer | 1 | 78.19 | 5.072 | -1.44 | 50.31 | 2.53 |
| dual_stream_lstm | 1 | 78.19 | 5.072 | -1.44 | 50.31 | 2.53 |

**Corn Stats:**
- Mean Return: 78.15%
- Mean Sharpe: 5.091
- Best Model: patchtst (h5)

## Soybeans Results

| Model | H | Return % | Sharpe | Max DD % | Win Rate | Profit Factor |
|---|---|---|---|---|---|---|
| patchtst | 5 | 56.67 | 3.996 | -1.78 | 47.47 | 2.26 |
| dual_stream_lstm | 5 | 56.67 | 3.996 | -1.78 | 47.47 | 2.26 |
| transformer | 5 | 56.67 | 3.996 | -1.78 | 47.47 | 2.26 |
| gru | 5 | 56.67 | 3.996 | -1.78 | 47.47 | 2.26 |
| tft | 5 | 56.67 | 3.996 | -1.78 | 47.47 | 2.26 |
| nbeats | 5 | 56.67 | 3.996 | -1.78 | 47.47 | 2.26 |
| resnet | 5 | 56.67 | 3.996 | -1.78 | 47.47 | 2.26 |
| tft | 10 | 56.19 | 3.99 | -1.79 | 47.45 | 2.26 |
| resnet | 10 | 56.19 | 3.99 | -1.79 | 47.45 | 2.26 |
| transformer | 10 | 56.19 | 3.99 | -1.79 | 47.45 | 2.26 |
| gru | 10 | 56.19 | 3.99 | -1.79 | 47.45 | 2.26 |
| dual_stream_lstm | 10 | 56.19 | 3.99 | -1.79 | 47.45 | 2.26 |
| resnet | 1 | 56.68 | 3.971 | -1.78 | 47.4 | 2.25 |
| patchtst | 1 | 56.68 | 3.971 | -1.78 | 47.4 | 2.25 |
| nbeats | 1 | 56.68 | 3.971 | -1.78 | 47.4 | 2.25 |
| tft | 1 | 56.68 | 3.971 | -1.78 | 47.4 | 2.25 |
| gru | 1 | 56.68 | 3.971 | -1.78 | 47.4 | 2.25 |
| transformer | 1 | 56.68 | 3.971 | -1.78 | 47.4 | 2.25 |
| dual_stream_lstm | 1 | 56.68 | 3.971 | -1.78 | 47.4 | 2.25 |

**Soybeans Stats:**
- Mean Return: 56.55%
- Mean Sharpe: 3.985
- Best Model: patchtst (h5)

## Wheat Results

| Model | H | Return % | Sharpe | Max DD % | Win Rate | Profit Factor |
|---|---|---|---|---|---|---|
| patchtst | 5 | 80.09 | 5.264 | -2.21 | 50.46 | 2.55 |
| transformer | 5 | 80.09 | 5.264 | -2.21 | 50.46 | 2.55 |
| gru | 5 | 80.09 | 5.264 | -2.21 | 50.46 | 2.55 |
| tft | 5 | 80.09 | 5.264 | -2.21 | 50.46 | 2.55 |
| nbeats | 5 | 80.09 | 5.264 | -2.21 | 50.46 | 2.55 |
| resnet | 5 | 80.09 | 5.264 | -2.21 | 50.46 | 2.55 |
| dual_stream_lstm | 5 | 80.09 | 5.264 | -2.21 | 50.46 | 2.55 |
| resnet | 1 | 80.57 | 5.263 | -2.2 | 50.46 | 2.55 |
| transformer | 1 | 80.57 | 5.263 | -2.2 | 50.46 | 2.55 |
| tft | 1 | 80.57 | 5.263 | -2.2 | 50.46 | 2.55 |
| dual_stream_lstm | 1 | 80.57 | 5.263 | -2.2 | 50.46 | 2.55 |
| patchtst | 1 | 80.57 | 5.263 | -2.2 | 50.46 | 2.55 |
| nbeats | 1 | 80.57 | 5.263 | -2.2 | 50.46 | 2.55 |
| gru | 1 | 80.57 | 5.263 | -2.2 | 50.46 | 2.55 |
| resnet | 10 | 78.91 | 5.211 | -2.23 | 50.31 | 2.53 |
| gru | 10 | 78.91 | 5.211 | -2.23 | 50.31 | 2.53 |
| tft | 10 | 78.91 | 5.211 | -2.23 | 50.31 | 2.53 |
| dual_stream_lstm | 10 | 78.91 | 5.211 | -2.23 | 50.31 | 2.53 |
| transformer | 10 | 78.91 | 5.211 | -2.23 | 50.31 | 2.53 |

**Wheat Stats:**
- Mean Return: 79.96%
- Mean Sharpe: 5.250
- Best Model: patchtst (h5)

