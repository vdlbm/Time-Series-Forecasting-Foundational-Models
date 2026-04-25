# Time Series Forecasting: Foundation Models vs LLMs



Comparative analysis of foundational time series models and general-purpose LLMs for zero-shot financial time series forecasting. Evaluates 11 models across 7 datasets using expanding window backtesting, statistical significance tests, and financial performance metrics.

## Models (11)

### Baseline
| Model | Type | Description |
|-------|------|-------------|
| **Naive (Random Walk)** | Baseline | Predicts last observed value — canonical financial benchmark |

### Classical Statistical (4)
| Model | Type | Description |
|-------|------|-------------|
| **AutoARIMA** | Statistical | Auto-tuned ARIMA with seasonal detection |
| **Simple Exp. Smoothing** | Statistical | AutoETS (ANN) — level only |
| **Holt** | Statistical | AutoETS (AAN) — level + trend |
| **Holt-Winters** | Statistical | AutoETS (AAA) — level + trend + seasonality |

### Foundation Models — Time Series Specific (4)
| Model | Type | Description |
|-------|------|-------------|
| **TimeGPT** (Nixtla) | Foundation (API) | Cloud-based, zero-shot |
| **Moirai** (Salesforce) | Foundation (local) | Uni2TS, small variant, 100-sample ensemble |
| **Chronos** (Amazon) | Foundation (local) | T5-small, 20-sample ensemble |
| **TimesFM 2.0** (Google) | Foundation (local) | 500M params, PyTorch backend, context up to 2048 |

### General-Purpose LLMs — Zero-Shot (2)
| Model | Type | Description |
|-------|------|-------------|
| **Gemma 4 E4B** (Google) | LLM local (GGUF Q8_0) | MoE architecture, ~4B active params per token |
| **Qwen3-8B** (Alibaba) | LLM local (GGUF Q8_0) | Dense 8B, thinking mode disabled for determinism |

## Datasets (7)

| Dataset | Frequency | Period | Market Type |
|---------|-----------|--------|-------------|
| S&P 500 Monthly | Monthly | 2022–2025 | Equity index |
| S&P 500 Daily | Business days (Mon-Fri) | 2022–2025 | Equity index |
| EUR/USD Monthly | Monthly | 2022–2025 | FX |
| EUR/USD Daily | Business days (Mon-Fri) | 2022–2025 | FX |
| BTC/USD Monthly | Monthly | 2022–2025 | Crypto |
| BTC/USD Daily | Calendar days (7d/week) | 2022–2025 | Crypto |
| US CPI Monthly | Monthly | 2022–2025 | Macro |

## Methodology

- **Backtesting**: Date-based expanding window (train grows each fold)
- **Horizons**: h=1 and h=3 (monthly), h=1 and h=5/7 (daily — adjusted per market)
- **Folds**: ~36 monthly folds, ~156 weekly-stepped daily folds
- **Metrics**: MSE, RMSE, MAE, MAPE, Directional Accuracy, Strategy Return
- **Financial Metrics**: Sharpe Ratio, Max Drawdown, Calmar Ratio
- **Statistical Tests**: ADF/KPSS stationarity, Diebold-Mariano (with Bonferroni), Friedman ranking + Nemenyi CD diagram
- **Market Regimes**: HMM-based bull/bear detection, metrics segmented by regime

## Project Structure

```
├── main.py                     # Experiment runner
├── config/experiments.yaml     # Full experiment configuration
├── src/
│   ├── models/
│   │   ├── base.py             # BaseForecaster ABC
│   │   ├── naive.py            # Random Walk baseline
│   │   ├── classical.py        # ARIMA, ETS wrappers
│   │   ├── foundation.py       # TimeGPT, Moirai, Chronos, TimesFM
│   │   ├── llm.py              # Model-agnostic GGUF LLM wrapper
│   │   └── factory.py          # Factory pattern for model instantiation
│   ├── data/
│   │   ├── loader.py           # CSV ingestion and preprocessing
│   │   └── adapters.py         # Numerical-to-text encoding for LLMs
│   ├── evaluation/
│   │   ├── splitter.py         # ExpandingWindowSplitter (date-based)
│   │   └── metrics.py          # Technical + financial metrics
│   └── analysis/
│       ├── statistical_tests.py # ADF, KPSS, Diebold-Mariano
│       ├── market_regimes.py    # HMM bull/bear detection
│       └── rankings.py          # Friedman + Nemenyi CD diagram
├── tests/                       # pytest test suite
├── DATA/                        # CSV datasets
├── results/                     # Output CSVs from experiments
└── logs/                        # Experiment logs
```

## Setup 

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download GGUF models (place in models/ directory)
# - gemma-3-12b-it-Q4_K_M.gguf (~7.3GB)
# - phi-4-Q4_K_M.gguf (~8.4GB)

# 3. Set TimeGPT API key
echo "NIXTLA_API_KEY=your_key_here" > .env

# 4. Run experiment
python main.py
```

