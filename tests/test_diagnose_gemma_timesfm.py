"""
Diagnostic Tests — gemma-4 GGUF + TimesFM.

Purpose:
    1. Diagnose WHY gemma-4 GGUF fails to load (corrupt file, bad version,
       incomplete download, etc.) and verify inference if loading succeeds.
    2. Verify TimesFM (Google DeepMind) loads and predicts correctly.

Run:
    pytest tests/test_diagnose_gemma_timesfm.py -v -s
"""

import os
import struct
import pytest
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
GEMMA_PATH = "models/gemma-4-E4B-it-Q8_0.gguf"
DATA_PATH = "DATA/SP500_Monthly.csv"
NUM_TRAIN_POINTS = 60

# GGUF magic number: "GGUF"
GGUF_MAGIC = 0x46554747
# Minimum sane size for a Q8_0 quantized ~4B param model (~4 GB)
MIN_GGUF_SIZE_BYTES = 3_000_000_000


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------
def _load_sp500_train() -> pd.DataFrame:
    from src.data.loader import TimeSeriesLoader

    loader = TimeSeriesLoader(
        file_path=DATA_PATH, time_col="Date", target_col="Close"
    )
    loaded = loader.load()
    df = loaded.df.reset_index()
    df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)
    df = df.sort_values("ds").tail(NUM_TRAIN_POINTS).reset_index(drop=True)
    return df[["ds", "y"]]


def _validate_forecast(forecast_df: pd.DataFrame, horizon: int, label: str):
    assert isinstance(forecast_df, pd.DataFrame), f"[{label}] Not a DataFrame"
    assert "y_pred" in forecast_df.columns, f"[{label}] Missing y_pred"
    assert len(forecast_df) == horizon, f"[{label}] Expected {horizon} rows, got {len(forecast_df)}"
    preds = forecast_df["y_pred"].values
    assert np.all(np.isfinite(preds)), f"[{label}] Non-finite preds: {preds}"
    assert np.all(preds > 0), f"[{label}] Preds should be > 0: {preds}"
    print(f"  ✅ {label} h={horizon}: {preds}")


@pytest.fixture(scope="module")
def sp500_train():
    return _load_sp500_train()


# ===========================================================================
# PART 1: gemma-4 GGUF DIAGNOSTICS
# ===========================================================================

@pytest.mark.skipif(
    not os.path.exists(GEMMA_PATH),
    reason=f"GGUF file not found: {GEMMA_PATH}",
)
class TestGemmaDiagnostics:
    """Step-by-step diagnostics to pinpoint the gemma-4 loading failure."""

    def test_step1_file_size_is_sane(self):
        """A Q8_0 quantized ~4B model should be at least ~3 GB.
        If much smaller, the download was likely incomplete."""
        size = os.path.getsize(GEMMA_PATH)
        size_gb = size / (1024**3)
        print(f"\n  �� File size: {size:,} bytes ({size_gb:.2f} GB)")
        assert size > MIN_GGUF_SIZE_BYTES, (
            f"File is too small ({size_gb:.2f} GB). "
            f"Expected ≥ {MIN_GGUF_SIZE_BYTES / 1024**3:.1f} GB. "
            "The download is likely incomplete — re-download the GGUF."
        )

    def test_step2_gguf_magic_number(self):
        """The first 4 bytes of a valid GGUF file must be 0x46475547 ('GGUF').
        If not, the file is corrupt or not a GGUF at all."""
        with open(GEMMA_PATH, "rb") as f:
            raw = f.read(4)
        magic = struct.unpack("<I", raw)[0]
        print(f"\n  �� Magic bytes: {raw.hex()} (expected: 47475546)")
        assert magic == GGUF_MAGIC, (
            f"Invalid GGUF magic: got 0x{magic:08X}, expected 0x{GGUF_MAGIC:08X}. "
            "File is either corrupt or not a GGUF. Re-download it."
        )

    def test_step3_gguf_version(self):
        """GGUF v3 is the current standard. v1/v2 are deprecated.
        Reads bytes 4-7 as little-endian uint32 for the version."""
        with open(GEMMA_PATH, "rb") as f:
            f.read(4)  # skip magic
            raw = f.read(4)
        version = struct.unpack("<I", raw)[0]
        print(f"\n  �� GGUF version: {version}")
        assert version in (2, 3), (
            f"Unexpected GGUF version {version}. "
            "Your llama-cpp-python may not support this version. "
            "Try: pip install --upgrade llama-cpp-python"
        )

    def test_step4_llama_cpp_version(self):
        """Print the installed llama-cpp-python version for debugging."""
        import llama_cpp
        version = getattr(llama_cpp, "__version__", "unknown")
        print(f"\n  �� llama-cpp-python version: {version}")
        # gemma-4 GGUF support was added around v0.3.x
        # If version is too old, it won't recognize the architecture
        assert version != "unknown", "Cannot determine llama-cpp-python version"
        print("  ℹ️  If loading still fails, try: pip install --upgrade llama-cpp-python")

    def test_step5_load_model(self):
        """Attempt to actually load the Gemma GGUF.
        If previous diagnostics passed, this should work."""
        from llama_cpp import Llama

        print("\n  ⏳ Attempting to load gemma-4 GGUF...")
        model = Llama(
            model_path=GEMMA_PATH,
            n_gpu_layers=-1,
            n_ctx=512,       # Small context for quick loading test
            verbose=True,    # Show llama.cpp logs to help debug
        )
        assert model is not None, "Llama() returned None"
        print("  ✅ gemma-4 loaded successfully!")
        del model

    def test_step6_inference(self, sp500_train):
        """If loading works, test a single inference."""
        from src.models.llm import LocalLLMWrapper

        config = {
            "model_path": GEMMA_PATH,
            "model_name": "Gemma4-E4B",
            "context_window_size": 4096,
            "llm_window_size": 40,
            "max_new_tokens": 20,
            "disable_thinking": False,
            "dataset_description": "Monthly closing price of the S&P 500 index.",
        }
        model = LocalLLMWrapper(config)
        model.fit(sp500_train)

        forecast = model.predict(horizon=1)
        _validate_forecast(forecast, 1, "Gemma4")

        forecast3 = model.predict(horizon=3)
        _validate_forecast(forecast3, 3, "Gemma4")
        del model


# ===========================================================================
# PART 2: TIMESFM (Google DeepMind)
# ===========================================================================

# Check if timesfm is importable
try:
    import timesfm as _tfm_check
    TIMESFM_AVAILABLE = True
except ImportError:
    TIMESFM_AVAILABLE = False

try:
    import torch as _torch_check
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@pytest.mark.skipif(
    not TIMESFM_AVAILABLE,
    reason="timesfm not installed (pip install timesfm)",
)
@pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="torch not installed (pip install torch)",
)
class TestTimesFMSmoke:
    """Verify TimesFM loads, fits, and predicts with real SP500 data."""

    @pytest.fixture(scope="class")
    def timesfm_model(self):
        from src.models.foundation import FoundationWrapper

        config = {
            "model_name": "TimesFM",
            "type": "foundation",
            "model_path": "google/timesfm-2.0-500m-pytorch",
            "freq": "MS",
            "context_length": 2048,
        }
        print("\n  ⏳ Loading TimesFM (first run downloads ~2 GB from HuggingFace)...")
        model = FoundationWrapper(config)
        print("  ✅ TimesFM loaded.")
        return model

    @pytest.mark.parametrize("horizon", [1, 3])
    def test_predict_returns_valid_forecast(self, timesfm_model, sp500_train, horizon):
        timesfm_model.fit(sp500_train)
        forecast = timesfm_model.predict(horizon=horizon)
        _validate_forecast(forecast, horizon, "TimesFM")

    def test_predict_output_dtype_is_float(self, timesfm_model, sp500_train):
        """TimesFM returns numpy float32 — ensure no type mismatch."""
        timesfm_model.fit(sp500_train)
        forecast = timesfm_model.predict(horizon=1)
        val = forecast["y_pred"].iloc[0]
        assert isinstance(val, (float, np.floating)), (
            f"Expected float, got {type(val)}: {val}"
        )
        print(f"  ✅ TimesFM dtype OK: {type(val).__name__} = {val:.2f}")
