from dataclasses import dataclass
import math

import mne
import numpy as np


@dataclass(frozen=True)
class PreprocessConfig:
    l_freq: float = 1.0
    h_freq: float = 40.0
    target_sfreq: float = 128.0
    line_freq: float | None = None
    peak_to_peak_limit_uv: float = 500.0


DEFAULT_PREPROCESS = PreprocessConfig()


def _resolve_line_freq(raw: mne.io.BaseRaw, cfg: PreprocessConfig) -> float | None:
    if cfg.line_freq is not None:
        return float(cfg.line_freq)

    metadata_value = raw.info.get("line_freq")
    if isinstance(metadata_value, (int, float)) and math.isfinite(float(metadata_value)):
        return float(metadata_value)

    return None


def preprocess_raw(
    raw: mne.io.BaseRaw,
    cfg: PreprocessConfig = DEFAULT_PREPROCESS,
) -> mne.io.BaseRaw:
    cleaned = raw.copy().pick("eeg").load_data()
    sfreq = float(cleaned.info["sfreq"])

    line_freq = _resolve_line_freq(cleaned, cfg)
    if line_freq is not None and 0.0 < line_freq < sfreq / 2.0:
        cleaned.notch_filter(freqs=[line_freq], verbose="ERROR")

    h_freq = min(cfg.h_freq, sfreq / 2.0 - 0.5)
    cleaned.filter(l_freq=cfg.l_freq, h_freq=h_freq, verbose="ERROR")
    cleaned.set_eeg_reference("average", projection=False, verbose="ERROR")

    if abs(sfreq - cfg.target_sfreq) > 1e-6:
        cleaned.resample(cfg.target_sfreq, verbose="ERROR")

    return cleaned


def normalize_trial(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(axis=-1, keepdims=True)
    std = np.maximum(x.std(axis=-1, keepdims=True), 1e-6)
    return (x - mean) / std


def is_acceptable_trial(
    x_volts: np.ndarray,
    cfg: PreprocessConfig = DEFAULT_PREPROCESS,
) -> bool:
    if not np.isfinite(x_volts).all():
        return False

    ptp_uv = np.ptp(x_volts, axis=-1) * 1e6
    return bool(np.all(ptp_uv <= cfg.peak_to_peak_limit_uv))
