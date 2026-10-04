from dataclasses import dataclass
import numpy as np
import mne


@dataclass(frozen=True)
class PreprocessConfig:
    l_freq: float = 1.0
    h_freq: float = 40.0
    target_sfreq: float = 128.0
    line_freq: float = 50.0
    peak_to_peak_limit_uv: float = 500.0


DEFAULT_PREPROCESS = PreprocessConfig()


def preprocess_raw(raw: mne.io.BaseRaw, cfg: PreprocessConfig = DEFAULT_PREPROCESS) -> mne.io.BaseRaw:
    cleaned = raw.copy().pick("eeg").load_data()
    sfreq = float(cleaned.info["sfreq"])
    if cfg.line_freq < sfreq / 2.0:
        cleaned.notch_filter(freqs=[cfg.line_freq], verbose="ERROR")
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


def is_acceptable_trial(x_volts: np.ndarray, cfg: PreprocessConfig = DEFAULT_PREPROCESS) -> bool:
    if not np.isfinite(x_volts).all():
        return False
    ptp_uv = np.ptp(x_volts, axis=-1) * 1e6
    return bool(np.all(ptp_uv <= cfg.peak_to_peak_limit_uv))
