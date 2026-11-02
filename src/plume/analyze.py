"""plume.analyze — WAV feature extraction (SPEC section 10).

Everything is computed with numpy/scipy only. Tempo comes from the
autocorrelation of a spectral-flux onset envelope (60-200 bpm search);
loudness is a K-weighted RMS approximation of LUFS (ITU-R BS.1770 filter
coefficients, tabulated for 48 kHz — approximate at other rates).
"""
from __future__ import annotations

import numpy as np
import soundfile as sf
from scipy.signal import find_peaks, lfilter, welch

__all__ = ["analyze"]

_N_FFT = 1024
_HOP = 512

# ITU-R BS.1770 K-weighting biquads @ 48 kHz (shelf, then RLB high-pass)
_K_SHELF_B = np.array([1.53512485958697, -2.69169618940638, 1.19839281085285])
_K_SHELF_A = np.array([1.0, -1.69065929318241, 0.73248077421585])
_K_HP_B = np.array([1.0, -2.0, 1.0])
_K_HP_A = np.array([1.0, -1.99004745483398, 0.99007225036621])

_BANDS = {
    "sub": (0.0, 60.0),
    "low": (60.0, 250.0),
    "mid": (250.0, 2000.0),
    "high": (2000.0, 8000.0),
    "air": (8000.0, np.inf),
}


def _db_amp(x: float) -> float:
    return float(20.0 * np.log10(max(float(x), 1e-10)))


def _stft_mag(x: np.ndarray, sr: int) -> np.ndarray:
    """Hann-windowed magnitude spectrogram, shape (n_frames, n_bins)."""
    if x.size < _N_FFT:
        pad = np.zeros(_N_FFT)
        pad[: x.size] = x
        x = pad
    n_frames = 1 + (x.size - _N_FFT) // _HOP
    idx = np.arange(_N_FFT)[None, :] + _HOP * np.arange(n_frames)[:, None]
    frames = x[idx] * np.hanning(_N_FFT)
    return np.abs(np.fft.rfft(frames, axis=1))


def _lufs_approx(data: np.ndarray, sr: int) -> float:
    """K-weighted RMS loudness (ungated BS.1770 approximation)."""
    y = lfilter(_K_SHELF_B, _K_SHELF_A, data, axis=0)
    y = lfilter(_K_HP_B, _K_HP_A, y, axis=0)
    # channel weights: 1.0 for L/R (only stereo/mono expected here)
    z = np.mean(y ** 2, axis=0).sum()
    return float(-0.691 + 10.0 * np.log10(max(float(z), 1e-12)))


def _onset_count(flux: np.ndarray, frame_rate: float) -> int:
    if flux.size == 0 or float(flux.max()) <= 0.0:
        return 0
    fl = flux / float(flux.max())
    height = float(fl.mean() + fl.std())
    distance = max(1, int(round(0.05 * frame_rate)))
    peaks, _ = find_peaks(fl, height=height, distance=distance)
    return int(peaks.size)


def _tempo(flux: np.ndarray, frame_rate: float,
           bpm_hint: float | None = None) -> tuple[float, float, list[dict]]:
    """Autocorrelation tempo estimate in 60-200 bpm.

    Returns ``(bpm, confidence, candidates)`` where candidates is a ranked
    list of ``{"bpm", "confidence"}`` dicts: the top comb-scored peaks plus
    the half/double (octave) relatives of the leader, so callers can see
    metrical-level ambiguity (half-time trap reads as bpm/2 etc.).

    Candidate lags are scored as autocorrelation x a log-Gaussian tempo
    prior (Ellis 2007 style) centered at ``bpm_hint`` when given (tight,
    sigma 0.25 octaves — the hint picks the metrical level) else at 115 bpm
    (wide, sigma 0.7). Each candidate's comb score adds the autocorrelation
    at its half and double lags so true-tempo families beat spurious peaks.
    """
    env = flux - (float(flux.mean()) if flux.size else 0.0)
    if env.size < 8 or not np.any(np.abs(env) > 1e-12):
        return 0.0, 0.0, []
    ac = np.correlate(env, env, mode="full")[env.size - 1:]
    if ac[0] <= 0.0:
        return 0.0, 0.0, []
    lmin = max(1, int(round(frame_rate * 60.0 / 200.0)))
    lmax = min(ac.size - 2, int(round(frame_rate * 60.0 / 60.0)))
    if lmax <= lmin:
        return 0.0, 0.0, []
    lags = np.arange(lmin, lmax + 1)
    bpms = 60.0 * frame_rate / lags
    if bpm_hint is not None and bpm_hint > 0:
        prior = np.exp(-0.5 * (np.log2(bpms / float(bpm_hint)) / 0.25) ** 2)
    else:
        prior = np.exp(-0.5 * (np.log2(bpms / 115.0) / 0.7) ** 2)
    base = ac[lmin:lmax + 1] / ac[0]

    def _ac_at_bpm(b: float) -> float:
        """Normalized autocorrelation at the lag of tempo b (0 if out of reach)."""
        if b <= 0:
            return 0.0
        lag = int(round(frame_rate * 60.0 / b))
        if 1 <= lag < ac.size:
            return float(max(ac[lag] / ac[0], 0.0))
        return 0.0

    # comb score: own lag + half-weight octave relatives (true-tempo families
    # reinforce across metrical levels; spurious peaks don't)
    comb = np.array([
        float(base[i]) + 0.5 * _ac_at_bpm(2.0 * bpms[i]) + 0.5 * _ac_at_bpm(bpms[i] / 2.0)
        for i in range(base.size)
    ])
    score = comb * prior

    # ranked local maxima -> candidates (dedupe within 4%)
    peak_idx, _ = find_peaks(score)
    if peak_idx.size == 0:
        peak_idx = np.array([int(np.argmax(score))])
    order = peak_idx[np.argsort(score[peak_idx])[::-1]]
    candidates: list[dict] = []
    for i in order:
        b = float(bpms[i])
        if any(abs(b - c["bpm"]) / c["bpm"] < 0.04 for c in candidates):
            continue
        candidates.append({"bpm": round(b, 2),
                           "confidence": round(float(np.clip(base[i], 0.0, 1.0)), 4)})
        if len(candidates) >= 3:
            break
    # octave/half/double relatives of the leader (even outside 60-200)
    if candidates:
        lead = candidates[0]["bpm"]
        for rel in (lead / 2.0, lead * 2.0):
            if not any(abs(rel - c["bpm"]) / c["bpm"] < 0.04 for c in candidates):
                candidates.append({"bpm": round(rel, 2),
                                   "confidence": round(_ac_at_bpm(rel), 4)})

    def _score_at(lag: int) -> float:
        return float(score[lag - lmin]) if lmin <= lag <= lmax else -np.inf

    best = lmin + int(np.argmax(score))
    # prefer the fundamental period: fold down while the half lag is
    # comparably strong (avoids locking onto 2x the beat period). Skipped
    # when a hint is given — the hint prior already picks the octave.
    if bpm_hint is None:
        while True:
            h0, h1 = best // 2, (best + 1) // 2
            if h0 < lmin:
                break
            half = h0 if _score_at(h0) >= _score_at(h1) else h1
            if _score_at(half) >= 0.85 * _score_at(best):
                best = half
            else:
                break
    # parabolic interpolation around the peak
    y0, y1, y2 = float(ac[best - 1]), float(ac[best]), float(ac[best + 1])
    denom = y0 - 2.0 * y1 + y2
    delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
    lag = best + float(np.clip(delta, -0.5, 0.5))
    bpm = float(np.clip(60.0 * frame_rate / lag, 60.0, 200.0))
    confidence = float(np.clip(ac[best] / ac[0], 0.0, 1.0))
    return bpm, confidence, candidates


def analyze(wav_path: str, bpm_hint: float | None = None) -> dict:
    """Analyze a WAV file; see SPEC section 10 for the returned keys.

    ``bpm_hint`` enables ``bar_rms_db`` (per-bar loudness, 4/4 assumed) AND
    biases the tempo estimate toward the hinted metrical level (log-Gaussian
    prior centered on the hint). ``tempo_candidates`` always reports the
    top ranked tempos plus octave relatives of the leader.
    """
    data, sr = sf.read(str(wav_path), dtype="float64", always_2d=True)
    n, channels = data.shape
    duration_s = n / float(sr)
    mono = data.mean(axis=1)

    peak = float(np.max(np.abs(data))) if n else 0.0
    rms = float(np.sqrt(np.mean(data ** 2))) if n else 0.0

    # framing / spectral features
    mag = _stft_mag(mono, sr)
    frame_rate = sr / _HOP
    freqs = np.fft.rfftfreq(_N_FFT, 1.0 / sr)
    frame_energy = mag.sum(axis=1)
    valid = frame_energy > 1e-9
    if valid.any():
        centroids = (mag @ freqs)[valid] / frame_energy[valid]
        spectral_centroid = float(centroids.mean())
    else:
        spectral_centroid = 0.0

    flux = np.zeros(mag.shape[0])
    if mag.shape[0] > 1:
        flux[1:] = np.maximum(mag[1:] - mag[:-1], 0.0).sum(axis=1)

    onset_count = _onset_count(flux, frame_rate)
    tempo_bpm, tempo_conf, tempo_candidates = _tempo(flux, frame_rate, bpm_hint)

    # band energies (Welch PSD integrated per band)
    band_energy_db: dict[str, float] = {}
    if n >= 32:
        f, pxx = welch(mono, fs=sr, nperseg=min(8192, n))
        df = f[1] - f[0] if f.size > 1 else 1.0
        for name, (lo, hi) in _BANDS.items():
            mask = (f >= lo) & (f < hi)
            band_energy_db[name] = float(
                10.0 * np.log10(max(float(pxx[mask].sum() * df), 1e-12)))
    else:
        band_energy_db = {name: -120.0 for name in _BANDS}

    # silence ratio: non-overlapping 50 ms frames below -60 dBFS RMS
    frame_len = max(1, int(round(0.05 * sr)))
    n_frames = n // frame_len
    if n_frames > 0:
        fr = mono[: n_frames * frame_len].reshape(n_frames, frame_len)
        frame_rms = np.sqrt(np.mean(fr ** 2, axis=1))
        silence_ratio = float(np.mean(frame_rms < 10.0 ** (-60.0 / 20.0)))
    else:
        silence_ratio = 1.0 if rms < 1e-3 else 0.0

    # stereo width: 1 - |corr(L, R)|
    stereo_width = 0.0
    if channels >= 2:
        left, right = data[:, 0], data[:, 1]
        if float(left.std()) > 1e-12 and float(right.std()) > 1e-12:
            corr = float(np.corrcoef(left, right)[0, 1])
            if np.isfinite(corr):
                stereo_width = float(1.0 - abs(corr))

    bar_rms_db: list[float] = []
    if bpm_hint is not None and bpm_hint > 0 and n:
        bar_len = max(1, int(round(sr * 240.0 / float(bpm_hint))))
        for i in range(0, n, bar_len):
            seg = data[i: i + bar_len]
            bar_rms_db.append(_db_amp(float(np.sqrt(np.mean(seg ** 2)))))

    return {
        "duration_s": duration_s,
        "sample_rate": int(sr),
        "channels": int(channels),
        "peak_db": _db_amp(peak),
        "rms_db": _db_amp(rms),
        "lufs_approx": _lufs_approx(data, sr) if n else -120.0,
        "clipped_sample_count": int(np.count_nonzero(np.abs(data) >= 0.999)),
        "silence_ratio": silence_ratio,
        "spectral_centroid_hz_mean": spectral_centroid,
        "band_energy_db": band_energy_db,
        "onset_count": onset_count,
        "onset_density_per_s": (onset_count / duration_s) if duration_s > 0 else 0.0,
        "tempo_bpm_estimate": tempo_bpm,
        "tempo_confidence": tempo_conf,
        "tempo_candidates": tempo_candidates,
        "stereo_width": stereo_width,
        "bar_rms_db": bar_rms_db,
    }
