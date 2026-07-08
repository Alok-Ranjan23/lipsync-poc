"""Mel-spectrogram extraction matching the original Wav2Lip preprocessing.

Wav2Lip's ONNX generator was trained on mel features produced with these exact
hyper-parameters, so the mel pipeline must reproduce them or the lips won't sync.
Ported from https://github.com/Rudrabha/Wav2Lip (audio.py / hparams.py).
"""
import librosa
import numpy as np
from scipy import signal

SAMPLE_RATE = 16000
N_FFT = 800
HOP_SIZE = 200
WIN_SIZE = 800
NUM_MELS = 80
FMIN = 55
FMAX = 7600
MIN_LEVEL_DB = -100
REF_LEVEL_DB = 20
PREEMPHASIS = 0.97
MAX_ABS_VALUE = 4.0

_mel_basis = None


def load_wav(path):
    wav, _ = librosa.load(path, sr=SAMPLE_RATE)
    return wav


def _preemphasis(wav):
    return signal.lfilter([1, -PREEMPHASIS], [1], wav)


def _stft(y):
    return librosa.stft(y=y, n_fft=N_FFT, hop_length=HOP_SIZE, win_length=WIN_SIZE)


def _get_mel_basis():
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = librosa.filters.mel(
            sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=NUM_MELS, fmin=FMIN, fmax=FMAX
        )
    return _mel_basis


def _amp_to_db(x):
    return 20.0 * np.log10(np.maximum(1e-5, x))


def _normalize(s):
    # symmetric mels, max_abs_value scaling with clipping
    return np.clip(
        (2 * MAX_ABS_VALUE) * ((s - MIN_LEVEL_DB) / (-MIN_LEVEL_DB)) - MAX_ABS_VALUE,
        -MAX_ABS_VALUE,
        MAX_ABS_VALUE,
    )


def melspectrogram(wav):
    d = _stft(_preemphasis(wav))
    s = _amp_to_db(np.dot(_get_mel_basis(), np.abs(d))) - REF_LEVEL_DB
    return _normalize(s)
