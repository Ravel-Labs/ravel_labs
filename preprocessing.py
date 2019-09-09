import librosa
import numpy as np
from scipy.fftpack import fft
from scipy.stats import rankdata

def freq_bin(signal, n, sr): return n * (sr / signal.shape[0])

def masking_formula(signal_a, signal_b, freq, time, rank_threshold, sr):
	rank_a, rank_b = rank_freq_regions(signal_a), rank_freq_regions(signal_b)
	if (rank_b[freq][time] <= rank_threshold) and (rank_threshold < rank_a[freq][time]):
		freq_bin = freq_bin(signal_a, time, sr)
		mask_ab = signal_a[freq][time] - signal_b[freq][time]
		return mask_ab, freq_bin
	else:
		return 0

def offline_fmag(audio_signal, sr):
	y, sr = librosa.load(audio_signal, sr=sr)
	D = np.abs(librosa.core.stft(y, hop_length=1024))
	D_db = librosa.amplitude_to_db(D)
	return D_db

def rank_freq_regions(audio_signal): 
    a = np.zeros(audio_signal.shape)
    for row in range(audio_signal.shape[1]):
        a[:, row] = np.abs(rankdata(audio_signal[:, row], method='ordinal') - (audio_signal.shape[0] - 1))
    return a

def spectrum(fft_signal): return np.mean(fft_signal, axis=0)