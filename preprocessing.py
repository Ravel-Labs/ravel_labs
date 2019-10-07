import os
import librosa
import numpy as np
from scipy.fftpack import fft
from scipy.stats import rankdata


def export_params(path, files, rank_threshold, sr, max_n):
    '''
    This function takes a directory path, list of files, rank threshold,
    sample rate, and the number of top mask values and returns a list of
    parameters in dictionary form. Each list element contains a file path
    and the top mask values according to the mask function.
    '''

    params_list = []
    for idx in range(len(files)):
        masker_path = os.path.join(path, files[idx])
        maskee_path = [os.path.join(path, files[i]) for i in range(len(files)) if i != idx]
        mask_array = mask(masker_path, maskee_path, rank_threshold, sr, max_n)
        masker_dic = {masker_path: mask_array}
        params_list.append(masker_dic)
    return params_list

def fft_2d(audio_signal, window_size, hop_length, sr):
	'''
	This function takes the path to an audio signal and the accompanying
	sample rate and returns the magnitude (decibels) and time of the 
	signal in a 2d numpy array.

	After loading the audio signal, the function uses a short-time
	fourier transform on non-overlapping windows of size 1024. We
	then transform the frequency from amplitude to decibel scale.
	'''	
	y, sr = librosa.load(audio_signal, sr=sr)
	D = np.abs(librosa.core.stft(y, n_fft=window_size, hop_length=hop_length))
	D_db = librosa.amplitude_to_db(D)
	return D_db

def fft_avg(audio_signal, window_size, hop_length, sr):
	'''
	This function takes the path to an audio signal and the accompanying
	sample rate and returns the spectrum of the signal.

	After loading the audio signal, the function uses a short-time
	fourier transform on non-overlapping windows of size 1024. We
	then transform the frequency from amplitude to decibel scale.

	Lastly, we create the average magnitude over the entire length of 
	the audio signal.
	'''
	y, sr = librosa.load(audio_signal, sr=sr)
	D = np.abs(librosa.core.stft(y, n_fft=window_size, hop_length=hop_length))
	D_db = librosa.amplitude_to_db(D)
	return np.mean(D_db, axis=1)

def file_scraper(path): return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

def freq_bin(signal, n, sr): return n * (sr / signal.shape[0])

def mask(signal_a, signals, rank_threshold, window_size, hop_length, sr, max_n):
    '''
    This function returns a list of parameters that can be used during
    the equalization process. 

    The input for this function is a list of
    loaded audio signals, the rank threshold for the masking formula,
    the sample rate, and the number of top mask values to return. 

    For the max_n occurences of masking within a frequency, the function 
    returns their signal index, frequency bin,
    and the value of the masking function.
    ''' 
    mask = np.array([])
    mask_info = []
    masker_fft = fft_avg(signal_a, window_size, hop_length, sr)
    maskee_ffts = [fft_avg(signal, window_size, hop_length, sr) for signal in signals]
    masker_rank = rank_signal_1d(masker_fft)
    maskee_ranks = [rank_signal_1d(maskee_fft) for maskee_fft in maskee_ffts]
    # creates boolean matrices that return 1 or 0 based on the rank threshold
    masker_rank_mat = np.where(masker_rank > rank_threshold, 1, 0)
    num_bins = masker_fft.shape[0]
    for i in range(len(signals)):
        maskee_fft = maskee_ffts[i]
        maskee_rank = maskee_ranks[i]
        maskee_rank_mat = np.where(maskee_rank <= 10, 1, 0)
        # uses elementwise multiplication between the boolean matrices and frequency
        # signals to calculate the spectral masking for all values that meet the rank
        # threshold conditions
        mask_ab = (masker_rank_mat * maskee_rank_mat) * (masker_fft - maskee_fft)
        mask = np.append(mask, mask_ab)
    # saves max_n mask values from the mask matrix and uses the indices to add the
    # frequency and mask value to a mask info array    
    top_m = np.argsort(mask)[-max_n:]
    idx = np.unravel_index(top_m, mask.shape)[0]
    for i in idx:
        freq_bin = (i % num_bins) * (sr / num_bins)
        mask_val = mask[i]
        if (mask_val) > 0 and (freq_bin <= 20000) and (freq_bin >= 20):
            mask_info.append([freq_bin, mask_val])
    return np.array(mask_info)

def mask_2d(signals, rank_threshold, window_size, hop_length, sr, top_n):
    '''
    This function returns a list of parameters that can be used during
    the equalization process using time and frequency opposed to an
    averaged frequency over time. 

    The input for this function is a list of
    loaded audio signals, the rank threshold for the masking formula,
    the sample rate, and the number of top mask values to return. 

    For the top x occurences of masking within a frequency, the function 
    returns their signal index, frequency bin,
    and the value of the masking function.
    '''
    n = len(signals)
    fft_signals = [fft_2d(signal, window_size, hop_length, sr) for signal in signals]
    rank_signals = [rank_signal_2d(fft_signal) for fft_signal in fft_signals]
    mask_info = []
    for i in range(n):
        for j in range(n-1):
            sig_a = fft_signals[i]
            sig_b = fft_signals[(i + j + 1) % n]
            rank_a = rank_signals[i]
            rank_b = rank_signals[(i+j+1) % n]
            # creates boolean matrices that return 1 or 0 based on the rank threshold
            r_a = np.where(rank_a > rank_threshold, 1, 0)
            r_b = np.where(rank_b <= rank_threshold, 1, 0)
            # uses elementwise multiplication between the boolean matrices and frequency
            # signals to calculate the spectral masking for all values that meet the rank
            # threshold conditions
            v = (r_a*r_b)*(sig_a - sig_b)
            # flattens matrix to find the top_n values and appends the signal
            # index, freq_bin, and mask value to the mask_info list
            v_1d = v.flatten()
            idx_1d = v_1d.argsort()[-top_n:]
            x_idx, y_idx = np.unravel_index(idx_1d, v.shape)
            for x, y, in zip(x_idx, y_idx):
                freq_bin = x * (sr / v.shape[0])
                mask_ab = v[x][y]
                mask_info.append([i, freq_bin, mask_ab])
    return mask_info

def rank_signal_1d(audio_signal):
	return np.abs(rankdata(audio_signal, method='ordinal') - (audio_signal.shape[0] - 1))

def rank_signal_2d(audio_signal): 
    a = np.zeros(audio_signal.shape)
    for row in range(audio_signal.shape[1]):
        a[:, row] = np.abs(rankdata(audio_signal[:, row], method='ordinal') - (audio_signal.shape[0] - 1))
    return a

def spectrum(fft_signal): return np.mean(fft_signal, axis=0)

def half_wave_rectifier(x): return (x + np.absolute(x)) / 2

def spectral_flux(fft_signal):
    difference = np.diff(np.absolute(fft_signal), axis=1)
    hwr = half_wave_rectifier(difference)
    spectral_flux = hwr / np.absolute(fft_signal)
    return np.sum(spectral_flux)

def alpha(time_constant, sr): 
    '''
    Alpha signifies the forget factor for parameter autonomation equations
    '''
    return np.exp(-1 / (time_constant * sr))

def rms_sqaured(audio_signal, time_constant, sr):
    y, sr = librosa.load(audio_signal, sr=sr)
    alpha = alpha(time_constant, sr)
    rms = librosa.feature.rms(y)
    rms_squared = np.zeros(rms.shape)
    for i in range(1, len(rms)):
        rms_squared[i] = alpha * rms[1,i-1]**2 + (1-alpha)*np.absolute(y[i]**2)
    return rms_squared

def peak(audio_signal, time_constant, sr):
    y, sr = librosa.load(audio_signal, sr=sr)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    alpha = alpha(time_constant, sr)
    peaks = librosa.util.peak_pick(onset_env, 3, 3, 3, 5, 0.5, 1)
    peaks_squared = np.zeros(peaks.shape)
    for i in range(1, len(peaks)):
        peak_factor = alpha * peaks[i-1]**2 + (1 - alpha) * np.absolute(y[i]**2)
        peaks_squared[i] = max(y[i]**2, peak_factor)
    return peaks_squared

def crest_factor(rms, peaks): return np.sqrt(peaks/rms)


    