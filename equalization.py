import scipy
from scipy.signal import iirnotch, lfilter
from preprocessing import *


def iir_bandpass(signal, freq, Q, fs):
    b, a = iirnotch(freq, Q, fs)
    y = lfilter(b, a, signal)
    return y

def processing(signals, rank_threshold, sr):
    # returns signal_name, freq_bin, and m_value for purposes of attentuating frequencies 
    freq_signals = [offline_fmag(signal) for signal in signals]
    maskers = []
    for sig in range(len(freq_signals[0])-1):
        for freq in range(freq_signals[sig].shape[0]):
            for time in range(freq_signals[sig].shape[1]):
                mask_ab, freq_bin = masking_formula(freq_signals[sig], freq_signals[sig+1], freq, time, rank_threshold, sr)
                if mask_ab != 0:
                    maskers.append([sig, mask_ab, freq_bin])
    return maskers

def three_band_equalizer(signal, fs, freqs, Q, gains):
    band1 = iir_bandpass(signal, freqs[0], Q, fs)*10**(gains[0]/20)
    band2 = iir_bandpass(signal, freqs[1], Q, fs)*10**(gains[1]/20)
    band3 = iir_bandpass(signal, freqs[2], Q, fs)*10**(gains[2]/20)
    signal = band1 + band2 + band3
    return signal

def eq_params(signals, rank_threshold, sr, top_n):
    '''
    This function returns a list of parameters that can be used during
    the equalization process. 

    The input for this function is a list of
    loaded audio signals, the rank threshold for the masking formula,
    the sample rate, and the number of top mask values to return. 

    For the top x occurences of masking within a frequency, the function 
    returns their signal index, frequency bin,
    and the value of the masking function.
    '''
    n = len(signals)
    fft_signals = [offline_fmag(signal) for signal in signals]
    rank_signals = [rank_freq_regions(fft_signal) for fft_signal in fft_signals]
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
            v = (r_a*rb)*(sig_a - sig_b)
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


