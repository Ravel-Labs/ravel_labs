import numpy as np
from scipy.signal import butter, lfilter, freqz
from preprocessing import *

def butter_filter(cutoff, sr, order, btype):
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    return b, a

def apply_bfilter(signal, cutoff, sr, order, btype):
    b, a = butter_filter(cutoff, sr, order, btype)
    y = lfilter(b, a, signal)
    return y

def preprocess_pll(x, high_cutoff, low_cutoff, sr, high_order, low_order, x_env):
    x_low = apply_bfilter(x, cutoff=low_cutoff, sr=sr, order=low_order, btype='low')
    x = apply_bfilter(x_low, cutoff=high_cutoff, sr=sr, order=high_order, btype='high')
    x_in = x * (1/x_env)
    return x_in

def h_lp(fc, sr, Q):
    k = np.tan(np.pi*(fc/sr))
    k_1 = 1 / (1+(k/Q)+k**2)
    a_0 = 1
    a_1 = 2 * (k**2 - 1) * k_1
    a_2 = (1 - (k / Q) + k**2) * k_1
    b_0 = k**2 * k_1
    b_1 = 2 * k * k_1
    b_2 = k**2 * k_1
    return np.array([b_0, b_1, b_2]), np.array([a_0, a_1, a_2])

def calc_f_osc(x_d, b, a): return lfilter([1+b[0], b[1], b[2]], a, x_d)

def calc_f0(x_d, b, a): return lfilter(b, a, x_d) * 2

# def calc_f0(x_d, cutoff, sr): return apply_bfilter(x_d, cutoff=cutoff, sr=sr, order=2, btype="low")

def eq_filter(x, fc, sr, G, f_b, f_type="boost"):
    d = -np.cos(2*np.pi * (fc/sr))
    V0 = 10**(G/20)
    H0 = V0 - 1
    c_boost = (np.tan(np.pi * (f_b / sr)) - 1) / np.tan(np.pi*(f_b/sr) + 1)
    c_cut = (np.tan(np.pi*(f_b/sr)) - V0) / (np.tan(np.pi*(f_b/sr)) + V0)
    x_h = np.zeros(x.shape[0])
    y1 = np.zeros(x.shape[0])
    y = np.zeros(x.shape[0])
    if f_type == "boost":
        c = c_boost
    elif f_type == "cut":
        c = c_cut
    for n in range(x_h.shape[0]):
        if n < 2:
            x_h[n] = x[n]
            y1[n] = -c * x_h[n]
        else:
            x_h[n] = x[n] - d * (1 - c) * x_h[n-1] + c * x_h[n-2]
            y1[n] = -c * x_h[n] + d * (1 - c) * x_h[n-1] + x_h[n-2]
        y[n] = (H0 / 2) * (x[n] - y1[n]) + x[n]
    return y

def tonal_balance(path, high_cutoff, low_cutoff, high_order, low_order, x_env, Q, K_d, fc, G, f_b, f_type):
    sig = Signal(path=path, window_size=1024, hop_length=512)
    x = sig.signal
    sr = sig.sr
    x_in = preprocess_pll(x, high_cutoff=high_cutoff, low_cutoff=low_cutoff, 
                          sr=sr, high_order=high_order, low_order=low_order, x_env=x_env)
    b, a = h_lp(fc=fc, sr=sr, Q=Q)
    x_d = x_in * K_d
    f_osc = calc_f_osc(x_d, b, a)
    y_cos_osc = np.zeros(f_osc.shape)
    y_sin_osc = np.zeros(f_osc.shape)
    for n in range(x_in.shape[0]):
        y_cos_osc[n] = np.cos(2*np.pi * (f_osc[n] / 44100) * n)
        y_sin_osc[n] = np.sin(2*np.pi * (f_osc[n] / 44100) * n)
    x_d = x_d * y_cos_osc
    f0 = calc_f0(x_d, b, a)
    f0_fft = np.abs(np.fft.fft(f0))
    bins_arr = [20, 40, 60, 80, 100, 150, 200, 250, 500, 750, 1000, 1250, 1500]
    val, bins = np.histogram(f0_fft, bins=bins_arr, range=(20, 1500), density=False)
    top_vals = np.argsort(mask)[-5:]
    top_freqs = bins[top_vals]
    y = x
    for freq in top_freqs:
        y = eq_filter(y, fc=fc, sr=sr, G=G, f_b=f_b, f_type=f_type)
    return y