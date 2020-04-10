import lib.preprocessing as preprocessing
from pyo import *
from pysndfx import AudioEffectsChain
import librosa
import numpy as np

class Signal:
    def __init__(self, path, signal, n_fft, window_size, hop_length, peak):
        self.path = path
        self.sr = librosa.get_samplerate(self.path)
        self.n_fft = n_fft
        self.window_size = window_size
        self.hop_length = hop_length
        self.signal = signal
        self.signal_db = librosa.amplitude_to_db(self.signal)
        self.peak = peak
        self.fft = np.abs(librosa.core.stft(self.signal, n_fft=self.n_fft, 
                                            win_length=self.window_size, hop_length=self.hop_length))
        self.num_bins = self.fft.shape[0]
        self.fft_db = librosa.amplitude_to_db(self.fft)
        self.norm_fft_db = preprocessing.compute_norm_fft_db(self.signal_db, self.peak, 
                                                self.n_fft, self.window_size, self.hop_length)
        self.freqs = np.array([i * self.sr / self.fft.shape[0] for i in range(self.num_bins)])


class EQSignal(Signal):
    def __init__(self, path, signal, n_fft, window_size, hop_length, peak, bins, roll_percent, seconds,
                rank_threshold, max_n, min_overlap_ratio, max_eq):
        super().__init__(path, signal, n_fft, window_size, hop_length, peak)
        self.bins = bins
        self.roll_percent = roll_percent
        self.seconds = seconds
        self.chunk_fft_db = preprocessing.compute_chunk(self.norm_fft_db, self.window_size, self.sr, self.seconds)
        self.rank = preprocessing.compute_rank(self.chunk_fft_db)
        self.sparse_vec = preprocessing.compute_sparsity(self.rank, self.num_bins)
        self.rank_threshold = rank_threshold
        self.max_n = max_n
        self.min_overlap_ratio = min_overlap_ratio
        self.max_eq = max_eq

    def compute_energy_percent(self):
        total_energy = np.sum(self.chunk_fft_db)
        energy_percents = []
        for i in range(len(self.bins)-1):
            arr = np.argwhere((self.freqs >= self.bins[i]) & (self.freqs < self.bins[i+1])).flatten()
            bin_sum = np.sum([self.chunk_fft_db[i] for i in arr])
            energy_percent = bin_sum / total_energy
            energy_percents.append(energy_percent)
        if energy_percents[0] < 0.2:
            return self.bins[1]

    def compute_rolloff(self):
        rolloffs = librosa.feature.spectral_rolloff(self.signal, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, 
                             win_length=self.window_size, roll_percent=self.roll_percent)
        r_active = rolloffs[0, np.argwhere(rolloffs > 0).flatten()]
        r_avg = np.mean(r_active)
        return r_avg

    def compute_mask(self, signal, overlap_vec, num_overlaps):
        soa_vec_i = preprocessing.sparse_overlap_avg(self.num_bins, self.chunk_fft_db, 
                                                    self.sparse_vec, overlap_vec, num_overlaps)
        soa_vec_j = preprocessing.sparse_overlap_avg(signal.num_bins, signal.chunk_fft_db, 
                                                    signal.sparse_vec, overlap_vec, num_overlaps)
        r_soa_vec_i = preprocessing.rank_soa_vec(soa_vec_i)
        r_soa_vec_j = preprocessing.rank_soa_vec(soa_vec_j)
        masker_vec_i = preprocessing.masker_rank_vec(r_soa_vec_i)
        maskee_vec_j = preprocessing.maskee_rank_vec(r_soa_vec_j)
        mask_ij = ((masker_vec_i * maskee_vec_j) * (soa_vec_i - soa_vec_j)).flatten()
        m_f = np.concatenate((mask_ij[:, np.newaxis], self.freqs[:, np.newaxis]), axis=1)
        return mask_ij      

    def eq_params(self, signals):
        num_signals = len(signals)
        num_bins = self.num_bins
        max_n = self.max_n
        sr = self.sr
        max_eq = self.max_eq
        for i in range(num_signals):
            # mask = np.empty(shape=[0, 2])
            mask = np.array([])
            eq_info = []
            overlap_vec, num_overlaps, overlap_ratio = preprocessing.overlap(self.sparse_vec, signals[i].sparse_vec)
            if (overlap_ratio > self.min_overlap_ratio):
                mask_ij = self.compute_mask(signals[i], overlap_vec, num_overlaps)
                # m_f = np.concatenate((mask_ij[:, np.newaxis], signals[i].freqs[:, np.newaxis]), axis=1)
                # mask = np.append(mask, m_f, axis=0)
                mask = np.append(mask, mask_ij, axis=0)
            else:
                mask_ij = 0
        mask_m = np.zeros(num_bins)
        for b in range(num_bins):
            # arr = mask[b::num_bins, :]
            arr = mask[b::num_bins]
            # max_b, _ = np.max(arr, axis=0)
            max_b = np.max(arr)
            mask_m[b] = max_b
        top_m = np.argsort(mask_m)[-max_n:]
        top_m_max = mask_m[top_m].max()
        idx = np.unravel_index(top_m, mask_m.shape)[0]
        for x in idx:
            freq_bin = x  * (sr / num_bins)
            mask_val = mask_m[x]
            if (mask_val > 0) and (freq_bin <= 20000) and (freq_bin >= 20):
                mask_val_scaled = (mask_val / top_m_max) * max_eq
                eq_type = 0
                eq_info.append([freq_bin, mask_val_scaled, eq_type])
        rolloff = self.compute_rolloff()
        energy_percent = self.compute_energy_percent()
        eq_info.append([rolloff, 0.71, 2])
        if energy_percent is not None:
            eq_info.append([energy_percent, 0.71, 1])
        return eq_info

    # def equalization(self, eq_info, Q):
    #     num_filters = len(eq_info)
    #     s = Server().boot()
    #     c = Converter(self.signal)
    #     out = c.numpy_to_pyo(s)
    #     for i in range(num_filters):
    #         freq = float(eq_info[i][0])
    #         gain = float(eq_info[i][1])
    #         eq_type = int(eq_info[i][2])
    #         eq = EQ(out, freq=freq, q=Q, boost=-gain, type=eq_type)
    #         out = eq
    #     out = out.out()
    #     numpy_out = c.pyo_to_numpy(out, s)
    #     return numpy_out

    def equalization(self, eq_info, Q):
        num_filters = len(eq_info)
        y = self.signal
        for i in range(num_filters):
            freq = float(eq_info[i][0])
            gain = float(eq_info[i][1])
            eq_type = int(eq_info[i][2])
            if eq_type == 0:
                eq = AudioEffectsChain().equalizer(freq, Q, gain)
                y = eq(y)
            elif eq_type == 1:
                eq = AudioEffectsChain().highpass(freq)
                y = eq(y)
            elif eq_type == 2:
                eq = AudioEffectsChain().lowpass(freq)
                y = eq(y)
        return y



class CompressSignal(Signal):
    def __init__(self, path, signal, n_fft, window_size, hop_length, peak, 
                time_constant, order, cutoff, std, attack_max, release_max):
        super().__init__(path, signal, n_fft, window_size, hop_length, peak)
        self.time_constant = time_constant
        self.order = order
        self.cutoff = cutoff
        self.std = std
        self.attack_max = attack_max
        self.release_max = release_max
        self.rms = librosa.feature.rms(signal, frame_length=self.window_size, hop_length=self.hop_length)
        self.rms_db = np.mean(librosa.amplitude_to_db(self.rms))
        self.peak_db = librosa.amplitude_to_db(np.sum(self.fft, axis=0)).max()
        self.crest_factor = self.peak_db / self.rms_db
        self.lfe = preprocessing.compute_lfe(self.signal, self.order, self.cutoff, self.sr)
    
    def compute_wp(self, cf_avg): return preprocessing.wp(self.crest_factor, cf_avg, self.std)

    def compute_lf_weighting(self, lfa): return self.lfe / lfa

    def ratio(self, wf, wp): return float(0.54*wp + 0.764*wf + 1)

    def threshold(self, wp):return float(-11.03 + 0.44*self.rms_db - 4.897*wp)

    def knee_width(self, threshold): return abs(threshold) / 2

    def attack(self): return float((2*self.attack_max) / self.crest_factor ** 2)

    def release(self): return float((2*self.release_max) / self.crest_factor ** 2)

    def comp_params(self, cfa, lfa):
        w_p = self.compute_wp(cfa)
        w_f = self.compute_lf_weighting(lfa)
        r = self.ratio(w_f, w_p)
        t = self.threshold(w_p)
        kw = self.knee_width(t)
        a = self.attack()
        rel = self.release()
        return [t, r, a, rel, kw]

    # def compression(self, params):
    #     s = Server().boot()
    #     c = Converter(self.signal)
    #     out = c.numpy_to_pyo(s)
    #     out = Compress(out, thresh=params[0], ratio=params[1], risetime=params[2], falltime=params[3], knee=0.4).out()
    #     numpy_out = c.pyo_to_numpy(out, s)
    #     makeup_gain = preprocessing.compute_makeup_gain(self.signal, numpy_out, self.sr)
    #     fx = (AudioEffectsChain().gain(makeup_gain))
    #     return fx(numpy_out)

    def compression(self, params):
        compress = AudioEffectsChain.compand(attack=params[2], decay=params[3], soft_knee=params[1], 
                    threshold=params[0], db_from=params[0], db_to=params[0])
        return compress(self.signal)

class FaderSignal(Signal):
    def __init__(self, path, signal, n_fft, window_size, hop_length, peak,
                decay, step, lead, B):
        super().__init__(path, signal, n_fft, window_size, hop_length, peak)
        self.decay = decay
        self.step = step
        self.lead = lead
        self.B = B
        self.x_norm = preprocessing.normalize(self.signal_db, self.peak)
    
    def full_loudness(self):
        x_norm = self.x_norm
        sr = self.sr
        step = self.step
        decay = self.decay
        N = x_norm.shape[0]
        num_segments = int(N / (step * sr))
        L_m = np.zeros(x_norm.shape)
        for n in range(num_segments):
            L_m[step*sr*n: step*sr*(n+1)] = preprocessing.loudness(x_norm[step*sr*n: step*sr*(n+1)], decay)
        L_m[step*sr*(n+1):] = preprocessing.loudness(x_norm[step*sr*(n+1):], decay)
        return L_m

    def compute_fader(self, L_av, L2):
        F_m = np.zeros(L2.shape)
        for n in range(len(F_m)):
            F_m[n] = 10 ** ((L_av[n] - L2[n]) / 20)
        if self.lead == True:
            F_m[n] = F_m[n] * 10 ** (self.B/20)
        return F_m

    def fader(self, fader_output): return self.signal * fader_output

class PanSignal(Signal):
    def __init__(self, path, signal, n_fft, window_size, hop_length, peak, 
                cutoffs, window, order, btype):
        super().__init__(path, signal, n_fft, window_size, hop_length, peak)
        self.window = window
        self.order = order
        self.btype = btype
        self.cutoffs = cutoffs
        self.window_step = int(self.sr * self.window)
        self.num_steps = int(self.signal.shape[0] / self.window_step)
        self.K = len(cutoffs)


    def lead_filter(self): 
        return preprocessing.peak_filter_bank(self.signal, self.cutoffs, 
                                self.sr, self.order, self.btype, 
                                self.window_step, self.num_steps)

    # def pan(self, P):
    #     s = Server().boot()
    #     c = Converter(self.signal)
    #     out = c.numpy_to_pyo(s)
    #     pan_out = Pan(out, pan=P).out()
    #     numpy_out = c.pyo_to_numpy(pan_out, s)
    #     return numpy_out

    def pan(self, P):
        right = np.sin(P * np.pi) * self.signal
        left = np.cos(P * np.pi) * self.signal
        return (right, left)


class Converter:
    def __init__(self, signal):
        self.signal = signal
        self.buffer_size = signal.shape[0]

    def numpy_to_pyo(self, s):
        s.start()
        t = DataTable(size=self.buffer_size)
        osc = TableRead(t, freq=t.getRate())
        arr = np.asarray(t.getBuffer())
        pyo_x = process(t, self.signal, osc)
        return pyo_x

    def pyo_to_numpy(self, out, s):
        t = DataTable(size=self.buffer_size)
        b = TableRec(out, t, 0.01).play()
        tf = TrigFunc(b["trig"], function=done, arg=t)
        numpy_x = done(t)
        s.shutdown()
        return numpy_x

class SignalAggregator:
    '''Computes all of the aggregated stats for each effect'''
    def __init__(self, sr, M):
        self.sr = sr
        self.M = M

    def lfa(self, lfes): return sum([lfe for lfe in lfes]) / self.M

    def cfa(self, cfs): return sum([cf for cf in cfs]) / self.M

    def panning_locations(self, filter_freqs, signal_peaks):
        N_ks = np.unique(signal_peaks, return_inverse=True, return_counts=True)
        num_k = N_ks[0].shape[0]
        Ps = []
        for k in np.arange(num_k):
            freq = N_ks[0][k]
            N_k = N_ks[2][k]
            P = np.zeros((N_k))
            for i in np.arange(N_k):
                if N_k == 1:
                    P[i] = 1/2
                elif N_k + i % 2 != 0:
                    P[i] = (N_k - i - 1) / (2 * (N_k - i))
                elif (N_k + i % 2 == 0) and (N_k != 1):
                    P[i][k] = 1 - ((N_k - i) / (2 * (N_k -1)))
                idx = np.argwhere(signal_peaks == freq)[0][i]
                Ps.append([idx, P[i]])
        return Ps

    def loudness_avg(self, channels, holdtime, ltrhold, utrhold, release, attack):
        gains = [preprocessing.noise_gate(channels[i], holdtime, ltrhold, utrhold, release, attack, self.sr) 
                 for i in range(len(channels))]
        gains = [np.where(gain < 1, 0, 1) for gain in gains]
        gain_val = np.array(gains).sum(axis=0)
        L_c = np.array(channels).sum(axis=0)
        L_av = np.where(gain_val > 0, L_c / gain_val, 0)
        # apply ema filter
        return L_av


def process(t, x, osc):
    "Fill the array (so the table) with white noise."
    arr = np.asarray(t.getBuffer())
    arr[:] = x
    return osc

def done(t):
    arr = np.asarray(t.getBuffer())
    return arr