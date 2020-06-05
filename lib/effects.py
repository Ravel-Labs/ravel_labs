import lib.preprocessing as preprocessing
import librosa
import numpy as np
import pyloudnorm as pyln
from pysndfx import AudioEffectsChain
from scipy.io.wavfile import write


class Signal:
    """
    A base class used to represent an audio signal.

    Attributes
    ----------
    signal: np.ndarray [shape=(n,)], real valued
        input signal

    sr: float
        sample rate

    n_fft: int > 0
        length of the windowed signal after padding with zeros

    window_size: int <= n_fft
        each frame of audio is windowed by `window()` of length 
        `window_size` and then padded with zeros to match `n_fft`

    hop_length: int > 0
        number of audio samples between adjacent STFT columns

    peak: float
        target peak of an audio signal on a decibel scale

    audio_type: {'vocal', 'drums', 'bass', other}
        description of the type of audio signal

    signal_db: np.ndarray [shape=(n,)], real valued
        input signal on a decibel scale

    x_norm: p.ndarray [shape=(n,)], real valued
        input signal normalized based on target peak

    fft: np.ndarray [shape=(1 + n_fft/2,n_frames)], real valued
        STFT of input signal

    num_bins: int > 0
        number of frequency bins for fft

    fft_db: np.ndarray [shape=(1 + n_fft/2,n_frames)], real valued
        STFT of input signal on a decibel scale

    norm_fft_db: np.ndarray [shape=(1 + n_fft/2,n_frames)], real valued
        normalized STFT of input signal on a decibel scale

    freqs: np.ndarray [shape=(num_bins)], float
        numpy array of frequencies represented by fft transform 
    """
    def __init__(self, signal, sr, n_fft, window_size, hop_length, peak, 
                audio_type):
        self.signal = signal
        self.sr = sr
        self.n_fft = n_fft
        self.window_size = window_size
        self.hop_length = hop_length
        self.peak = peak
        self.audio_type = audio_type
        self.signal_db = librosa.amplitude_to_db(self.signal)
        self.x_norm = preprocessing.normalize(self.signal, self.peak)
        self.fft = np.abs(librosa.core.stft(self.signal, 
                                            n_fft=self.n_fft, 
                                            win_length=self.window_size,
                                            hop_length=self.hop_length))
        self.num_bins = self.fft.shape[0]
        self.fft_db = librosa.amplitude_to_db(self.fft)
        self.norm_fft_db = preprocessing.compute_norm_fft_db(self.x_norm, 
                                self.n_fft, self.window_size, 
                                self.hop_length)
        self.freqs = np.array([i * self.sr / self.fft.shape[0] 
                                for i in range(self.num_bins)])


class EQSignal(Signal):
    """
    A class that represents an audio signal that requires equalization.

    Attributes
    ----------
    Base Class: Signal
        reference Signal class for base class attributes

    rank_threshold: int > 0
        threshold for whether a frequency bin is considered essential

    max_n: int > 0
        max number of filters for equalization

    max_eq: float
        max amount of attentuation for equalization

    fft_db_avg: np.ndarray [shape=(num_bins,)], real valued
        an average of energy across fft frequency bins on a decibel 
        scale

    rank: np.ndarray [shape=(num_bins,)], real valued
        rank of energy in averaged fft from largest to smallest

    masker_rank_vec: np.ndarray [shape=(num_bins,)], boolean
        vector that returns 1 if the rank of a frequency bin is 
        considered nonessential and 0 if the bin is considered essential
   
    maskee_rank_vec: np.ndarray [shape=(num_bins,)], boolean
        vector that returns 1 if the rank of a frequency bin is 
        considered essential and 0 if the bin is considered nonessential        

    Methods
    ----------
    compute_mask(signal)
        computes the amount of spectral masking between two EQSignal
        objects

    eq_params(signals)
        computes the equalization parameters for an EQSignal based on
        its spectral masking between a list of EQSignals

    equalization(eq_info, Q)
        equalizes an input signal based on a set of filter parameters
    """
    def __init__(self, sr, signal, n_fft, window_size, hop_length, 
                peak, audio_type, rank_threshold, max_n, max_eq):
        super().__init__(signal, sr, n_fft, window_size, hop_length, 
                            peak, audio_type)
        self.rank_threshold = rank_threshold
        self.max_n = max_n
        self.max_eq = max_eq
        self.fft_db_avg = np.mean(self.fft_db, axis=1)
        self.rank = preprocessing.rank_signal_1d(self.fft_db_avg)
        self.masker_rank_vec = np.where(self.rank > self.rank_threshold, 1, 0)
        self.maskee_rank_vec = np.where(self.rank <= self.rank_threshold, 1, 0)

    def compute_mask(self, signal):
        """
        Computes the amount of spectral masking between two EQSignal
        objects

        Parameters
        ----------
        signal: EQSignal
            an EQSignal object

        Returns
        ----------
        mask_ab: np.ndarray [shape=(num_bins,)], real valued
            array of values that details the amount of spectral masking
            between a masker and maskee signal 
        """
        mask_ab = (self.masker_rank_vec * signal.maskee_rank_vec) \
                    * (self.fft_db_avg - signal.fft_db_avg)
        return mask_ab

    def eq_params(self, signals):
        """
        Computes the equalization parameters for an EQSignal based on
        its spectral masking between a list of EQSignals.
        
        Parameters
        ----------
        signals: list, EQSignal
            a list of EQSignals
        
        Returns
        ----------
        eq_info: nested list [shape=(n, 3)]
            nested list with information for applying each equalization 
            filter
        """
        num_signals = len(signals)
        num_bins = self.num_bins
        max_n = self.max_n
        sr = self.sr
        max_eq = self.max_eq
        mask = np.array([])
        eq_info = []
        for m in range(num_signals):
            mask_ab = self.compute_mask(signals[m])
            mask = np.append(mask, mask_ab, axis=0)
        mask_m = np.zeros(num_bins)
        for b in range(num_bins):
            arr = mask[b::num_bins]
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
        if self.audio_type == "vocal":
            # logic for subtractive EQ with highpass and shelf EQ
            eq_info.append([270, 5, 3]) # low shelf filter
            eq_info.append([100, 0 , 1]) # highpass filter best practice
        return eq_info

    def equalization(self, eq_info, Q):
        """
        Equalizes an input signal based on a set of filter parameters.

        Parameters
        ----------
        eq_info: nested list [shape=(n, 3)]
            nested list with information for applying each equalization 
            filter

        Q: float
            Q factor used to give the width of the equalization filter

        Returns
        ----------
        y: np.ndarray [shape=(n,)], real valued
            output signal
        """
        num_filters = len(eq_info)
        y = self.signal
        for i in range(num_filters):
            freq = float(eq_info[i][0])
            gain = float(eq_info[i][1])
            eq_type = int(eq_info[i][2])
            if eq_type == 0:
                eq = AudioEffectsChain().equalizer(freq, Q, -gain)
                y = eq(y)
            elif eq_type == 1:
                eq = AudioEffectsChain().highpass(freq)
                y = eq(y)
            elif eq_type == 2:
                eq = AudioEffectsChain().lowpass(freq)
                y = eq(y)
            elif eq_type == 3:
                eq = AudioEffectsChain().lowshelf(-gain, freq)

            elif eq_type == 4:
                eq = AudioEffectsChain().highshelf(gain, freq)
        return y


    # def compute_energy_percent(self):
    #     total_energy = np.sum(self.chunk_fft_db)
    #     energy_percents = []
    #     for i in range(len(self.bins)-1):
    #         arr = np.argwhere((self.freqs >= self.bins[i]) & (self.freqs < self.bins[i+1])).flatten()
    #         bin_sum = np.sum([self.chunk_fft_db[i] for i in arr])
    #         energy_percent = bin_sum / total_energy
    #         energy_percents.append(energy_percent)
    #     if energy_percents[0] < 0.2:
    #         return self.bins[1]

    # def compute_rolloff(self):
    #     rolloffs = librosa.feature.spectral_rolloff(self.signal, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, 
    #                          win_length=self.window_size, roll_percent=self.roll_percent)
    #     r_active = rolloffs[0, np.argwhere(rolloffs > 0).flatten()]
    #     r_avg = np.mean(r_active)
    #     return r_avg

    # def compute_mask(self, signal, overlap_vec, num_overlaps):
    #     soa_vec_i = preprocessing.sparse_overlap_avg(self.num_bins, self.chunk_fft_db, 
    #                                                 self.sparse_vec, overlap_vec, num_overlaps)
    #     soa_vec_j = preprocessing.sparse_overlap_avg(signal.num_bins, signal.chunk_fft_db, 
    #                                                 signal.sparse_vec, overlap_vec, num_overlaps)
    #     r_soa_vec_i = preprocessing.rank_soa_vec(soa_vec_i)
    #     r_soa_vec_j = preprocessing.rank_soa_vec(soa_vec_j)
    #     masker_vec_i = preprocessing.masker_rank_vec(r_soa_vec_i)
    #     maskee_vec_j = preprocessing.maskee_rank_vec(r_soa_vec_j)
    #     mask_ij = ((masker_vec_i * maskee_vec_j) * (soa_vec_i - soa_vec_j)).flatten()
    #     m_f = np.concatenate((mask_ij[:, np.newaxis], self.freqs[:, np.newaxis]), axis=1)
    #     return mask_ij 


class CompressSignal(Signal):
    """
    A class that represents an audio signal that requires compression.

    Attributes
    ----------
    Base Class: Signal
        reference Signal class for base class attributes

    order: int
        the order of the butterworth filter used for low frequency
        energy calculation

    cutoff: float
        critical frequency for filter computing low frequency energy

    std: float
        standard deviation

    attack_max: float
        max time for attack

    release_max: float
        max time for release

    Methods
    ----------
    compute_wp(cf_avg)
        computes the weighted percussivity according to a crest factor
        average

    compute_lf_weighting(lfa)
        computes the low frequency weighting based on a low frequency
        average

    ratio(wf, wp)
        computes the ratio parameter based on its low frequency energy
        and weighted percussivity

    threshold(wp)
        computes the threshold parameter based on its weighted
        percussivity

    knee_width(threshold)
        computes the knee_width parameter based on its threshold

    attack()
        computes the attack parameter for compression

    release()
        computes the release parameter for compression

    comp_params(cfa, lfa)
        computes the parameters for the compression of an input signal
        based on its crest factor and low frequency average

    compression(params)
        Compresses an input signal
    """
    def __init__(self, signal, sr, n_fft, window_size, hop_length, peak, audio_type, 
                order, cutoff, std, attack_max, release_max):
        super().__init__(signal, sr, n_fft, window_size, hop_length, peak, audio_type)
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
    
    def compute_wp(self, cf_avg): 
        """
        Computes the weighted percussivity according to a crest factor
        average

        Parameters
        ----------
        cf_avg: float
            crest factor average - measure of transient nature of
            group of signals

        Returns
        ----------
        wp: float
            weighted percussivity of an audio signal - measure of
            percussivity of weighted by the crest factor average 
        """
        wp = preprocessing.wp(self.crest_factor, cf_avg, self.std)
        return wp

    def compute_lf_weighting(self, lfa): 
        """
        Computes the low frequency weighting based on a low frequency
        average.

        Parameters
        ----------
        lfa: float
            low frequency average - measure of low frequency energy
            across audio signals

        Returns
        ----------
        lfw: float
            low frequency weighting based on low frequency average
        """       
        lfw = self.lfe / lfa
        return lfw

    def ratio(self, wf, wp): 
        """
        Computes the ratio parameter based on its low frequency energy
        and weighted percussivity.

        Parameters
        ----------
        wf: float
            low frequency weighting

        wp: float
            weighted percussivity

        Returns
        ----------
        r: float
            ratio for compression
        """
        r = float(0.54*wp + 0.764*wf + 1)
        return r

    def threshold(self, wp):
        """
        Computes the threshold parameter based on its weighted
        percussivity.

        Parameters
        ----------
        wp: float
            weighted percussivity        

        Returns
        ----------
        t: float
            threshold for compression

        """
        t =  float(-11.03 + 0.44*self.rms_db - 4.897*wp)
        return t

    def knee_width(self, threshold): 
        """
        Computes the knee_width parameter based on its threshold. 
        
        Parameters
        ----------
        threshold: float
            level threshold that activates compression

        Returns
        ----------
        kw: float
            knee width - determines smoothness of compression

        """
        kw = abs(threshold) / 2
        return 

    def attack(self): 
        """
        Computes the attack parameter for compression

        Returns
        ----------
        a: float
            attack parameter that determines speed at which compression
            begins after level is above threshold
        """
        a = float((2*self.attack_max) / self.crest_factor ** 2)
        return a

    def release(self): 
        """
        Computes the release parameter for compression
        
        Returns
        ----------
        rel:
            relaease parameter that determines the speed at which
            compression ends after level is below threshold
        """
        rel = float((2*self.release_max) / self.crest_factor ** 2)
        return rel

    def comp_params(self, cfa, lfa):
        """
        Computes the parameters for the compression of an input signal
        based on its crest factor and low frequency average.

        Parameters
        ----------
        cfa: float
            crest factor average

        lfa: float
            low frequency average

        Returns
        ----------
        params: list, float
            list of compression parameters to be used for audio effect
        """
        w_p = self.compute_wp(cfa)
        w_f = self.compute_lf_weighting(lfa)
        r = self.ratio(w_f, w_p)
        t = self.threshold(w_p)
        kw = self.knee_width(t)
        a = self.attack()
        rel = self.release()
        params = [t, r, a, rel, kw]
        return params

    def compression(self, params):
        """
        Compresses an input signal.
        
        Parameters
        ----------
        params: list, float
            list of compression parameters to be used for audio effect

        Returns
        ----------
        y_out: np.ndarray [shape=(n,)], real valued
            output signal after compression
        """
        compress = AudioEffectsChain().compand(attack=params[2], decay=params[3], soft_knee=params[1], 
                    threshold=params[0], db_from=params[0], db_to=params[0])
        y = compress(self.signal)
        makeup_gain = preprocessing.compute_makeup_gain(self.signal, y, self.sr)
        gain = (AudioEffectsChain().gain(makeup_gain))
        y_out = gain(y)
        return y_out

class FaderSignal(Signal):
    def __init__(self, signal, sr, n_fft, window_size, hop_length, peak, audio_type,
                decay, step, lead, max_fader, min_fader, B):
        super().__init__(signal, sr, n_fft, window_size, hop_length, peak, audio_type)
        self.decay = decay
        self.step = step
        self.lead = lead
        self.max_fader = max_fader
        self.min_fader = min_fader
        self.B = B
    
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
        # ind = np.where(~np.isnan(L_m))[0] # temp fix until figuring out why function returns NaN values
        # first, last = ind[0], ind[-1]
        # L_m[:first] = L_m[first]
        # L_m[last + 1:] = L_m[last]
        mask = np.isnan(L_m)
        L_m[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), L_m[~mask])
        return L_m

    def compute_fader(self, L_av, L2):
        F_m = np.zeros(L2.shape)
        for n in range(len(F_m)):
            F_m[n] = 10 ** ((L_av[n] - L2[n]) / 20)
        if self.lead == True:
            F_m[n] = F_m[n] * 10 ** (self.B/20)

        F_m = np.where(F_m > self.max_fader, self.max_fader, F_m)
        F_m = np.where(F_m < self.min_fader, self.min_fader, F_m)
        return F_m

    def fader(self, fader_output): return self.signal * fader_output

class PanSignal(Signal):
    def __init__(self, signal, sr, n_fft, window_size, hop_length, peak, audio_type, 
                cutoffs, window, order, btype):
        super().__init__(signal, sr, n_fft, window_size, hop_length, peak, audio_type)
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


    def pan(self, P):
        # quick fix - should be done from the signal aggregator function
        if self.audio_type == "vocal":
            P = 0.5
        left = np.cos(P * np.pi) * self.signal
        right = np.sin(P * np.pi) * self.signal
        return np.dstack((left,right))[0]

class DeEsserSignal(Signal):
    def __init__(self, signal, sr, n_fft, window_size, hop_length, peak, audio_type,
                critical_bands, c, sharp_thresh, max_reduction):
        super().__init__(signal, sr, n_fft, window_size, hop_length, peak, audio_type)
        self.critical_bands = critical_bands
        self.bark_idx = preprocessing.freq_bark_map(self.freqs, self.critical_bands)
        self.c = c
        self.sharp_thresh = sharp_thresh
        self.max_reduction = max_reduction
        self.cb_fft = preprocessing.critical_band_sum(self.fft, self.bark_idx, len(critical_bands))
        self.N_z = preprocessing.compute_Nz(self.cb_fft, self.critical_bands)
        self.g_z = np.apply_along_axis(np.exp, 0, (0.171*self.cb_fft))

    def compute_sharpness(self):
        numr = np.sum(self.N_z*self.g_z, axis=0)
        denom = np.sum(self.N_z, axis=0)
        S = self.c * (numr / (denom+1e-9))
        return S

    def compute_zcr(self):
        y0 = preprocessing.apply_bfilter(self.signal, 60, self.sr, 1, 'highpass')
        y1 = preprocessing.apply_bfilter(y1, 600, self.sr, 1, 'lowpass')
        zcr = librosa.feature.zero_crossing_rate(y1, frame_length=self.window_size, 
                                                hop_length=self.hop_length)
        return zcr

    def compute_ste(self, rab):
        N = self.window_size
        frames = librosa.util.frame(self.signal, frame_length=self.window_size,
                                    hop_length=self.hop_length)
        if rab:
            idx = np.array(range(256))
            hn = 0.54 - 0.46 * np.cos(2*np.pi * idx / (N-1))
            ste = np.sum(frames*hn, axis=0, keepdims=True)
            return ste
        ste = np.mean(np.abs(frames)**2, axis=0, keepdims=True)
        return ste

    def gain_reduction(self, sharpness):
        N = sharpness.shape[0]
        sharpness
        gain = np.ones(N)
        for n in range(N):
            if sharpness[n] > self.sharp_thresh:
                gain[n] = 1 + (np.log(sharpness[n]) / np.log(0.1))
                if gain[n] < self.max_reduction:
                    gain[n] = self.max_reduction

        return gain

    def deesser(self, gain):
        y_out = np.zeros(self.signal.shape)
        frame_sig = librosa.util.frame(self.signal, frame_length=self.n_fft, hop_length=self.hop_length)
        M, N = frame_sig.shape[0], frame_sig.shape[1]
        # fix padding that makes these dimensions not always aligned via stft
        # quick fix is a truncation of array 
        y = np.zeros((M, N))
        for m in range(M):
            y[m] = gain[m] * frame_sig[m]
        y = y.flatten('F')
        n = y.shape[0]
        y_out[:n] = y
        return y_out


class PitchCorrectionSignal(Signal):
    def __init__(self, signal, sr, n_fft, window_size, hop_length, peak, audio_type):
        super().__init__(signal, sr, n_fft, window_size, hop_length, peak, audio_type)
        pass


class ReverbSignal(Signal):
    def __init__(self, signal, sr,n_fft, window_size, hop_length, peak, audio_type,
                reverbance, hf_damping, room_scale, wet_gain, effect_percent, hp_freq, lp_freq, order,
                stereo_depth, pre_delay):
        super().__init__(signal, sr, n_fft, window_size, hop_length, peak, audio_type)
        self.reverbance = reverbance
        self.hf_damping = hf_damping
        self.room_scale = room_scale
        self.wet_gain = wet_gain
        self.effect_percent = effect_percent
        self.hp_freq = hp_freq
        self.lp_freq = lp_freq
        self.order = order
        self.stereo_depth = stereo_depth
        self.pre_delay = pre_delay
        self.effect_signal = preprocessing.compute_effect_signal(self.signal, self.effect_percent,
                                                                self.hp_freq, self.lp_freq, self.order, self.sr)
        self.dry_signal = self.signal * (1 - self.effect_percent)

    def reverb(self):
        fx = AudioEffectsChain().reverb(self.reverbance, self.hf_damping, self.room_scale, self.stereo_depth,
                                        self.pre_delay, self.wet_gain)
        y_fx = fx(self.effect_signal)
        y_out = self.dry_signal + y_fx
        return y_out


class SignalAggregator:
    """
    Computes all of the aggregation based stats for audio effects.
    
    Attributes
    ----------
    sr: float
        sample rate

    M: int > 0
        number of audio signals

    Methods
    ----------
    lfa(lfes)
        computes the low frequency average

    cfa(cfs)
        computes the crest factor average

    panning_locations(filter_freqs, signal_peaks)
        computes the panning location for each audio signal

    loudness_avg(channels, holdtime, ltrhold, utrhold, release, attack)
        computes the loudness average
    """
    def __init__(self, sr, M):
        self.sr = sr
        self.M = M

    def lfa(self, lfes): 
        """
        Computes the low frequency average.

        Parameters
        ----------

        lfes: list [len(M)], float
            list of low frequency energy calculations for M audio 
            signals

        Returns
        ----------
        lfa: float
            low frequency average for audio signals
        """
        lf_avg = sum([lfe for lfe in lfes]) / self.M
        return lf_avg

    def cfa(self, cfs): 
        """
        Computes the crest factor average.
        
        Parameters
        ----------
        cfs: list [len(M)], float
            list of crest factor calculations for M audio signals

        Returns
        ----------
        cf_avg
            crest factor average for audio signals
        """
        cf_avg = sum([cf for cf in cfs]) / self.M
        return cf_avg

    def panning_locations(self, signal_peaks):
        """
        Computes the panning location for each audio signal.

        Parameters
        ---------- 
        signal_peaks:
            List of signal peaks within filter banks for signals

        Returns
        ----------

        Ps: nested list [(M, M)]
            nested list of panning locations
        """
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
                    P[i] = 1 - ((N_k - i) / (2 * (N_k -1)))
                idx = np.argwhere(signal_peaks == freq)[i][0]
                Ps.append([idx, P[i]])
        return Ps

    def loudness_avg(self, channels, holdtime, ltrhold, utrhold, 
                    release, attack):
        """
        Computes the loudness average.

        Parameters
        ----------

        channels: list, np.ndarray [shape=(n,)], real valued
            loudness per audio signal

        holdtime: float
            holdtime for noise gate

        ltrhold: float
            lower threshold for noise gate

        utr: float
            upper threshold for noise gate

        release: float > 0
            release for noise gate

        attack: float > 0
            attack for noise gate 

        Returns
        ----------

        L_av: np.ndarray [shape=(n,)]
            Loudness average for each sample
        """
        gains = [preprocessing.noise_gate(channels[i], holdtime, 
                ltrhold, utrhold, release, attack, self.sr) 
                 for i in range(len(channels))]
        gains = [np.where(gain < 1, 0, 1) for gain in gains]
        gain_val = np.array(gains).sum(axis=0)
        L_g = [channel * gain for channel, gain in zip(channels, gains)] 
        L_c = np.array(L_g).sum(axis=0)
        # L_av = np.where(gain_val > 0, L_c / gain_val, -50)
        L_av = np.ones(gain_val.shape[0]) * -30
        np.divide(L_c, gain_val, out=L_av, where=gain_val != 0)
        # apply ema filter
        return L_av


class Mixer:
    def __init__(self, signals, output_path, sr):
        self.signals = signals
        self.output_path = output_path
        self.sr = sr

    def mix(self): 
        output = np.sum(self.signals, axis=0)
        return output.astype(np.float32)

    def output_wav(self, mixed_file): 
        write(self.output_path, self.sr, mixed_file)


class Track:
    def __init__(self, track, sr):
        self.track = track
        self.sr = sr

    def calculate_peak(self):
        peak = self.track.max()
        peak_db = librosa.amplitude_to_db([peak])
        return peak_db[0]

    def calculate_loudness(self):
        meter = pyln.Meter(self.sr)
        loudness = meter.integrated_loudness(self.track)
        return loudness

    def calculate_rms(self):
        rms_frame = librosa.feature.rms(self.track)
        rms = np.mean(rms_frame, axis=1)
        rms_db = librosa.amplitude_to_db(rms)
        return rms_db[0]
