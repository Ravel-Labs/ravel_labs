from pyo import *

class Signal:
    def __init__(self, path, signal, n_fft, window_size, hop_length, R):
        self.path = path
        self.sr = librosa.get_samplerate(self.path)
        self.n_fft = n_fft
        self.window_size = window_size
        self.hop_length = hop_length
        self.signal = signal
        self.R = R
        self.fft = np.abs(librosa.core.stft(self.signal, n_fft=self.n_fft, 
                                            win_length=self.window_size, hop_length=self.hop_length))
        self.num_bins = self.fft.shape[0]
        self.fft_db = librosa.amplitude_to_db(self.fft)
        self.norm_fft_db = compute_norm_fft_db(self.signal, self.R, 
                                                self.n_fft, self.window_size, self.hop_length)
        self.freqs = np.array([i * self.sr / self.fft.shape[0] for i in range(self.num_bins)])


class EQSignal(Signal):
    def __init__(self, path, signal, n_fft, window_size, hop_length, R, bins, roll_percent, seconds,
                rank_threshold, max_n, min_overlap_ratio, max_eq):
        super().__init__(path, signal, n_fft, window_size, hop_length, R)
        self.bins = bins
        self.roll_percent = roll_percent
        self.chunk_fft_db = compute_chunk(self.norm_fft_db, self.window_size, self.sr, self.seconds)
        self.rank = compute_rank(self.chunk_fft_db)
        self.sparse_vec = compute_sparsity(self.rank, self.num_bins)
        self.rank_threshold = rank_threshold
        self.max_n = max_n,
        self.min_overlap_ratio = min_overlap_ratio
        self.max_eq = max_eq

    def compute_energy_percent(self):
        total_energy = np.sum(self.chunk_fft)
        energy_percents = []
        for i in range(len(self.bins)-1):
            arr = np.argwhere((self.freqs >= self.bins[i]) & (self.freqs < self.bins[i+1])).flatten()
            bin_sum = np.sum([self.chunk_fft[i] for i in arr])
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
        soa_vec_i = sparse_overlap_avg(self.num_bins, self.chunk_fft_db, self.sparse_vec, overlap_vec, num_overlaps)
        soa_vec_j = sparse_overlap_avg(signal.num_bins, signal.chunk_fft_db, signal.sparse_vec, overlap_vec, num_overlaps)
        r_soa_vec_i = rank_soa_vec(soa_vec_i)
        r_soa_vec_j = rank_soa_vec(soa_vec_j)
        masker_vec_i = masker_rank_vec(r_soa_vec_i)
        maskee_vec_j = maskee_rank_vec(r_soa_vec_j)
        mask_ij = ((masker_vec_i * maskee_vec_j) * (soa_vec_i - soa_vec_j)).flatten()
        m_f = np.concatenate((mask_ij[:, np.newaxis], self.freqs[:, np.newaxis]), axis=1)
        return mask_ij      

    def eq_params(self, signals):
        num_signals = len(signals)
        for i in range(num_signals):
            mask = np.empty(shape=[0, 2])
            eq_info = []
            overlap_vec, num_overlaps, overlap_ratio = overlap(self.sparse_vec, signals[i].sparse_vec)
            if (overlap_ratio > min_overlap_ratio) and (i != j):
                mask_ij = self.compute_mask(signals[i], overlap_vec, num_overlaps)
                m_f = np.concatenate((mask_ij[:, np.newaxis], signals[i].freqs[:, np.newaxis]), axis=1)
                mask = np.append(mask, m_f, axis=0)
            else:
                mask_ij = 0
        mask_m = np.zeros(num_bins)
        for b in range(num_bins):
            arr = mask[b::num_bins, :]
            max_b, _ = np.max(arr, axis=0)
            mask_m[b] = max_b
        top_m = np.argsort(mask_m)[-max_n:]
        top_m_max = mask_m[top_m].max()
        idx = np.unravel_index(top_m, mask_m.shape)[0]
        for x in idx:
            freq_bin = x  * (sr / num_bins)
            mask_val = mask_m[x]
            if (mask_val) > 0 and (freq_bin <= 20000) and (freq_bin >= 20):
                mask_val_scaled = (mask_val / top_m_max) * max_eq
                eq_type = 0
                eq_info.append([freq_bin, mask_val_scaled, eq_type])
        rolloff = self.compute_rolloff()
        energy_percent = self.compute_energy_percent()
        eq_info.append([rolloff, 0.71, 2])
        if energy_percent is not None:
            eq_info.append([energy_percent, 0.71, 1])
        # params_list.append({signals[i].path: eq_info})
        return eq_info

    def equalization(self, eq_info, Q):
        num_filters = len(eq_info)
        s = Server.boot()
        c = Converter(self.signal)
        out = c.numpy_to_pyo(s)
        for i in range(num_filters):
            freq = float(eq_info[i][0])
            gain = float(eq_info[i][1])
            eq_type = int(eq_info[i][2])
            out = EQ(out, freq=freq, q=Q, boost=-gain, type=eq_type)
        out = out.out()
        numpy_out = c.pyo_to_numpy(out, s)
        return numpy_out


class CompressSignal(Signal):
    def __init__(self, path, signal, n_fft, window_size, hop_length, R, 
                time_constant, order, cutoff, std, attack_max, release_max):
    super().__init__(path, signal, n_fft, window_size, hop_length, R)
    self.time_constant = time_constant
    self.order = order
    self.cutoff = cutoff
    self.std = std
    self.attack_max = attack_max
    self.release_max = release_max
    self.crest_factor = preprocessing.cf(self.signal)
    self.rms = librosa.feature.rms(signal, frame_length=1024, hop_length=512)
    self.rms_db = np.mean(librosa.amplitude_to_db(self.rms))
    
    def compute_wp(self, cf_avg):
        return preprocessing.wp(self.crest_factor, cf_avg, self.std)

    def compute_lf_weighting(self, lfa):
        return preprocessing.lf_weighting(self.signal, lf_avg, self.order, self.cutoff, self.sr)

    def ratio(self, wf, wp): return float(0.54*wp + 0.764*wf + 1)

    def threshold(self, wp):return float(-11.03 + 0.44*self.rms_db - 4.897*wp)

    def knee_width(self, threshold): return abs(threshold) / 2

    def attack(self):
        return float((2*self.attack_max) / self.crest_factor ** 2)

    def release(self):
        return float((2*self.release_max) / self.crest_factor ** 2)

    def comp_params(self, cfa, lfa):
        w_p, cf = self.compute_wp(cfa)
        w_f = self.compute_lf_weighting(lfa)
        rms = librosa.feature.rms(signal, frame_length=1024, hop_length=512)
        r = self.ratio(wf, wp)
        t = self.threshold(wp)
        kw = self.knee_width(t)
        a = self.attack()
        rel = self.release()
        return [t, r, a, rel, kw]

    def compression(self, params):
        s = Server.boot()
        c = Converter(self.signal)
        out = c.numpy_to_pyo(s)
        # out = SfPlayer(full_file_path)
        out = Compress(out, thresh=params[0], ratio=params[1], risetime=params[2], falltime=params[3], knee=0.4).out()
        numpy_out = c.pyo_to_numpy(out, s)


        # outp, rate = sf.read(numpy_out)
        # inp, _ = sf.read(file)
        meter = pyln.Meter(self.sr)
        out_l = meter.integrated_loudness(numpy_out)
        inp_l = meter.integrated_loudness(self.signal)
        makeup_gain = inp_l - out_l
        compressed_signal = AudioSegment.from_wav(file)
        compressed_signal = compressed_signal + makeup_gain
        compressed_signal.export(filename, format="wav")

class PanSignal(Signal):
    def __init__(self, path, signal, n_fft, window_size, hop_length, R, 
                cutoffs, window, order, btype)
    super().__init__(path, signal, n_fft, window_size, hop_length, R)
    self.window = window
    self.order = order
    self.btype = btype
    self.cutoffs = cutoffs
    self.window_step = int(self.sr * self.window)
    self.num_steps = int(self.signal.shape[0] / self.window_step)
    self.K = len(cutoffs)


    def lead_filter(self): 
        return peak_filter_bank(self.signal, self.cutoffs, 
                                self.sr, self.order, self.btype, 
                                self.window_step, self.num_steps)

    def pan(self, P):
        s = Server.boot()
        c = Converter(self.signal)
        out = c.numpy_to_pyo(s)
        out = Pan(out, pan=P).out()
        numpy_out = c.pyo_to_numpy(out, s)
        return numpy_out



class Converter:
    def __init__(self, signal):
        self.signal = signal
        self.buffer_size = signal.shape[0]

    def numpy_to_pyo(self, s):
        s.start()
        t = DataTable(size=self.buffer_size)
        osc = TableRead(t, freq=t.getRate(), loop=True, mul=0.1).out()
        arr = np.asarray(t.getBuffer())
        pyo_x = process(arr, self.signal, osc)
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
    pass


def compute_chunk(norm_fft_db, window_size, sr, seconds):
    fft_length = norm_fft_db.shape[1]
    num_freqs = norm_fft_db.shape[0]
    chunk_size = int(np.ceil((1 / (window_size / sr)) * seconds))
    total_chunks = int(np.ceil(fft_length / chunk_size))
    avg_mat = np.zeros((num_freqs, total_chunks))
    avg_vec = np.zeros((1, chunk_size))
    for i in range(num_freqs):
        for j in range(total_chunks):
            if j > total_chunks - 1:
                avg_vec = norm_fft_db[i][chunk_size * j:]
                mu = np.mean(avg_vec)
                avg_mat[i][j] = mu
            avg_vec = norm_fft_db[i][chunk_size * j: chunk_size * (j+1)]
            mu = np.mean(avg_vec)
            avg_mat[i][j] = mu
    return avg_mat

def compute_rank(chunk_fft_db): 
    a = np.zeros(chunk_fft_db.shape)
    for row in range(chunk_fft_db.shape[1]):
        a[:, row] = np.abs(rankdata(chunk_fft_db[:, row], method='min') - (chunk_fft_db.shape[0])) + 1
    return a

def compute_sparsity(rank, num_bins):
    sparse_vec = np.zeros((1, rank.shape[1]))
    min_val = num_bins
    for i in range(rank.shape[1]):
        mu = np.mean(rank.T[i])
        if mu == min_val:
            sparse_vec[0, i] = 0
        else:
            sparse_vec[0, i] = 1
    return sparse_vec

def overlap(sv0, sv1):
    overlap_vec = sv0 * sv1
    num_overlaps = np.sum(overlap_vec)
    overlap_ratio = num_overlaps / overlap_vec.shape[1]
    return overlap_vec, num_overlaps, overlap_ratio

def sparse_overlap_avg(num_bins, chunk_fft_db, sparse_vec, overlap_vec, num_overlaps):
    soa_vec = np.zeros((num_bins, 1))
    for i in range(num_bins):
        soa_vec[i] = np.sum((chunk_fft_db[i] * sparse_vec) * overlap_vec) / num_overlaps
    return soa_vec

def peak_filter(signal, cutoff, sr, order, btype, window_step, num_steps):
    y = apply_bfilter(signal, cutoff, sr, order, btype)
    window_step = int(sr * window)
    num_steps = int(signal.shape[0] / window_step)
    peaks = np.zeros(num_steps)
    for i in range(num_steps):
        y_window = y[i*window_step:(i+1)*window_step]
        peak = y_window.max()
        peaks[i] = peak
    return peaks

def peak_filter_bank(signal, cutoffs, sr, order, btype, window_step, num_steps):
    num_cutoffs = len(cutoffs)
    peaks = np.zeros((num_cutoffs, num_steps))
    for i in range(num_cutoffs):
        peaks[i] = peak_filter(signal, cutoffs[i], sr, order, btype, window_step, num_steps)
    maxs = np.argmax(peaks, axis=0)
    freq_counts = np.unique(maxs, return_counts=True)
    max_idx = np.argmax(freq_counts[:][1])
    return cutoffs[max_idx]

def panning_locations(filter_freqs, signal_peaks):
    '''Method used for the signal aggregator'''
    num_freqs = len(filter_freqs)
    N_ks = np.unique(signal_peaks, return_inverse=True, return_counts=True)
    Ps = []
    for k in range(num_freqs):   
        N_k = N_ks[k][1]
        P = np.zeros(N_k, k)
        for i in range(N_k):
            if N_k == 1:
                P[i][k] = 1/2
            if N_k + i % 2 != 0:
                P[i][k] = (N_k - i - 1) / (2 * (N_k - i))
            if (N_k + i % 2 == 0) and (N_k != 1):
                P[i][k] = 1 - ((N_k - i) / (2 * (N_k -1)))
            idx = np.argwhere(signal_peaks == k)[i]
            Ps.append(idx, P[i][k])
    return Ps





def rank_soa_vec(soa_vec): return np.abs(rankdata(soa_vec, method='min') - (soa_vec.shape[0])) + 1

def masker_rank_vec(r_soa_vec): return np.expand_dims(np.where(r_soa_vec > 10, 1, 0), axis=1)

def maskee_rank_vec(r_soa_vec): return np.expand_dims(np.where(r_soa_vec <= 10, 1, 0), axis=1)

def process(arr, x, osc):
    "Fill the array (so the table) with white noise."
    arr[:] = x
    return osc

def done(t):
    arr = np.asarray(t.getBuffer())
    return arr


def compute_norm_fft_db(signal, R, n_fft, window_size, hop_length):
    """Calculate scalar for signal on a linear scale"""
    x = signal
    n = x.shape[0]
    a = np.sqrt((n * R**2) / np.sum(x**2))
    x_norm_db = librosa.amplitude_to_db(a * x)
    norm_fft_db = np.abs(librosa.core.stft(x_norm_db, n_fft=n_fft, 
                                        win_length=window_size, hop_length=hop_length))
    return norm_fft_db