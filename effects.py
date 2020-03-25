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
    pass


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