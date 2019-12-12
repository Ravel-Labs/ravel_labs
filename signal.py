import os
import librosa
import numpy as np
import pyloudnorm as pyln
import scipy
from scipy.fftpack import fft
from scipy.stats import rankdata

class Signal:
	def __init__(self, path, window_size, hop_length):
		self.path = path
		self.sr = librosa.get_samplerate(self.path)
		self.window_size = window_size
		self.hop_length = hop_length
		self.signal, _ = librosa.load(self.path, sr=self.sr)
		self.fft = np.abs(librosa.core.stft(self.signal, n_fft=self.window_size, hop_length=self.hop_length))
		self.freq_bins = self.fft.shape[0]
		self.fft_db = librosa.amplitude_to_db(self.fft)

	def set_chunk(self, seconds):
		fft_length = self.fft_db.shape[1]
		num_freqs = self.fft_db.shape[0]
		chunk_size = int(np.ceil((1 / (self.window_size / self.sr)) * seconds))
		total_chunks = int(np.ceil(fft_length / chunk_size))
		avg_mat = np.zeros((num_freqs, total_chunks))
		avg_vec = np.zeros((1, chunk_size))
		for i in range(num_freqs):
			for j in range(total_chunks):
				if j > total_chunks - 1:
					avg_vec = self.fft_db[i][chunk_size * j:]
					mu = np.mean(avg_vec)
					avg_mat[i][j] = mu
				avg_vec = self.fft_db[i][chunk_size * j: chunk_size * (j+1)]
				mu = np.mean(avg_vec)
				avg_mat[i][j] = mu
		self.chunk_fft = avg_mat

	def set_rank_2d(self): 
		a = np.zeros(self.chunk_fft.shape)
		for row in range(self.chunk_fft.shape[1]):
			a[:, row] = np.abs(rankdata(self.chunk_fft[:, row], method='min') - (self.chunk_fft.shape[0])) + 1
		self.rank = a

	def set_sparsity(self):
		sparse_vec = np.zeros((1, self.rank.shape[1]))
		for i in range(self.rank.shape[1]):
			mu = np.mean(self.rank.T[i])
			if mu == self.freq_bins:
				sparse_vec[0, i] = 0
			else:
				sparse_vec[0, i] = 1
		self.sparse_vec = sparse_vec

	def overlap(self, sv1):
		overlap_vec = self.sparse_vec * sv1
		num_overlaps = np.sum(overlap_vec)
		overlap_ratio = num_overlaps / overlap_vec.shape[1]
		return overlap_vec, num_overlaps, overlap_ratio

	def sparse_overlap_avg(self, overlap_vec, num_overlaps):
		soa_vec = np.zeros((self.freq_bins, 1))
		for i in range(self.freq_bins):
			soa_vec[i] = np.sum((self.chunk_fft * self.sparse_vec) * overlap_vec) / num_overlaps
		return soa_vec

	def rank_soa_vec(self, soa_vec): return np.abs(rankdata(soa_vec, method='min') - (soa_vec.shape[0])) + 1

	def masker_rank_vec(self, rank_soa_vec): np.expand_dims(np.where(rank_soa_vec > 10, 1, 0), axis=1)

	def maskee_rank_vec(self, rank_soa_vec): np.expand_dims(np.where(rank_soa_vec <= 10, 1, 0), axis=1)