import scipy
from preprocessing import *
from pyo import *


def EQ_signal(wav_file, freqs, Q, gains):
    out = wav_file
    if len(freqs) > 0:
        return out
    for i in range(len(freqs)):
        out = EQ(out, freqs[i], Q, gains[i])
    return out

def EQ_signals():
    '''Iterates all audio signals and EQs them according to their frequences and gains'''
    pass

def mix_signals(audio_signals):
    '''Use Mixer Pyo Object to add the equalized signals to create a finalized output'''
    mixer = Mixer()
    for i, audio_signal in enumerate(audio_signals):
        mixer.addInput(i, audio_signal)
    return mixer