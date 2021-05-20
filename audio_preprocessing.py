import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

from matplotlib.mlab import window_hanning, specgram


nfft = 1024 #256#1024 #NFFT value for spectrogram
overlap = 1000 #512 #overlap value for spectrogram


class Transforms():
    def __init__(self, *args, **kwargs):
        super(Transforms, self).__init__(*args, **kwargs)



    def get_spectrogram(self, waveform, sample_rate):

        n_fft = 1024
        win_length = None
        hop_length = 512

        # Define transformation
        spectrogram = T.Spectrogram(
            n_fft = n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0
        )

        #Perform transformation
        return spectrogram(waveform)

    def get_specgram(self, signal, rate):
        arr2D,freqs,bins = specgram(signal, window=window_hanning,
                                   Fs = rate, NFFT=nfft, noverlap=overlap)
        return arr2D,freqs,bins

    
